# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[KUNLUN] FlashAttention MLA-prefill backend for P800 XPU.

Drop-in replacement for
``vllm.v1.attention.backends.mla.prefill.flash_attn.FlashAttnPrefillBackend``.
The upstream backend needs ``flash_attn_varlen_func`` (unavailable on Kunlun);
here both the new-token and context-chunk prefill attentions are run through
``kunlun_ops.attention`` (varlen, optionally causal, returns softmax LSE),
mirroring the vllm 0.11.0 adaptation in
``vLLM-Kunlun/vllm_kunlun/v1/attention/backends/mla/common.py``.

Registered as an override for ``MLAPrefillBackendEnum.FLASH_ATTN`` in
``vllm_kunlun/__init__.py::register()`` so ``get_mla_prefill_backend`` returns
this class on P800.
"""

from typing import TYPE_CHECKING

import kunlun_ops
import torch

from vllm.v1.attention.backends.mla.prefill.base import (
    MLADimensions,
    MLAPrefillBackend,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
    from vllm.platforms.interface import DeviceCapability

# [KUNLUN][VERIFY] DeepSeek/Kimi MLA softmax scale (mscale-adjusted); must match
# the value used by the inline prefill in kimi_k3/nvidia/mla.py.
_DS_ALPHA = 1.8738542070926265

class FlashAttnPrefillBackend(MLAPrefillBackend):
    """kunlun_ops-backed MLA prefill backend."""

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        # Kunlun XPU masquerades as CUDA; accept whatever capability is reported.
        return True

    @classmethod
    def supports_mla_dimensions(cls, mla_dimensions: MLADimensions) -> bool:
        return mla_dimensions in [
            MLADimensions(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128),
            MLADimensions(qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256),
            MLADimensions(qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=128),
        ]

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # V is narrower than QK for MLA; pad it to QK width for the kernel.
        self.requires_v_padding = self.qk_head_dim != v_head_dim

    def supports_out(self) -> bool:
        # Padded V yields a qk_head_dim output that cannot be written into a
        # v_head_dim `out`; the caller slices to v_head_dim afterwards.
        return not self.requires_v_padding

    def supports_quant_output(self, quant_key: "QuantKey") -> bool:
        return False
    def _query_start_loc_cpu(self) -> torch.Tensor:
        # MLACommonPrefillMetadata has no query_start_loc_cpu field in this vllm
        # version; derive it (enforce-eager, so no cudagraph capture concerns).
        prefill = self._prefill_metadata
        qsl_cpu = getattr(prefill, "query_start_loc_cpu", None)
        if qsl_cpu is None:
            raise RuntimeError(
                "[KUNLUN] prefill.query_start_loc_cpu missing; the "
                "MLACommonMetadataBuilder.build hook in vllm_kunlun/__init__.py "
                "must attach it at metadata-build time"
            )
        return qsl_cpu

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context_seq_lod_xpu: torch.Tensor,
        context_seq_lod_cpu: torch.Tensor,
        return_softmax_lse: bool = False,
        causal: bool = True,
        softmax_scale: float | None = None,
        context_kvlen_lod_xpu: torch.Tensor | None = None,
        context_kvlen_lod_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        maybe_padded_v = v
        if self.requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )
        attn_out = torch.empty_like(q)
        tp_q_head_num = q.size(1)
        softmax_lse = torch.full(
            (tp_q_head_num, q.size(0)),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        kunlun_ops.attention(
            q=q,
            k_cache=k,
            v_cache=maybe_padded_v,
            out=attn_out,
            is_causal=causal,
            is_prefill=True,
            prefill_len=0,
            k_perchannel_scale=None,
            v_perchannel_scale=None,
            smooth=None,
            # context_seq_lod = query LOD; context_kvlen_lod = key/value LOD.
            # For new-token self-attention they are identical, so kvlen is left
            # None. For the context chunk (cross-attention) K/V length differs
            # from Q length, so the caller passes the context cu_seq_lens here;
            # omitting it makes the kernel read K/V with the query LOD and hang.
            context_seq_lod_cpu=context_seq_lod_cpu,
            context_seq_lod_xpu=context_seq_lod_xpu,
            slot_mapping_cpu=None,
            slot_mapping_xpu=None,
            v_trans=False,
            v_trans_threshold=0,
            alpha=_DS_ALPHA,
            softmax_lse=softmax_lse,
            unpadded_lse=True,
        )
        if return_softmax_lse:
            return attn_out, softmax_lse
        return attn_out

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        prefill = self._prefill_metadata
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            context_seq_lod_xpu=prefill.query_start_loc,
            context_seq_lod_cpu=self._query_start_loc_cpu(),
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefill = self._prefill_metadata
        assert prefill.chunked_context is not None
        # K/V are the gathered context for this chunk, laid out per the chunk's
        # context cu_seq_lens (not the query LOD). Pass it as context_kvlen_lod so
        # the kernel segments K/V by context length instead of query length.
        kv_lod_xpu = prefill.chunked_context.cu_seq_lens[chunk_idx]
        kv_lod_cpu = getattr(prefill.chunked_context, "cu_seq_lens_cpu", None)
        if kv_lod_cpu is not None:
            kv_lod_cpu = kv_lod_cpu[chunk_idx]
        else:
            kv_lod_cpu = kv_lod_xpu.cpu()
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            context_seq_lod_xpu=prefill.query_start_loc,
            context_seq_lod_cpu=self._query_start_loc_cpu(),
            softmax_scale=self.scale,
            causal=False,  # context is unmasked
            return_softmax_lse=True,
            context_kvlen_lod_xpu=kv_lod_xpu,
            context_kvlen_lod_cpu=kv_lod_cpu,
        )
