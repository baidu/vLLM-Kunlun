# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Kunlun (P800) port of vllm/v1/attention/backends/mla/flashattn_mla.py.
#
# The upstream file hard-depends on the NVIDIA FlashAttention MLA kernels
# (``vllm.vllm_flash_attn`` -> ``_vllm_fa2_C`` / ``_vllm_fa3_C``) which do not
# exist on Kunlun. Here we keep the exact backend/metadata/impl class structure
# so the vLLM MLA machinery (MLACommonBackend / MLACommonMetadataBuilder /
# MLACommonImpl) still wires up, but:
#   * drop the ``vllm.vllm_flash_attn`` / ``fa_utils`` imports and the
#     ``flash_attn_supports_mla`` gating,
#   * ``forward_mqa`` (decode) now calls the Kunlun paged-attention op via
#     ``vllm_kunlun.ops.attention.flashmla.flash_mla_with_kvcache``.

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_kunlun.ops.attention.flashmla import flash_mla_with_kvcache

logger = init_logger(__name__)


class FlashAttnMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLAMetadataBuilder"]:
        return FlashAttnMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashAttnMLAImpl"]:
        return FlashAttnMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Kunlun masquerades as CUDA; do not gate on Hopper (major == 9).
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # Kunlun provides its own MLA decode kernel; no FA-MLA capability probe.
        return None


@dataclass
class FlashAttnMLADecodeMetadata(MLACommonDecodeMetadata):
    query_start_loc: torch.Tensor
    max_query_len: int
    max_seq_len: int
    # Slice of the builder's persistent pinned buffer holding the per-request
    # context lengths for `paged_attention`'s `context_lens_cpu`. Filled in
    # `_build_decode`, i.e. outside the cuda graph capture region.
    seq_lens_cpu: torch.Tensor
    # Kept for API compatibility with the upstream metadata; unused on Kunlun.
    scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0


@dataclass
class FlashAttnMLAMetadata(MLACommonMetadata[FlashAttnMLADecodeMetadata]):
    pass


class FlashAttnMLAMetadataBuilder(MLACommonMetadataBuilder[FlashAttnMLAMetadata]):
    # NOTE(kunlun): `paged_attention` takes the per-request context lengths as
    # int32 twice, once on the host and once on the device (`context_lens_cpu` /
    # `context_lens_xpu`). Only the device copy drives the kernel -- probed
    # standalone: changing the host values leaves the output bit-identical, and a
    # captured graph follows in-place updates of the device tensor on replay. The
    # device side therefore needs no staging (vLLM's `seq_lens` is already a
    # persistent int32 buffer); the host copy is produced in `_build_decode`,
    # outside the capture region, because the `.cpu()` that used to produce it
    # per step is an illegal sync during capture.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN
    # Upstream routes query_len <= 512 through the decode (MQA) pathway. Kunlun's
    # `paged_attention` only computes the *first* query token of each request
    # (verified standalone: it either returns ret != 0, or leaves rows 1..n-1 of
    # the output at zero), so anything longer than one token must go through the
    # varlen prefill path instead.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            FlashAttnMLAMetadata,
            supports_dcp_with_varlen=(interleave_size == 1),
        )
        # Kunlun paged-attention does not use FA3 AOT scheduler metadata.
        self.max_num_splits = 0

        # Persistent pinned buffer for `paged_attention`'s `context_lens_cpu`.
        # A fixed address keeps the host side valid across cuda graph replays.
        max_bs = max(
            vllm_config.scheduler_config.max_num_seqs,
            self.compilation_config.max_cudagraph_capture_size or 0,
        )
        self._seq_lens_cpu = torch.zeros(
            max_bs, dtype=torch.int32, device="cpu", pin_memory=True
        )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashAttnMLADecodeMetadata:
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_query_len = query_lens_cpu.max().item()

        # Host copy for `context_lens_cpu`. Produced here, in `build()`, because
        # a `.cpu()` inside the captured region is an illegal sync; the copy is
        # async since the host values do not affect the kernel result.
        num_reqs = seq_lens_device.shape[0]
        seq_lens_cpu = self._seq_lens_cpu[:num_reqs]
        seq_lens_cpu.copy_(seq_lens_device, non_blocking=True)

        return FlashAttnMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            query_start_loc=query_start_loc_device,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            seq_lens_cpu=seq_lens_cpu,
            scheduler_metadata=None,
            max_num_splits=0,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlashAttnMLAMetadata:
        attn_metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        # Capturing with seq_lens == max_model_len makes the paged-attention
        # kernel walk the whole block table, which makes capture extremely slow.
        # Replay reads the refreshed values from the same buffers anyway.
        assert attn_metadata.decode is not None
        attn_metadata.decode.seq_lens.fill_(1)
        attn_metadata.decode.seq_lens_cpu.fill_(1)
        return attn_metadata


class FlashAttnMLAImpl(MLACommonImpl[FlashAttnMLAMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashAttnMLAImpl"
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashAttnMLA V1 with FP8 KV cache not yet supported"
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("FP8 FlashAttention MLA not yet supported")

        # Decode: one query token per request. Kunlun's paged MLA op expects
        # q of shape [batch, seq_len_q, num_heads, head_dim]; the vLLM decode
        # layout is [num_decode_tokens, num_heads, head_dim] with seq_len_q == 1.
        # NOTE(verify): confirm num_decode_tokens == num_reqs (seq_len_q == 1)
        # and the kv_c_and_k_pe_cache paged layout matches what
        # flash_mla_with_kvcache expects on P800.
        num_tokens, num_heads, head_dim = q.shape
        q = q.view(num_tokens, 1, num_heads, head_dim)

        # Read-only: the host-side copy must not be produced here. A `.cpu()`
        # inside the captured region is an illegal sync during capture and cannot
        # be replayed. `flash_mla_with_kvcache` forwards these as
        # `context_lens_cpu` / `context_lens_xpu`.
        decode_meta = attn_metadata.decode
        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache,
            block_table=decode_meta.block_table,
            cache_seqlens=decode_meta.seq_lens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=decode_meta.seq_lens_cpu,
            num_splits=decode_meta.seq_lens,
            softmax_scale=self.scale,
            causal=True,
        )

        # [batch, seq_len_q(=1), num_heads, kv_lora_rank] -> [num_tokens, heads, dim]
        out = out.view(num_tokens, num_heads, self.kv_lora_rank)

        if self.need_to_return_lse_for_decode:
            # KLX helper currently returns None for lse; DCP path is not wired.
            return out, lse
        return out, None