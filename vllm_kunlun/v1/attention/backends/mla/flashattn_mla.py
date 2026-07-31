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
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_kunlun.ops.attention.flashmla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)

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
    # Kept for API compatibility with the upstream metadata; unused on Kunlun.
    scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0


@dataclass
class FlashAttnMLAMetadata(MLACommonMetadata[FlashAttnMLADecodeMetadata]):
    pass


class FlashAttnMLAMetadataBuilder(MLACommonMetadataBuilder[FlashAttnMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN
    reorder_batch_threshold: int = 512  # process small prefills with decode pathway

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

        return FlashAttnMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            query_start_loc=query_start_loc_device,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            scheduler_metadata=None,
            max_num_splits=0,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )


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

        cache_seqlens = attn_metadata.decode.seq_lens.to(torch.int32)
        tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens)

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache,
            block_table=attn_metadata.decode.block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=self.scale,
            causal=True,
        )

        # [batch, seq_len_q(=1), num_heads, kv_lora_rank] -> [num_tokens, heads, dim]
        out = out.view(num_tokens, num_heads, self.kv_lora_rank)

        if self.need_to_return_lse_for_decode:
            # KLX helper currently returns None for lse; DCP path is not wired.
            return out, lse
        return out, None