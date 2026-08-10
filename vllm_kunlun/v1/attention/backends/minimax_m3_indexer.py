# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun MiniMax-M3 MSA index-cache attention backend.

MiniMax-M3's sparse attention has two physically separate caches:

* the normal main K/V cache consumed by ``msa_sparse_attention``;
* a key-only side cache that stores one compact index-key vector per token and
  is consumed by ``msa_block_score``.

The side cache is not a full attention cache and must not allocate a V tensor.
This backend mirrors upstream MiniMax-M3's indexer cache layout while keeping the
implementation local to the Kunlun OOT plugin.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec


class MiniMaxM3KunlunIndexerImpl(nn.Module):
    """Placeholder impl for the index side-cache backend.

    The model calls Kunlun MSA kernels directly from
    ``MiniMaxM3Attention._sparse_forward``. The KV manager still requires an
    ``AttentionBackend`` with an impl class so it can allocate and bind the side
    cache, but this class is not instantiated as a normal attention layer.
    """


@dataclass
class MiniMaxM3KunlunIndexerMetadata(AttentionMetadata):
    """Minimal metadata for the MiniMax-M3 index side cache.

    The Kunlun model path reads this cache group's own ``block_table`` when
    scoring the independent index K cache, while logical lengths can be shared
    with the main attention group. ``ForwardContext.slot_mapping`` supplies the
    matching index-cache write addresses.
    """

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_seq_len: int
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_reqs: int


class MiniMaxM3KunlunIndexerMetadataBuilder(
    AttentionMetadataBuilder[MiniMaxM3KunlunIndexerMetadata]
):
    """Build metadata for the MiniMax-M3 Kunlun index side cache."""

    # The Kunlun MiniMax-M3 sparse path calls msa_block_score/topk/sparse-attn
    # directly from the model body. Raised to UNIFORM_BATCH to align with the
    # dense KunlunAttentionBackend so batched decode / uniform batches are not
    # capped by this backend. The model's decode branch now uses device-only
    # LOD and fixed physical upper bounds, while prefill/mixed stays on the
    # dynamic piecewise path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        """Initialize the MiniMax-M3 Kunlun index side-cache metadata builder."""
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if kv_cache_spec.block_size != 128:
            raise ValueError(
                "MiniMax-M3 MSA requires the index cache block size to be 128, "
                f"got {kv_cache_spec.block_size}."
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3KunlunIndexerMetadata:
        """Build runtime index side-cache metadata from common metadata."""
        return MiniMaxM3KunlunIndexerMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_reqs=common_attn_metadata.num_reqs,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> MiniMaxM3KunlunIndexerMetadata:
        """Build side-cache metadata for full CUDA graph decode capture.

        MiniMax-M3's index cache is key-only and is consumed by the model body,
        not by a normal AttentionImpl. During full CUDA graph capture, vLLM
        calls this method for every attention-like cache group. Keep this path
        decode-only so the indexer backend's support level matches the actual
        MSA path we are allowing into full graph capture.
        """
        if common_attn_metadata.max_query_len > 1:
            raise AssertionError(
                "MiniMax-M3 indexer only supports full CUDAGraph capture for "
                "single-token decode; mixed/prefill should use piecewise graph."
            )
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )


class MiniMaxM3KunlunIndexerBackend(AttentionBackend):
    """Key-only index-cache backend for Kunlun MiniMax-M3 MSA."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "float16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        """Return the backend name identifier."""
        return "MINIMAX_M3_KUNLUN_INDEXER"

    @staticmethod
    def get_impl_cls() -> type[MiniMaxM3KunlunIndexerImpl]:
        """Return the attention impl class for this backend."""
        return MiniMaxM3KunlunIndexerImpl

    @staticmethod
    def get_builder_cls() -> type[MiniMaxM3KunlunIndexerMetadataBuilder]:
        """Return the metadata builder class for this backend."""
        return MiniMaxM3KunlunIndexerMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """Return supported head sizes for the index cache."""
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """Return supported kernel block sizes for the index cache."""
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        """Return True as this backend manages a sparse index side-cache."""
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Return the KV cache tensor shape for the index side-cache.

        The index cache is key-only with a single KV head and index_dim=128.
        """
        if num_kv_heads != 1:
            raise ValueError(
                "MiniMax-M3 index cache is key-only and expects one KV head, "
                f"got {num_kv_heads}."
            )
        if head_size != 128:
            raise ValueError(
                "MiniMax-M3 index cache expects head_size/index_dim 128, "
                f"got {head_size}."
            )
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """Return the stride order for the index cache tensor dimensions."""
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)
