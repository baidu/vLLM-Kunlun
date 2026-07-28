# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)

logger = init_logger(__name__)


def get_max_prefill_buffer_size(vllm_config: VllmConfig) -> int:
    """Size of the flattened-KV workspace consumed by the Kunlun indexer op.

    Upstream raised its own factor to 40 for the fp8 workspace layout; the
    Kunlun kernel still sizes its buffer as 2 * max_model_len, and the same
    value is handed to ``sparse_attn_indexer_vllm_kunlun``, so both sides must
    agree here.
    """
    return vllm_config.model_config.max_model_len * 2


def kv_spans_from_batches(
    start_seq_loc: torch.Tensor,
    seq_len_per_batch: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token [start, end) row bounds in the flattened KV cache.

    Vendored: dropped from vllm upstream when the indexer moved to a triton
    metadata kernel.
    """
    q = start_seq_loc.to(dtype=torch.long)
    L = seq_len_per_batch.to(dtype=torch.long)
    assert q.dim() == 1 and L.dim() == 1
    assert q.numel() == L.numel() + 1, "start_seq_loc must have length B+1"

    counts = q[1:] - q[:-1]
    N = int(q[-1].item())
    B = L.numel()

    if N == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    kv_starts_per_batch = torch.cumsum(L, dim=0) - L
    batch_id = torch.repeat_interleave(torch.arange(B), counts)
    start_tensor = kv_starts_per_batch[batch_id]

    L_expand = torch.repeat_interleave(L, counts)
    m_expand = torch.repeat_interleave(counts, counts)
    pos_within = (
        torch.arange(N, dtype=torch.long)
        - torch.repeat_interleave(q[:-1], counts)
        + 1
    )

    local_pos = L_expand - m_expand + pos_within
    end_location = start_tensor + local_pos

    return start_tensor.int().to(device), end_location.int().to(device)


def split_prefill_chunks(
    seq_lens_cpu: torch.Tensor,
    max_prefill_buffer_size: int,
    request_offset: int = 0,
) -> list[tuple[int, int]]:
    """Group prefill requests so each chunk fits the flattened-KV workspace.

    Vendored: upstream replaced it with ``split_indexer_prefill_chunks``, which
    also sub-chunks on the query dimension and returns slices.
    """
    chunk_seq_ids: list[tuple[int, int]] = []
    total_seq_lens = 0
    start = 0
    for i in range(len(seq_lens_cpu)):
        cur_seq_len = seq_lens_cpu[i].item()
        assert cur_seq_len <= max_prefill_buffer_size
        total_seq_lens += cur_seq_len
        if total_seq_lens > max_prefill_buffer_size:
            chunk_seq_ids.append((start + request_offset, i + request_offset))
            start = i
            total_seq_lens = cur_seq_len
    if total_seq_lens > 0:
        chunk_seq_ids.append(
            (start + request_offset, len(seq_lens_cpu) + request_offset)
        )
    return chunk_seq_ids


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    context_q_lens: torch.Tensor
    context_q_lens_cpu: torch.Tensor
    context_k_lens: torch.Tensor
    context_k_lens_cpu: torch.Tensor


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor


@dataclass
class DeepseekV32IndexerMetadata:

    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    # The dimension of the attention heads
    head_dim: int

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: Optional[DeepSeekV32IndexerDecodeMetadata] = None
    prefill: Optional[DeepseekV32IndexerPrefillMetadata] = None


def kunlun_build_one_prefill_chunk(
    self, reqs_start, reqs_end, query_start_loc_cpu, seq_lens_cpu, block_table
):
    prefill_query_start_loc = (
        query_start_loc_cpu[reqs_start : reqs_end + 1] - query_start_loc_cpu[reqs_start]
    )
    cu_seqlen_ks, cu_seqlen_ke = kv_spans_from_batches(
        prefill_query_start_loc, seq_lens_cpu[reqs_start:reqs_end], self.device
    )
    token_start = query_start_loc_cpu[reqs_start].item()
    token_end = query_start_loc_cpu[reqs_end].item()
    total_seq_lens = seq_lens_cpu[reqs_start:reqs_end].sum()
    assert total_seq_lens <= self.max_prefill_buffer_size
    cu_seq_lens = (
        torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                seq_lens_cpu[reqs_start:reqs_end].cumsum(dim=0),
            ]
        )
        .to(torch.int32)
        .to(self.device)
    )
    seq_len_q = token_end - token_start
    seq_len_kv = total_seq_lens
    context_q_lens = torch.tensor([0, seq_len_q], dtype=torch.int32, device=self.device)
    context_k_lens = torch.tensor(
        [0, seq_len_kv], dtype=torch.int32, device=self.device
    )
    context_q_lens_cpu = torch.tensor([0, seq_len_q], dtype=torch.int32, device="cpu")
    context_k_lens_cpu = torch.tensor([0, seq_len_kv], dtype=torch.int32, device="cpu")

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
        cu_seq_lens=cu_seq_lens,
        total_seq_lens=total_seq_lens,
        block_table=block_table[reqs_start:reqs_end],
        token_start=token_start,
        token_end=token_end,
        num_reqs=reqs_end - reqs_start,
        context_q_lens=context_q_lens,
        context_q_lens_cpu=context_q_lens_cpu,
        context_k_lens=context_k_lens,
        context_k_lens_cpu=context_k_lens_cpu,
    )


def kunlun_build(
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    fast_build: bool = False,
) -> DeepseekV32IndexerMetadata:

    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.reorder_batch_threshold
        )
    )

    assert num_decodes + num_prefills == num_reqs
    assert num_decode_tokens + num_prefill_tokens == num_tokens

    prefill_metadata = None
    if num_prefills > 0:
        chunk_seq_ids = split_prefill_chunks(
            common_attn_metadata.seq_lens_cpu[num_decodes:],
            self.max_prefill_buffer_size,
            num_decodes,
        )
        chunks = [
            self.build_one_prefill_chunk(
                reqs_start,
                reqs_end,
                query_start_loc_cpu,
                common_attn_metadata.seq_lens_cpu,
                common_attn_metadata.block_table_tensor,
            )
            for reqs_start, reqs_end in chunk_seq_ids
        ]
        prefill_metadata = DeepseekV32IndexerPrefillMetadata(
            chunks=chunks,
        )

    decode_metadata = None
    if num_decodes > 0:
        torch.diff(
            common_attn_metadata.query_start_loc[: num_decodes + 1],
            out=self.decode_lens_buffer[:num_decodes],
        )
        decode_lens = self.decode_lens_buffer[:num_decodes]
        decode_lens_cpu = torch.diff(
            common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
        )

        # Use CPU to avoid GPU sync; breaking async scheduling
        requires_padding = (decode_lens_cpu.max() > decode_lens_cpu.min()).item()

        # seq_lens = common_attn_metadata.seq_lens[:num_decodes]

        decode_metadata = DeepSeekV32IndexerDecodeMetadata(
            block_table=common_attn_metadata.block_table_tensor[:num_decodes, ...],
            seq_lens=common_attn_metadata.seq_lens[:num_decodes],
            seq_lens_cpu=common_attn_metadata.seq_lens[:num_decodes].cpu(),
            decode_lens=decode_lens,
            requires_padding=requires_padding,
            schedule_metadata=self.scheduler_metadata_buffer,
        )

    attn_metadata = DeepseekV32IndexerMetadata(
        seq_lens=common_attn_metadata.seq_lens,
        seq_lens_cpu=common_attn_metadata.seq_lens.cpu(),
        num_reqs=common_attn_metadata.num_reqs,
        max_query_len=common_attn_metadata.max_query_len,
        max_seq_len=common_attn_metadata.max_seq_len,
        num_actual_tokens=common_attn_metadata.num_actual_tokens,
        query_start_loc=common_attn_metadata.query_start_loc,
        slot_mapping=common_attn_metadata.slot_mapping,
        head_dim=128,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        prefill=prefill_metadata,
        decode=decode_metadata,
    )

    # if get_tensor_model_parallel_rank() == 0:
    #     logger.info(f"attn_metadata: {attn_metadata}")
    return attn_metadata


_upstream_builder_init = DeepseekV32IndexerMetadataBuilder.__init__


def kunlun_builder_init(self, *args, **kwargs):
    _upstream_builder_init(self, *args, **kwargs)
    self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)


DeepseekV32IndexerMetadataBuilder.__init__ = kunlun_builder_init
DeepseekV32IndexerMetadataBuilder.build_one_prefill_chunk = (
    kunlun_build_one_prefill_chunk
)
DeepseekV32IndexerMetadataBuilder.build = kunlun_build

# Monkey patch: Upgrade cudagraph_support to UNIFORM_BATCH for spec-decode compatibility
from vllm.v1.attention.backend import AttentionCGSupport  # noqa

DeepseekV32IndexerMetadataBuilder._cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
