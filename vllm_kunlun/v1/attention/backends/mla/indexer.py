# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    kv_spans_from_batches,
    split_prefill_chunks,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)

logger = init_logger(__name__)


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
            common_attn_metadata.seq_lens_cpu,
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

        seq_lens = common_attn_metadata.seq_lens[:num_decodes]

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


DeepseekV32IndexerMetadataBuilder.build_one_prefill_chunk = (
    kunlun_build_one_prefill_chunk
)
DeepseekV32IndexerMetadataBuilder.build = kunlun_build

# Monkey patch: Upgrade cudagraph_support to UNIFORM_BATCH for spec-decode compatibility
from vllm.v1.attention.backends.utils import AttentionCGSupport

DeepseekV32IndexerMetadataBuilder.cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
