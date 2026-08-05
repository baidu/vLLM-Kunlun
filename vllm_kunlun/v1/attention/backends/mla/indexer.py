# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)

logger = init_logger(__name__)


# TODO (zyongye) optimize this, this is now vibe coded
def kv_spans_from_batches(
    start_seq_loc: torch.Tensor, seq_len_per_batch: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      start_seq_loc: 1D long tensor [B+1], cumulative counts of
                     selected tokens per batch.
            Example: [0, 2, 4, 7] ->
                     batch sizes (selected) [2, 2, 3], N=7 tokens total.
      seq_len_per_batch: 1D long tensor [B],
                         full sequence length (KV length) of each batch.
                         Example: [5, 9, 4].

    Returns:
      start_tensor: 1D long tensor [N], start offset in the
                    concatenated KV cache for each token's batch.
      end_location: 1D long tensor [N],
                    **exclusive** end = start + token's local position.
                    (So the attended KV slice is kv[start:end].)

    Assumes each batch contributes its full `seq_len_per_batch[i]`
    keys to the KV cache, andthe selected tokens within a batch
    are the **last** `counts[i]` positions of that sequence.
    """
    q = start_seq_loc.to(dtype=torch.long)
    L = seq_len_per_batch.to(dtype=torch.long)
    assert q.dim() == 1 and L.dim() == 1
    assert q.numel() == L.numel() + 1, "start_seq_loc must have length B+1"

    # Selected tokens per batch and totals
    counts = q[1:] - q[:-1]  # [B]
    N = int(q[-1].item())  # total selected tokens
    B = L.numel()

    if N == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    # KV start offsets per batch in the concatenated KV cache
    kv_starts_per_batch = torch.cumsum(L, dim=0) - L  # [B]

    # For each selected token, which batch does it belong to?
    batch_id = torch.repeat_interleave(torch.arange(B), counts)  # [N]

    # Map batch KV start to each token
    start_tensor = kv_starts_per_batch[batch_id]  # [N]

    # End-align local positions inside each batch:
    # local_pos = L[b] - counts[b] + (1..counts[b])  for each batch b
    L_expand = torch.repeat_interleave(L, counts)  # [N]
    m_expand = torch.repeat_interleave(counts, counts)  # [N]
    # position within the selected block: 1..counts[b]
    pos_within = (
        torch.arange(N, dtype=torch.long) - torch.repeat_interleave(q[:-1], counts) + 1
    )

    local_pos = L_expand - m_expand + pos_within  # [N], 1-based
    end_location = start_tensor + local_pos  # exclusive end

    return start_tensor.int().to(device), end_location.int().to(device)


def split_prefill_chunks(
    seq_lens_cpu: torch.Tensor, workspace_size: int, request_offset: int = 0
) -> list[tuple[int, int]]:
    """
    Split the prefill requests into chunks such that the total sequence length
    of each chunk is less than or equal to the workspace size.

    Args:
        seq_lens_cpu: The sequence lengths of the prefill requests on CPU.
        workspace_size: The maximum workspace size (in tokens) per chunk.
        request_offset: The offset to add to the request indices.
    Returns:
        A list of tuples of (reqs_start, reqs_end) representing chunk boundaries.
    """
    chunk_bounds = []
    i, n = 0, len(seq_lens_cpu)
    assert torch.all(seq_lens_cpu <= workspace_size).item()

    while i < n:
        start, chunk_total = i, 0
        while i < n and (chunk_total + (s := seq_lens_cpu[i].item())) <= workspace_size:
            chunk_total += s
            i += 1
        chunk_bounds.append((start + request_offset, i + request_offset))
    return chunk_bounds


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
    total_seq_lens = int(seq_lens_cpu[reqs_start:reqs_end].sum())
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
            # Reuse the host-side copy the model runner already built instead of
            # issuing a fresh D2H copy, which would sync and break async
            # scheduling. NOTE: the EAGLE/MTP proposer clamps the device
            # seq_lens for requests past max_model_len without mirroring that to
            # the host copy, so the two can differ for those (discarded)
            # requests once speculative decoding is enabled.
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu[:num_decodes],
            decode_lens=decode_lens,
            requires_padding=requires_padding,
            schedule_metadata=self.scheduler_metadata_buffer,
        )

    attn_metadata = DeepseekV32IndexerMetadata(
        seq_lens=common_attn_metadata.seq_lens,
        seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
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
