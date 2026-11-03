# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
"""MiniMax M3 lightning-indexer block scoring + top-k, in torch, for Kunlun XPU.

Index queries score each 128-token block of index keys (max over the block), then the
top-k blocks -- plus forced init/local blocks -- are selected per query token. The KV
page size is forced to equal the sparse block size (128), so one sparse block maps to
exactly one page. Index-K cache layout: ``(num_blocks, 128, idx_head_dim)``, single
head. Only the paths M3 uses are implemented: score_type="max", index value disabled
(score-only indexer), one shared index head.

Upstream computes this in Triton. Triton compiles and runs on P800, but this kernel does
not fit: it asks for more shared memory than the device has, and the launch dies on the
first real request, after the server has already reported ``Application startup
complete``:

    common/ops/index_topk.py _index_block_score_kernel
    triton.runtime.errors.OutOfResources: out of resource: shared memory,
        Required: 50180, Hardware limit: 49152

Tuning block sizes only moves the failure to the next shape, so the arithmetic is
expressed in torch. The public API is unchanged; the indexer does not care what computes
its scores.

The semantics below were cross-checked against the vendor's ``msa_block_score`` and
``msa_block_score_topk_transform`` kernels: a block score is the max over the block of
scaled q·k, the transform reserves a local block, and unused top-k slots are -1.

Layouts are upstream's: score is ``[heads, total_q, max_block]`` and top-k is
``[heads, total_q, topk]``, both head-major. The score's last dimension keeps upstream's
round-up to 16 so a caller that pre-allocated a buffer still fits.

These are per-request python loops: correct, and the obvious thing to fuse next.
"""

import torch
from vllm.utils.math_utils import round_up

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128

__all__ = [
    "SPARSE_BLOCK_SIZE",
    "minimax_m3_index_decode",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
]


def _gather_index_keys(
    cache: torch.Tensor, table_row: torch.Tensor, length: int
) -> torch.Tensor:
    """The visible index keys of one request, [length, head_dim]."""
    blocks = (length + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    pages = table_row[:blocks].to(torch.long)
    keys = cache.index_select(0, pages).reshape(-1, cache.shape[-1])
    return keys[:length]


def _block_scores(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    num_idx_heads: int,
    score: torch.Tensor,
) -> torch.Tensor:
    starts = cu_seqlens_q.tolist()
    lengths = seq_lens.tolist()
    prefixes = prefix_lens.tolist()
    for request in range(len(lengths)):
        lo, hi = int(starts[request]), int(starts[request + 1])
        if hi <= lo:
            continue
        keys = _gather_index_keys(
            index_kv_cache, block_table[request], int(lengths[request])
        ).to(torch.float32)
        q = idx_q[lo:hi].to(torch.float32)  # [q, heads, dim]
        logits = torch.einsum("qhd,kd->hqk", q, keys)
        # Causal: query token i of this request sits at prefix + i.
        positions = torch.arange(hi - lo, device=q.device) + int(prefixes[request])
        invalid = (
            torch.arange(keys.shape[0], device=q.device)[None, :] > positions[:, None]
        )
        logits = logits.masked_fill(invalid[None, :, :], float("-inf"))
        blocks = (keys.shape[0] + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
        padded = blocks * SPARSE_BLOCK_SIZE - keys.shape[0]
        if padded:
            logits = torch.nn.functional.pad(logits, (0, padded), value=float("-inf"))
        tiled = logits.reshape(num_idx_heads, hi - lo, blocks, SPARSE_BLOCK_SIZE)
        score[:, lo:hi, :blocks] = tiled.amax(dim=-1)
    return score


@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Compute per-token index scores for each visible sparse block.

    Returns score [num_kv_heads, total_q, max_block], where each score is the
    max over a 128-token index-K block. M3 has num_idx_heads == num_kv_heads.
    """
    total_q, num_idx_heads, _ = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    max_block = (max_seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    score = idx_q.new_full(
        (num_idx_heads, total_q, round_up(max_block, 16)),
        float("-inf"),
        dtype=torch.float32,
    )
    return _block_scores(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_idx_heads,
        score,
    )


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,  # [num_idx_heads, total_q, max_block]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select index top-k from a precomputed score tensor.

    Returns [num_idx_heads, total_q, topk] 0-indexed block ids, -1 in unused slots.
    When ``out`` is provided (a ``[num_idx_heads, >=total_q, topk]`` buffer), the result
    is written into ``out[:, :total_q, :]`` and that view is returned, so the caller
    keeps a stable address.
    """
    heads, total_q, _ = score.shape
    result = score.new_full((heads, total_q, topk), -1, dtype=torch.int32)
    starts = cu_seqlens_q.tolist()
    prefixes = prefix_lens.tolist()
    for request in range(len(starts) - 1):
        lo, hi = int(starts[request]), int(starts[request + 1])
        if hi <= lo:
            continue
        positions = torch.arange(hi - lo, device=score.device) + int(prefixes[request])
        # Blocks up to and including the one holding the query token.
        visible = (positions // SPARSE_BLOCK_SIZE) + 1
        for row in range(hi - lo):
            count = int(visible[row])
            keep = min(topk, count)
            forced: list[int] = list(range(min(init_blocks, count)))
            forced += [
                block
                for block in range(max(0, count - local_blocks), count)
                if block not in forced
            ]
            forced = forced[:keep]
            for head in range(heads):
                values = score[head, lo + row, :count].clone()
                for block in forced:
                    values[block] = float("inf")
                chosen = torch.topk(values, keep).indices.to(torch.int32)
                result[head, lo + row, :keep] = chosen
    if out is not None:
        out[:, :total_q, :].copy_(result)
        return out[:, :total_q, :]
    return result


def _decode_segments(
    idx_q: torch.Tensor, seq_lens: torch.Tensor, decode_query_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode as prefill: `decode_query_len` query tokens at the tail of each sequence."""
    total_q = idx_q.shape[0]
    step = max(1, decode_query_len)
    batch = total_q // step
    cu_seqlens_q = torch.arange(
        0, total_q + 1, step, dtype=torch.int32, device=idx_q.device
    )
    lengths = seq_lens[:batch].to(torch.int32)
    prefixes = (lengths - step).clamp(min=0)
    return cu_seqlens_q, lengths, prefixes


@torch.no_grad()
def minimax_m3_index_decode_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    score_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode-side block scores.

    When ``score_out`` is given the scores are written into it (and it is reset first,
    because a stale row would otherwise be selected), so the prefill and decode sides can
    share one buffer.
    """
    total_q, num_idx_heads, _ = idx_q.shape
    cu_seqlens_q, lengths, prefixes = _decode_segments(
        idx_q, seq_lens, decode_query_len
    )
    max_block = (max_seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    if score_out is not None:
        score = score_out[:, :total_q, :]
        score.fill_(float("-inf"))
    else:
        score = idx_q.new_full(
            (num_idx_heads, total_q, round_up(max_block, 16)),
            float("-inf"),
            dtype=torch.float32,
        )
    return _block_scores(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        lengths,
        prefixes,
        num_idx_heads,
        score,
    )


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    out: torch.Tensor | None = None,
    score_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode index block-score + top-k.

    Returns topk_idx [num_kv_heads, total_q, topk] (0-indexed block ids, -1 pad).
    When ``out`` ([num_kv_heads, >=total_q, topk]) is given, writes into
    ``out[:, :total_q, :]`` instead of allocating.
    """
    cu_seqlens_q, _, prefixes = _decode_segments(idx_q, seq_lens, decode_query_len)
    score = minimax_m3_index_decode_score(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        max_decode_query_len,
        score_out=score_out,
    )
    return minimax_m3_index_topk(
        score,
        cu_seqlens_q,
        prefixes,
        decode_query_len,
        topk,
        init_blocks,
        local_blocks,
        out=out,
    )
