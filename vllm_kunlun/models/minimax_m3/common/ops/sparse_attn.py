# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
"""MiniMax M3 block-sparse GQA attention, in torch, for Kunlun XPU.

The main heads attend only to the blocks selected by the lightning indexer (see
``index_topk``). The KV page size is forced to equal the sparse block size (128), so one
selected block maps to exactly one page. Only the paths M3 uses are implemented: no
attention sink.

Upstream computes this in Triton, and on P800 that kernel is over the shared-memory budget
by a wider margin than the indexer's -- 69636 bytes against a 49152 limit -- so it is
expressed in torch instead. The public API is unchanged.

Two layout facts here were measured rather than read, and both are the kind that produce
silently wrong numbers instead of an exception:

* **The pair axis position differs between M3's caches.** This module's cache is
  ``(num_blocks, 2, 128, num_kv_heads, head_dim)`` -- pair axis at dim 1 -- while the dense
  layers use the platform's ``(2, num_blocks, num_kv_heads, block_size, head_dim)``.
  ``_split_kv`` therefore looks for the size-2 axis instead of assuming dim 0.
* **The head axis and the length axis cannot be told apart from the tensor** when a rank
  holds a single kv head, which is the case at TP=8, so the paged block size comes from the
  config rather than from a shape.

The selection buffer is also not the shape its type hint suggests: the indexer hands out
the whole persistent ``topk_indices_buffer``, token-major
``(max_num_batched_tokens, num_kv_heads, topk)``, rather than a
``[num_kv_heads, total_q, topk]`` slice. ``_selection_view`` matches the extents against
the two known orientations and refuses anything else instead of indexing blindly.

Both entry points zero their output first: the caller passes an uninitialised
``torch.empty_like(q)``, and rows with no visible keys are skipped, so without the zeroing
the attend returns whatever was in that memory (measured as NaN on a warmup batch).

These are per-row python loops: correct, and the obvious thing to fuse next.
"""

import os

import torch

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128

__all__ = [
    "SPARSE_BLOCK_SIZE",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
]

# A differential switch, not a feature: with M3_FORCE_DENSE_ATTEND=1 the attend ignores
# the selection and reads the whole visible context, which is a mathematical superset of
# any block selection. Useful for deciding whether a wrong answer comes from the selection
# or from the attend, on prompts long enough for the two to differ (>128*topk tokens).
FORCE_DENSE = os.environ.get("M3_FORCE_DENSE_ATTEND") == "1"


def _split_kv(kv_cache: torch.Tensor):
    """The (k, v) halves of the paged cache, whichever axis carries the pair."""
    if kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
        return kv_cache[0], kv_cache[1]
    if kv_cache.dim() == 5 and kv_cache.shape[1] == 2:
        return kv_cache[:, 0], kv_cache[:, 1]
    raise NotImplementedError(f"unsupported kv cache shape {tuple(kv_cache.shape)}")


def _paged_layout(kv_cache: torch.Tensor, num_kv_heads: int) -> tuple[int, bool]:
    """(block_size, head_major) for the paged cache."""
    half = _split_kv(kv_cache)[0]
    try:
        from vllm.config import get_current_vllm_config

        block_size = int(get_current_vllm_config().cache_config.block_size)
    except Exception:
        block_size = None
    if block_size is None:
        # get_current_vllm_config() is only set while the model is being built, not during
        # a forward, so fall back to the axis that cannot be the head axis. With one kv head
        # per rank exactly one of the two candidates differs from num_kv_heads, which is
        # enough -- and it matches the convention the model itself uses when it passes
        # ``self.kv_cache.size(2)`` as the paged block size.
        if half.shape[1] == num_kv_heads and half.shape[2] != num_kv_heads:
            block_size = int(half.shape[2])
        elif half.shape[2] == num_kv_heads and half.shape[1] != num_kv_heads:
            block_size = int(half.shape[1])
        else:
            raise NotImplementedError(
                f"cannot infer the paged block size from {tuple(kv_cache.shape)}"
            )
    if half.shape[1] == num_kv_heads and half.shape[2] == block_size:
        return block_size, True
    if half.shape[1] == block_size and half.shape[2] == num_kv_heads:
        return block_size, False
    raise NotImplementedError(
        f"cache layout {tuple(half.shape)} matches neither heads={num_kv_heads} "
        f"block_size={block_size} ordering"
    )


def _positions(blocks: list[int], length: int) -> list[int]:
    """Absolute token positions covered by the selected blocks, clipped to length."""
    positions: list[int] = []
    for block in blocks:
        if block < 0:
            continue
        start = block * SPARSE_BLOCK_SIZE
        positions.extend(range(start, min(start + SPARSE_BLOCK_SIZE, length)))
    return sorted(set(positions))


def _selection_view(topk_idx: torch.Tensor, total_q: int, num_kv_heads: int):
    """A [total_q, num_kv_heads, topk] view, whichever way round it arrived."""
    shape = tuple(topk_idx.shape)
    if len(shape) != 3:
        raise NotImplementedError(f"topk_idx must be 3-D, got {shape}")
    if shape[1] == num_kv_heads and shape[0] >= total_q:
        return topk_idx[:total_q]
    if shape[0] == num_kv_heads and shape[1] >= total_q:
        return topk_idx.permute(1, 0, 2)[:total_q]
    raise NotImplementedError(
        f"cannot orient topk_idx {shape} against total_q={total_q} and "
        f"num_kv_heads={num_kv_heads}"
    )


def _attend_row(
    q_row,
    k_cache,
    v_cache,
    table_row,
    selected,
    length,
    position,
    num_kv_heads,
    group,
    scale,
    out_row,
    block_size,
    head_major,
) -> None:
    """One query token: per kv head, attend over that head's selected blocks.

    The *selection* is in units of SPARSE_BLOCK_SIZE (the model's sparse_block_size); the
    *cache* is paged in units of ``block_size``. Both are 128 in this deployment, which is
    exactly why they are kept apart here.
    """
    for head in range(num_kv_heads):
        if FORCE_DENSE:
            positions = list(range(min(length, position + 1)))
        else:
            positions = [p for p in _positions(selected[head], length) if p <= position]
        if not positions:
            continue
        index = torch.tensor(positions, device=q_row.device)
        pages = (index // block_size).to(torch.long)
        offsets = (index % block_size).to(torch.long)
        physical = table_row.to(torch.long)[pages]
        if head_major:
            keys = k_cache[physical, head, offsets, :].to(torch.float32)
            values = v_cache[physical, head, offsets, :].to(torch.float32)
        else:
            keys = k_cache[physical, offsets, head, :].to(torch.float32)
            values = v_cache[physical, offsets, head, :].to(torch.float32)
        lo, hi = head * group, (head + 1) * group
        queries = q_row[lo:hi].to(torch.float32)
        logits = (queries @ keys.transpose(0, 1)) * scale
        weights = torch.softmax(logits, dim=-1)
        out_row[lo:hi] = (weights @ values).to(out_row.dtype)


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
) -> None:
    """GQA block-sparse attention over the selected blocks, prefill side."""
    output.zero_()
    k_cache, v_cache = _split_kv(kv_cache)
    block_size, head_major = _paged_layout(kv_cache, num_kv_heads)
    group = q.shape[1] // num_kv_heads
    selection = _selection_view(topk_idx, q.shape[0], num_kv_heads)
    starts = cu_seqlens_q.tolist()
    lengths = seq_lens.tolist()
    prefixes = prefix_lens.tolist()
    for request in range(len(starts) - 1):
        lo, hi = int(starts[request]), int(starts[request + 1])
        for row in range(hi - lo):
            token = lo + row
            selected = [selection[token, head].tolist() for head in range(num_kv_heads)]
            _attend_row(
                q[token],
                k_cache,
                v_cache,
                block_table[request],
                selected,
                int(lengths[request]),
                int(prefixes[request]) + row,
                num_kv_heads,
                group,
                sm_scale,
                output[token],
                block_size,
                head_major,
            )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
) -> None:
    """GQA block-sparse attention for decode: the query tokens are the sequence tail."""
    output.zero_()
    k_cache, v_cache = _split_kv(kv_cache)
    block_size, head_major = _paged_layout(kv_cache, num_kv_heads)
    group = q.shape[1] // num_kv_heads
    selection = _selection_view(topk_idx, q.shape[0], num_kv_heads)
    lengths = seq_lens.tolist()
    for request in range(len(lengths)):
        length = int(lengths[request])
        for row in range(decode_query_len):
            token = request * decode_query_len + row
            selected = [selection[token, head].tolist() for head in range(num_kv_heads)]
            _attend_row(
                q[token],
                k_cache,
                v_cache,
                block_table[request],
                selected,
                length,
                length - decode_query_len + row,
                num_kv_heads,
                group,
                sm_scale,
                output[token],
                block_size,
                head_major,
            )
