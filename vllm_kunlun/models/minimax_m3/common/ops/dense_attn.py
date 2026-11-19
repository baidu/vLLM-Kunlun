# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
"""Torch dense attention for MiniMax-M3's leading layers, for Kunlun XPU.

M3's first three layers are dense, and the platform's attention kernel gets them wrong.
Measured inside the serving process with no golden weights and no tensor-parallel
assumptions: during the prefill of a fresh sequence, causal attention over the very
q/k/v handed to ``Attention.forward`` is the answer, and the kernel's output against
that reference was

    layer 0  relL2 0.326      layer 1  relL2 0.438      layer 2  relL2 0.076

while this implementation, graded by the same instrument, gives 0.00148 / 0.00084 /
0.00152 -- bf16 noise. Three wrong dense layers feed sixty sparse ones, which is why
the symptom was a fluent but prompt-independent answer rather than a crash.

Everything upstream of the kernel had been cleared first, which is what left the kernel
as the only candidate: the input to attention matches a from-checkpoint fp32 golden at
relL2 0.0017, and the fused QK-norm/RoPE op matches vLLM's own RotaryEmbedding
bit-for-bit. Every structural explanation for the gap was swept and eliminated --
QK-norm/RoPE order, causality, the softmax scale, the kv-head-to-rank mapping, the
q-head block, reading past the sequence bound, sliding windows, V being normed or roped
by mistake, per-token int8 activation quantisation, and an fp8 paged cache -- and all of
them sat at 0.245 +/- 0.002.

Set M3_TORCH_DENSE_ATTN=0 to route the dense layers back through the platform kernel and
re-measure.

This writes k/v into the same paged cache the kernel would have written, so decode steps
and prefix reuse keep working. It is a per-row python loop: correct, and the obvious
thing to fuse next.
"""

import os

import torch

USE_TORCH_DENSE_ATTN = os.environ.get("M3_TORCH_DENSE_ATTN", "1") != "0"

__all__ = ["USE_TORCH_DENSE_ATTN", "minimax_m3_dense_attn", "dense_attn_inputs"]


def _split_kv(kv_cache: torch.Tensor):
    """The (k, v) halves of the paged cache, whichever axis carries the pair."""
    if kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
        return kv_cache[0], kv_cache[1]
    if kv_cache.dim() == 5 and kv_cache.shape[1] == 2:
        return kv_cache[:, 0], kv_cache[:, 1]
    raise NotImplementedError(f"unsupported kv cache shape {tuple(kv_cache.shape)}")


def _block_size(kv_cache: torch.Tensor, num_kv_heads: int) -> int:
    try:
        from vllm.config import get_current_vllm_config

        configured = get_current_vllm_config().cache_config.block_size
    except Exception:
        configured = None
    if configured:
        return int(configured)
    half = _split_kv(kv_cache)[0]
    if half.shape[1] == num_kv_heads and half.shape[2] != num_kv_heads:
        return int(half.shape[2])
    if half.shape[2] == num_kv_heads and half.shape[1] != num_kv_heads:
        return int(half.shape[1])
    raise NotImplementedError(
        f"cannot infer the paged block size from {tuple(kv_cache.shape)}"
    )


def _head_major(target: torch.Tensor, heads: int, block_size: int) -> bool:
    """With one kv head per rank the two layouts are shape-identical apart from a
    size-1 axis, so the block size has to come from the config, not the tensor.

    Measured: the dense layers' cache is head-major -- the layout
    ``vllm_kunlun/ops/paged_attn.py`` allocates -- while M3's own sparse cache is
    block-major. Both live in one process.
    """
    _, first, second, _ = target.shape
    if first == heads and second == block_size:
        return True
    if first == block_size and second == heads:
        return False
    raise NotImplementedError(
        f"cache layout {tuple(target.shape)} matches neither heads={heads} "
        f"block_size={block_size} ordering"
    )


def _write(cache_half, values, slot_mapping, block_size, head_major) -> None:
    """Scatter one token per slot, skipping vLLM's PAD_SLOT_ID (-1) entries.

    Floor division maps -1 to block -1 offset block_size-1, i.e. the last block, so a
    padded token would silently overwrite another sequence's KV. Validity is resolved once
    into a python list rather than per token with a device-to-host read.
    """
    flat = slot_mapping.view(-1).to(torch.long)
    blocks = flat // block_size
    offsets = flat % block_size
    writable = (flat >= 0).tolist()
    stored = values.to(cache_half.dtype)
    for token in range(stored.shape[0]):
        if not writable[token]:
            continue
        if head_major:
            cache_half[blocks[token], :, offsets[token], :] = stored[token]
        else:
            cache_half[blocks[token], offsets[token], :, :] = stored[token]


def _read(cache_half, table_row, positions, head, block_size, head_major):
    index = torch.as_tensor(positions, device=cache_half.device)
    pages = (index // block_size).to(torch.long)
    offsets = (index % block_size).to(torch.long)
    physical = table_row.to(torch.long)[pages]
    if head_major:
        return cache_half[physical, head, offsets, :]
    return cache_half[physical, offsets, head, :]


def dense_attn_inputs(layer, attn):
    """The paged cache and metadata for this layer, or None while they are unbound.

    Warmup batches are the reason this is a function rather than two attribute reads:
    before ``bind_kv_cache`` runs, ``Attention.kv_cache`` is a per-virtual-engine list of
    zero-element tensors -- and indexing a zero-element tensor is itself an IndexError --
    and the warmup metadata has its fields present but empty.
    """
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    metadata = getattr(context, "attn_metadata", None)
    if isinstance(metadata, dict):
        metadata = metadata.get(attn.layer_name)
    if metadata is None:
        return None
    for name in ("block_tables", "seq_lens_tensor", "slot_mapping"):
        value = getattr(metadata, name, None)
        if value is None or value.numel() == 0:
            return None
    cache = getattr(attn, "kv_cache", None)
    if isinstance(cache, (list, tuple)):
        index = getattr(context, "virtual_engine", 0)
        if len(cache) <= index:
            return None
        cache = cache[index]
    if not isinstance(cache, torch.Tensor) or cache.dim() < 4 or cache.numel() == 0:
        return None
    return cache, metadata


@torch.no_grad()
def minimax_m3_dense_attn(layer, q, k, v, kv_cache, metadata) -> torch.Tensor:
    """Full causal attention over the paged cache, one query row at a time."""
    num_kv_heads = layer.num_kv_heads
    head_dim = layer.head_dim
    tokens = q.shape[0]
    q = q.reshape(tokens, layer.num_heads, head_dim)
    k = k.reshape(tokens, num_kv_heads, head_dim)
    v = v.reshape(tokens, num_kv_heads, head_dim)
    group = layer.num_heads // num_kv_heads

    k_cache, v_cache = _split_kv(kv_cache)
    block_size = _block_size(kv_cache, num_kv_heads)
    head_major = _head_major(k_cache, num_kv_heads, block_size)

    slot_mapping = metadata.slot_mapping.reshape(-1)[:tokens]
    _write(k_cache, k, slot_mapping, block_size, head_major)
    _write(v_cache, v, slot_mapping, block_size, head_major)

    starts = metadata.query_start_loc
    if starts is None:
        # Decode-only batches carry no query_start_loc: one query token per request.
        starts = list(range(metadata.seq_lens_tensor.shape[0] + 1))
    else:
        starts = starts.tolist()
    seq_lens = metadata.seq_lens_tensor.tolist()
    block_tables = metadata.block_tables
    output = torch.zeros_like(q)
    scale = layer.scaling

    for request in range(len(starts) - 1):
        lo, hi = int(starts[request]), int(starts[request + 1])
        if hi <= lo:
            continue
        length = int(seq_lens[request])
        context = length - (hi - lo)
        table_row = block_tables[request]
        for row in range(hi - lo):
            token = lo + row
            positions = list(range(context + row + 1))
            for head in range(num_kv_heads):
                keys = _read(
                    k_cache, table_row, positions, head, block_size, head_major
                )
                values = _read(
                    v_cache, table_row, positions, head, block_size, head_major
                )
                queries = q[token, head * group : (head + 1) * group].float()
                logits = (queries @ keys.float().transpose(0, 1)) * scale
                weights = torch.softmax(logits, dim=-1)
                output[token, head * group : (head + 1) * group] = (
                    weights @ values.float()
                ).to(output.dtype)

    return output.reshape(tokens, layer.num_heads * head_dim)
