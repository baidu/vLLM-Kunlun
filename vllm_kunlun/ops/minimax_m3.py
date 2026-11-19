# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Kunlun implementation of MiniMax-M3's fused QK-norm + partial RoPE + KV insert.

MiniMax-M3's attention layers call exactly one vLLM custom op, and no backend but
NVIDIA's registers it, so on Kunlun every attention layer died with

    AttributeError: '_OpNamespace' '_C' object has no attribute
                    'fused_minimax_m3_qknorm_rope_kv_insert'

The op has three branches and all three are implemented here:

* dense -- per-head Gemma RMSNorm on q and k, then partial NeoX RoPE, in place;
* paged insert -- the normed and roped k/v scattered into the main cache by
  ``slot_mapping``;
* lightning index -- ``index_q``/``index_k`` read out of the same fused tensor,
  normed and roped, with ``index_k`` scattered into the index cache.

The arithmetic is torch, not a kunlun_ops kernel: correct and verified, but a
per-token python loop on the insert path. Replacing the loop with a fused kernel is
the natural follow-up and does not change this interface.

Two things here were measured rather than assumed, and both cost a debugging round:

* **The k/v pair axis moves.** Three cache layouts coexist in one M3 process --
  ``(2, num_blocks, num_kv_heads, block_size, head_dim)`` for the dense layers (what
  ``vllm_kunlun/ops/paged_attn.py`` allocates), ``(num_blocks, 2, block_size,
  num_kv_heads, head_dim)`` for the sparse layers (M3's own allocation), and a 3-D
  keys-only ``(num_blocks, 128, head_dim)`` index cache. Selecting the half with
  ``cache[which]`` is right for the first and silently wrong for the second, where it
  picks *block number ``which``*: k lands on the right half of block 0 and looks
  fine, while v is written into block 1's k half, so the v half is never written and
  attention weights perfectly good values that were never stored.
* **The head/length order also moves**, so it is read off the tensor.

A quantized cache is refused rather than half-supported: writing unconverted values
into one is wrong quietly instead of loudly.

Norm semantics follow the reference implementation, ``out = x * rstd * (1 + w)`` in
float32, which is why M3's norm weights sit near -1.
"""

from typing import Optional

import torch
from torch.library import custom_op, impl, register_fake

__all__ = ["fused_minimax_m3_qknorm_rope_kv_insert"]


def _gemma_norm_per_head(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """x is [N, heads, head_dim]; weight is [head_dim], shared across heads."""
    f = x.float()
    rstd = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    return (f * rstd * (1.0 + weight.float())).to(x.dtype)


def _neox_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """Rotate the leading ``rotary_dim`` channels of each head, NeoX halves layout.

    cos and sin are [N, rotary_dim // 2]; x is [N, heads, head_dim].
    """
    rotated = x[..., :rotary_dim]
    passthrough = x[..., rotary_dim:]
    half = rotary_dim // 2
    first, second = rotated[..., :half].float(), rotated[..., half:].float()
    cos = cos[:, None, :].float()
    sin = sin[:, None, :].float()
    out = torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)
    return torch.cat([out.to(x.dtype), passthrough], dim=-1)


def _slots(
    slot_mapping: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
    """(block, offset, writable) per token, from a slot mapping that may contain padding.

    vLLM pads slot mappings with ``PAD_SLOT_ID = -1``
    (``vllm/v1/attention/backends/utils.py``) for tokens whose KV must not be stored, and
    floor division sends -1 to block -1 offset block_size-1 -- measured, for block_size
    128: slots [-1, 0, 5, 130] give blocks [-1, 0, 0, 1] and offsets [127, 0, 5, 2]. So a
    padded token silently overwrites the *last* block, which some other sequence is
    using: no exception, no out-of-range index, just corrupted KV in the place most
    likely to be reused.

    Validity is resolved once here, as a python list, rather than per token with
    ``.item()``: the callers are per-token loops and a device-to-host sync inside one of
    those is the last thing they need.
    """
    flat = slot_mapping.view(-1).to(torch.long)
    return flat // block_size, flat % block_size, (flat >= 0).tolist()


def _pair_half(cache: torch.Tensor, which: int) -> torch.Tensor:
    """Select the k (0) or v (1) half of a 5-D paged cache, wherever the pair axis is."""
    if cache.shape[0] == 2:
        return cache[which]
    if cache.shape[1] == 2:
        return cache[:, which]
    raise NotImplementedError(f"cannot find the k/v pair axis in {tuple(cache.shape)}")


def _insert(
    cache: torch.Tensor,
    which: int,
    values: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter [N, heads, dim] into a paged cache by slot, layout read off the tensor."""
    blocks, offsets, writable = _slots(slot_mapping, block_size)
    target = _pair_half(cache, which) if cache.dim() == 5 else cache
    if target.dim() == 3:
        # The index cache: [num_blocks, block_size, head_dim], keys only, one head.
        if values.shape[1] != 1:
            raise NotImplementedError(
                f"a 3-D cache holds one head; got {values.shape[1]}"
            )
        stored = values.reshape(values.shape[0], -1).to(target.dtype)
        for token in range(stored.shape[0]):
            if not writable[token]:
                continue
            target[blocks[token], offsets[token], :] = stored[token]
        return
    if target.dim() != 4:
        raise NotImplementedError(
            f"unsupported cache rank {tuple(cache.shape)}; expected 3, 4 or 5 dimensions"
        )
    heads = values.shape[1]
    _, first, second, _ = target.shape
    if first == heads and second == block_size and heads != block_size:
        head_major = True
    elif first == block_size and second == heads and heads != block_size:
        head_major = False
    else:
        raise NotImplementedError(
            f"cannot tell the head axis from the length axis in {tuple(target.shape)} "
            f"with {heads} heads and block_size {block_size}"
        )
    stored = values.to(target.dtype)
    for token in range(stored.shape[0]):
        if not writable[token]:
            continue
        if head_major:
            target[blocks[token], :, offsets[token], :] = stored[token]
        else:
            target[blocks[token], offsets[token], :, :] = stored[token]


def _apply(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight,
    index_k_norm_weight,
    num_index_heads: int,
    slot_mapping,
    index_slot_mapping,
    kv_cache,
    index_cache,
    block_size: int,
    q_out,
    index_q_out,
    kv_cache_dtype: str,
) -> None:
    if kv_cache_dtype not in ("auto", ""):
        raise NotImplementedError(
            f"kv_cache_dtype={kv_cache_dtype!r} needs the cache conversion path"
        )

    head_dim = q_norm_weight.shape[-1]
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    sizes = [q_size, kv_size, kv_size]
    index_dim = 0
    if num_index_heads:
        if index_q_norm_weight is None or index_k_norm_weight is None:
            raise ValueError("a sparse layer needs both index norm weights")
        index_dim = index_q_norm_weight.shape[-1]
        # One fused projection carries [q | k | v | index_q | index_k]; index_k is
        # single-head.
        sizes += [num_index_heads * index_dim, index_dim]
    parts = qkv.split(sizes, dim=-1)
    q, k, v = parts[0], parts[1], parts[2]

    cos_sin = cos_sin_cache.index_select(0, positions.view(-1).to(torch.long))
    cos, sin = cos_sin.chunk(2, dim=-1)

    q_heads = _gemma_norm_per_head(q.view(-1, num_heads, head_dim), q_norm_weight, eps)
    k_heads = _gemma_norm_per_head(
        k.view(-1, num_kv_heads, head_dim), k_norm_weight, eps
    )
    q_heads = _neox_partial_rope(q_heads, cos, sin, rotary_dim)
    k_heads = _neox_partial_rope(k_heads, cos, sin, rotary_dim)

    # In place: the caller re-splits the same tensor afterwards.
    q.copy_(q_heads.reshape(q.shape))
    k.copy_(k_heads.reshape(k.shape))
    if q_out is not None:
        q_out.copy_(q_heads.reshape(q_out.shape))

    index_k_heads = None
    if num_index_heads:
        index_q, index_k = parts[3], parts[4]
        index_q_heads = _gemma_norm_per_head(
            index_q.view(-1, num_index_heads, index_dim), index_q_norm_weight, eps
        )
        index_k_heads = _gemma_norm_per_head(
            index_k.view(-1, 1, index_dim), index_k_norm_weight, eps
        )
        index_q_heads = _neox_partial_rope(index_q_heads, cos, sin, rotary_dim)
        index_k_heads = _neox_partial_rope(index_k_heads, cos, sin, rotary_dim)
        index_k.copy_(index_k_heads.reshape(index_k.shape))
        if index_q_out is not None:
            index_q_out.copy_(
                index_q_heads.reshape(index_q_out.shape).to(index_q_out.dtype)
            )
        else:
            index_q.copy_(index_q_heads.reshape(index_q.shape))

    if kv_cache is not None and kv_cache.numel():
        if slot_mapping is None or not block_size:
            raise ValueError(
                "inserting into a paged cache needs slot_mapping and block_size"
            )
        _insert(kv_cache, 0, k_heads, slot_mapping, block_size)
        _insert(
            kv_cache,
            1,
            v.reshape(-1, num_kv_heads, head_dim),
            slot_mapping,
            block_size,
        )
    if index_cache is not None and index_cache.numel():
        if index_k_heads is None:
            raise ValueError("an index cache was given but num_index_heads is 0")
        # Upstream: when index_slot_mapping is omitted, slot_mapping serves both.
        mapping = index_slot_mapping if index_slot_mapping is not None else slot_mapping
        if mapping is None or not block_size:
            raise ValueError("inserting into the index cache needs a slot mapping")
        _insert(index_cache, 0, index_k_heads, mapping, block_size)


@custom_op("_C::fused_minimax_m3_qknorm_rope_kv_insert", mutates_args=())
def fused_minimax_m3_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: Optional[torch.Tensor] = None,
    index_k_norm_weight: Optional[torch.Tensor] = None,
    num_index_heads: int = 0,
    slot_mapping: Optional[torch.Tensor] = None,
    index_slot_mapping: Optional[torch.Tensor] = None,
    kv_cache: Optional[torch.Tensor] = None,
    index_cache: Optional[torch.Tensor] = None,
    block_size: int = 0,
    q_out: Optional[torch.Tensor] = None,
    index_q_out: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
) -> None:
    _apply(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        kv_cache_dtype,
    )


@impl("_C::fused_minimax_m3_qknorm_rope_kv_insert", "CUDA")
def fused_minimax_m3_qknorm_rope_kv_insert_xpu(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: Optional[torch.Tensor] = None,
    index_k_norm_weight: Optional[torch.Tensor] = None,
    num_index_heads: int = 0,
    slot_mapping: Optional[torch.Tensor] = None,
    index_slot_mapping: Optional[torch.Tensor] = None,
    kv_cache: Optional[torch.Tensor] = None,
    index_cache: Optional[torch.Tensor] = None,
    block_size: int = 0,
    q_out: Optional[torch.Tensor] = None,
    index_q_out: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
) -> None:
    _apply(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        kv_cache_dtype,
    )


def _fake_fused_minimax_m3_qknorm_rope_kv_insert(*args, **kwargs) -> None:
    return None


register_fake(
    "_C::fused_minimax_m3_qknorm_rope_kv_insert",
    _fake_fused_minimax_m3_qknorm_rope_kv_insert,
)
