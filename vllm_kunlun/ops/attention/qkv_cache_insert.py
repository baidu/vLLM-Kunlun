"""DSV4 fused Q-norm + RoPE + KV-cache insertion adapter.

DeepSeek-V4's attention does not use the standard vLLM rotary-embedding/
cache-write helpers directly.  Instead it calls custom op symbols:

* ``torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``
* ``torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert``

Upstream only registers these inside the GPU-specific CUDA/AMD implementations.
Kunlun's equivalent native op is
:obj:`kunlun_ops.fused_deepseek_v4_qnorm_rope_kv_insert`;
this adapter wires that implementation into
:mod:`vllm.models.deepseek_v4.attention`'s namespace at import time,
and provides a bit-identical PyTorch fallback for the BF16 cache path when the
native kernel is absent or fails at runtime.
"""
import logging
from typing import Callable, List

import torch

from vllm_kunlun.runtime_utils import WarningOnce, find_op

LOGGER = logging.getLogger("vllm_kunlun.ops.attention.qkv_cache_insert")
_APPLIED_SENTINEL = "_dsv4_qkv_cache_wired"
_WARNED_BF16_FALLBACK_KEY = "qkv-insert-bf16-native-failed-once"
_INSERT_OP_NAME = "fused_deepseek_v4_qnorm_rope_kv_insert"
_QUANT_ATTR = (
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
)
_BF16_CACHE_ATTR = (
    "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert"
)
_ROPE_DIM = 64
_HALF_DIM = _ROPE_DIM // 2
_FALSE = object()
_NATIVE_OP_CACHE: dict[str, object] = {}


def _cached_native_op(name: str):
    """Cache a single probe for ``kunlun_ops.<name>``."""
    value = _NATIVE_OP_CACHE.get(name, _FALSE)
    if value is not _FALSE:
        return value if value is not None else None
    try:
        mod = __import__("kunlun_ops", fromlist=[name])
        handle = getattr(mod, name)
    except Exception:
        handle = None
    _NATIVE_OP_CACHE[name] = handle
    return handle


def _bf16_torch_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache_3d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    block_size: int,
) -> torch.Tensor:
    """Pure-PyTorch equivalent used when the BF16 fused insert kernel fails.

    Mirrors exactly what the native kernel promises:
      - apply RMSNorm (no extra scale/weight) to Q along last dim;
      - GPT-J style RoPE to both Q and KV on their trailing rope_dim entries;
      - write KV rows with valid (>=0) slot indices into swa_kv_cache_3d.
    """
    rms = torch.rsqrt(q.float().square().mean(dim=-1, keepdim=True) + eps)
    q.copy_((rms * q.float()).to(dtype=q.dtype))

    selected_cos_sin = cos_sin[positions]
    cos_val = selected_cos_sin[:, :_HALF_DIM]
    sin_val = selected_cos_sin[:, _HALF_DIM:_ROPE_DIM]

    def apply_gptj(x):
        x1 = x[..., ::2].float()
        x2 = x[..., 1::2].float()
        if x.ndim == 3:
            cos_b = cos_val.unsqueeze(1).float()
            sin_b = sin_val.unsqueeze(1).float()
        else:
            cos_b = cos_val.float()
            sin_b = sin_val.float()
        out1 = x1 * cos_b - x2 * sin_b
        out2 = x1 * sin_b + x2 * cos_b
        return torch.stack([out1, out2], dim=-1).flatten(-2).to(dtype=x.dtype)

    q[..., -_ROPE_DIM:] = apply_gptj(q[..., -_ROPE_DIM:])
    kv_rotated = kv.clone()
    kv_rotated[..., -_ROPE_DIM:] = apply_gptj(kv[..., -_ROPE_DIM:])

    slots_long = slot_mapping.long()
    valid_mask = slots_long >= 0
    if bool(valid_mask.any()):
        row_indices = slots_long[valid_mask] // block_size
        col_in_block = slots_long[valid_mask] % block_size
        swa_kv_cache_3d[row_indices, col_in_block] = kv_rotated[valid_mask].to(swa_kv_cache_3d.dtype)
    return q


def _install_quantized_alias(insert_fn: Callable) -> Callable[[object, object, object, object, object, object, float, float], torch.Tensor]:
    def _insert_quantized(
        q: torch.Tensor,
        kv: torch.Tensor,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        padded_heads: int,
        eps: float,
        block_size: int,
    ) -> torch.Tensor:
        # Caller convention matches upstream CUDA attention helper;
        # native Kunlun signature rearranges argument order.
        del padded_heads  # unused by this backend path
        return insert_fn(kv, cos_sin, positions, slot_mapping, q, cache, block_size, eps, None)

    return _insert_quantized


def _install_bf16_alias(insert_fn: Callable | None) -> Callable[[object, object, object, object, object, object, float, float], torch.Tensor]:
    def _insert_bf16(
        q: torch.Tensor,
        kv: torch.Tensor,
        swa_kv_cache_3d: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        eps: float,
        block_size: int,
    ) -> torch.Tensor:
        try_call = insert_fn
        if try_call is not None:
            try:
                cache_2d = swa_kv_cache_3d.view(swa_kv_cache_3d.shape[0], -1)
                try_call(
                    kv.contiguous(),
                    cos_sin,
                    positions.long(),
                    slot_mapping.long(),
                    q,
                    cache_2d,
                    block_size,
                    eps,
                    None,
                )
                return q
            except Exception as exc:  # noqa: BLE001
                WarningOnce.emit(
                    _WARNED_BF16_FALLBACK_KEY,
                    "Native BF16 Q-norm/RoPE/KV-cache failed once (%s); using PyTorch reference",
                    str(exc),
                )
        else:
            WarningOnce.emit(
                f"qkv-insert-missing:{_INSERT_OP_NAME}",
                "%s absent; BF16 cache-insert will use PyTorch reference",
                _INSERT_OP_NAME,
            )
        return _bf16_torch_reference(q, kv, swa_kv_cache_3d, slot_mapping, positions, cos_sin, eps, block_size)

    return _insert_bf16


def _applier(attention_module: object) -> None:
    """Install aliases onto :mod:`torch.ops._C`."""
    import torch as _t_local

    op_handle = _cached_native_op(_INSERT_OP_NAME)

    setattr(_t_local.ops._C, _QUANT_ATTR, _install_quantized_alias(op_handle))
    setattr(_t_local.ops._C, _BF16_CACHE_ATTR, _install_bf16_alias(op_handle))

    LOGGER.info("Wired DSV4 QKV-cache insert paths through %r", _INSERT_OP_NAME)
    setattr(attention_module, _APPLIED_SENTINEL, True)


def _predicate(attention_module: object) -> bool:
    return bool(getattr(attention_module, _APPLIED_SENTINEL, False))
