"""FP8 helpers for DeepSeek-V4: indexer Q RoPE+quant and block dequant.

Each entry tries the native XPU kernel first and falls back to a host
round-trip; the fallbacks cross PCIe on the decode hot path, so each warns
once instead of limping silently.
"""
import logging
import os

import torch

logger = logging.getLogger("vllm_kunlun")

try:
    import xspeedgate_ops  # noqa: F401  registers torch.ops.xspeedgate_ops
    # Locally hand-written op, absent from upstream builds; opt in with
    # KUNLUN_XSG_DEQUANT_FP8_BLOCKS=1.
    _HAS_XSG_DEQUANT_FP8_BLOCKS = (
        os.getenv("KUNLUN_XSG_DEQUANT_FP8_BLOCKS", "0") == "1"
        and hasattr(torch.ops.xspeedgate_ops, "dequantize_fp8_blocks")
    )
except Exception:
    _HAS_XSG_DEQUANT_FP8_BLOCKS = False

_warned_rope_fallback = [False]
_warned_dequant_fallback = [False]


def _warn_once(flag, msg):
    if not flag[0]:
        flag[0] = True
        logger.warning(msg)


_FP8_BLOCK_SIZE = 128


_HAS_FUSED_ROPE_INT8 = None


def _probe_fused_rope_int8():
    global _HAS_FUSED_ROPE_INT8
    if _HAS_FUSED_ROPE_INT8 is None:
        try:
            import kunlun_ops
            _HAS_FUSED_ROPE_INT8 = hasattr(kunlun_ops, "fused_rope_int8_quant")
        except Exception:
            _HAS_FUSED_ROPE_INT8 = False
    return _HAS_FUSED_ROPE_INT8


def fused_indexer_q_rope_quant_kunlun(
    positions,
    index_q,
    index_q_cos_sin_cache,
    index_weights,
    index_weights_softmax_scale,
    index_weights_head_scale,
    use_fp4=False,
):
    """V4 Indexer Q RoPE + INT8 quantization.

    Native path: kunlun_ops.fused_rope_int8_quant (single kernel, no CPU round-trip).
    Fallback: Python implementation with .cpu().to(fp8) (slow, PCIe bound).
    """
    assert not use_fp4, "Kunlun does not support FP4 Indexer Q"

    if _probe_fused_rope_int8():
        import kunlun_ops
        T, n_heads, head_dim = index_q.shape
        y = torch.empty_like(index_q, dtype=torch.int8)
        scale = torch.empty(T, n_heads, dtype=torch.float32, device=index_q.device)

        kunlun_ops.fused_rope_int8_quant(
            index_q.contiguous(),               # x: [T, heads, head_dim] bf16
            positions.long().contiguous(),       # positions: [T] int64
            index_q_cos_sin_cache.contiguous(),  # cos_sin: [max_pos, 64] fp32
            y,                                   # output: [T, heads, head_dim] int8
            scale,                               # output: [T, heads] fp32
            False,                               # inverse_rope=False (Q side)
            -1,                                  # rot_offset=-1 (auto: head_dim - 64)
            index_weights.float().contiguous(),  # index_weights: [T, heads] fp32
            float(index_weights_softmax_scale),  # softmax_scale
            float(index_weights_head_scale),     # head_scale
        )
        # Kernel folds the weights into scale (weight * q_scale *
        # softmax_scale * head_scale).
        return y, scale

    # Fallback: correctness only, slow (.cpu() round-trip per decode step).
    _warn_once(
        _warned_rope_fallback,
        "kunlun_ops.fused_rope_int8_quant is unavailable; V4 indexer Q "
        "RoPE+quant falls back to a host round-trip (.cpu() per decode step). "
        "This is a large slowdown, not a cosmetic one.",
    )
    rope_dim = index_q_cos_sin_cache.shape[-1]
    half_rope_dim = rope_dim // 2
    nope_dim = index_q.shape[-1] - rope_dim
    assert rope_dim % 2 == 0 and nope_dim >= 0

    cos_sin = index_q_cos_sin_cache.index_select(0, positions.long()).float()
    cos = cos_sin[..., :half_rope_dim].unsqueeze(1)
    sin = cos_sin[..., half_rope_dim:].unsqueeze(1)

    q_float = index_q.float()
    q_rot = q_float[..., nope_dim:]
    q_even = q_rot[..., 0::2]
    q_odd = q_rot[..., 1::2]
    rotated = torch.empty_like(q_rot)
    rotated[..., 0::2] = q_even * cos - q_odd * sin
    rotated[..., 1::2] = q_odd * cos + q_even * sin
    rotated = rotated.to(torch.bfloat16).float()
    q_rope = torch.cat((q_float[..., :nope_dim], rotated), dim=-1)

    fp8_max = 448.0
    q_scale = q_rope.abs().amax(dim=-1).clamp_min(1e-4) / fp8_max
    q_scale = torch.pow(2.0, torch.ceil(torch.log2(q_scale)))
    q_normalized = q_rope / q_scale.unsqueeze(-1)
    q_fp8 = q_normalized.cpu().to(torch.float8_e4m3fn).to(index_q.device)

    weights_out = index_weights.float() * q_scale
    weights_out *= index_weights_softmax_scale * index_weights_head_scale
    return q_fp8, weights_out


def dequantize_fp8_blocks(weight, weight_scale):
    """Dequantize a block-scaled FP8 matrix to BF16.

    Fast path: fully on-device XPU kernel
    ``torch.ops.xspeedgate_ops.dequantize_fp8_blocks``, which fuses fp8->bf16
    decode and per-block scale multiply in a single kernel launch (no PCIe
    round-trip). Requires ``weight`` to be contiguous and on XPU.

    Fallback: original CPU-cast path (fp8->bf16 cast on host, scale multiply
    back on device). Used when the fast op is unavailable, the weight lives on
    CPU, or the weight is only transposed-contiguous.
    """
    n, k = weight.shape
    assert n % _FP8_BLOCK_SIZE == 0
    assert k % _FP8_BLOCK_SIZE == 0
    n_blocks = n // _FP8_BLOCK_SIZE
    k_blocks = k // _FP8_BLOCK_SIZE
    assert weight_scale.shape[0] >= n_blocks
    assert weight_scale.shape[1] >= k_blocks

    # Fast path: on-device fused decode + block scale.
    if (
        _HAS_XSG_DEQUANT_FP8_BLOCKS
        and weight.is_contiguous()
        and weight.device.type != "cpu"
        and weight_scale.device == weight.device
        and weight_scale.dtype == torch.float32
    ):
        return torch.ops.xspeedgate_ops.dequantize_fp8_blocks(weight, weight_scale)

    # Fallback: CPU round-trip cast + on-device scale multiply.
    _warn_once(
        _warned_dequant_fallback,
        "xspeedgate_ops.dequantize_fp8_blocks unavailable or inputs "
        f"unsuitable (has_op={_HAS_XSG_DEQUANT_FP8_BLOCKS}, "
        f"contiguous={weight.is_contiguous()}, device={weight.device}, "
        f"scale_dtype={weight_scale.dtype}); falling back to a host round-trip. "
        "On the FP8 MoE decode path this runs per step and is catastrophic -- "
        "check that PYTHONPATH points at the XSpeedGate source tree.",
    )
    weight_scale = weight_scale[:n_blocks, :k_blocks]
    device = weight.device
    if weight.is_contiguous():
        weight_bf16 = weight.cpu().to(torch.bfloat16).to(device)
    else:
        assert weight.ndim == 2 and weight.t().is_contiguous()
        weight_bf16 = weight.t().cpu().to(torch.bfloat16).t().contiguous().to(device)
    weight_bf16 = weight_bf16.view(
        n_blocks, _FP8_BLOCK_SIZE, k_blocks, _FP8_BLOCK_SIZE
    )
    weight_bf16 = weight_bf16 * weight_scale.to(torch.bfloat16).view(
        n_blocks, 1, k_blocks, 1
    )
    return weight_bf16.reshape(n, k)
