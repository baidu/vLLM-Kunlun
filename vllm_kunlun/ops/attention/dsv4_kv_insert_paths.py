"""Kunlun-native KV-insert path implementations used by V4 attention alias wiring.

These were originally inlined as closures inside ``vllm_kunlun/__init__``\'s
``_v4_attention_alias_apply`` bootstrap hook. They are relocated here so the
package-root init stays free of concrete algorithmic implementations; the hook
body now merely imports these symbols and binds them onto ``torch.ops._C``.
"""
import logging

import torch

try:
    import kunlun_ops  # type: ignore[import]
except Exception:  # noqa: BLE001
    kunlun_ops = None


LOGGER = logging.getLogger("vllm_kunlun.ops.attention.dsv4_kv_insert_paths")


def fp8_quant_insert(
    q,
    kv,
    cache,
    slot_mapping,
    positions,
    cos_sin,
    padded_heads,
    eps,
    block_size,
):
    """FP8 packed-cache insertion -- thin adapter around native op.

    Community vLLM convention::

        q:[N,H,512]   kv:[N,512]   cache:[num_blocks,block_stride]
        slot_mapping:[M]   positions:[N]   cos_sin:[max_pos,64]

    kunlun_ops convention::

        (kv, cos_sin_cache, position_ids, slot_mapping, q, k_cache,
         cache_block_size, eps, scale=None)

    Returns whatever the underlying op returns (typically mutated q).
    """
    return kunlun_ops.fused_deepseek_v4_qnorm_rope_kv_insert(
        kv,
        cos_sin,
        positions,
        slot_mapping,
        q,
        cache,
        block_size,
        eps,
        None,
    )


_BF16_INSERT_WARNED_KEY = "dsv4-bf16-insert-native-failed"
_BF16_INSERT_WARNED = [False]


def bf16_full_cache_insert(
    q,
    kv,
    swa_kv_cache_3d,
    slot_mapping,
    positions,
    cos_sin,
    eps,
    block_size,
):
    """BF16 full-cache path: fused Q RMSNorm(no-weight)+GPT-J RoPE+KV RoPE+
    paged BF16 write--same native kernel as FP8 path with BF16 k_cache dtype;
    falls back to pure-torch reference impl log-once when native fails.
    """
    try:
        cache_2d = swa_kv_cache_3d.view(swa_kv_cache_3d.shape[0], -1)
        kunlun_ops.fused_deepseek_v4_qnorm_rope_kv_insert(
            kv, cos_sin, positions.long(), slot_mapping.long(),
            q, cache_2d, block_size, eps, None,
        )
        return q
    except Exception as exc:  # noqa: BLE001
        if not _BF16_INSERT_WARNED[0]:
            LOGGER.warning(
                "native bf16 fused_deepseek_v4_qnorm_rope_kv_insert "
                "failed (%s); using torch fallback", exc,
            )
            _BF16_INSERT_WARNED[0] = True

    num_tokens = q.shape[0]
    rope_dim = 64
    q_float = q.float()
    rms = torch.rsqrt(q_float.square().mean(dim=-1, keepdim=True) + eps)
    q.copy_((q_float * rms).to(q.dtype))

    cos_sin_selected = cos_sin[positions]
    cos_val = cos_sin_selected[:, :32]
    sin_val = cos_sin_selected[:, 32:]

    def apply_gptj_rope(x_rope):
        x1 = x_rope[..., ::2]
        x2 = x_rope[..., 1::2]
        if x_rope.ndim == 3:
            cos_b = cos_val.unsqueeze(1)
            sin_b = sin_val.unsqueeze(1)
        else:
            cos_b = cos_val
            sin_b = sin_val
        out1 = x1.float() * cos_b.float() - x2.float() * sin_b.float()
        out2 = x1.float() * sin_b.float() + x2.float() * cos_b.float()
        return torch.stack([out1, out2], dim=-1).flatten(-2).to(x_rope.dtype)

    q[..., -rope_dim:] = apply_gptj_rope(q[..., -rope_dim:])
    kv_roped = kv.clone()
    kv_roped[..., -rope_dim:] = apply_gptj_rope(kv_roped[..., -rope_dim:])

    slots = slot_mapping.to(torch.long)
    valid = slots >= 0
    if bool(valid.any()):
        v_slots = slots[valid]
        swa_kv_cache_3d[v_slots // block_size, v_slots % block_size, :] = (
            kv_roped[valid].to(swa_kv_cache_3d.dtype)
        )
    return q
