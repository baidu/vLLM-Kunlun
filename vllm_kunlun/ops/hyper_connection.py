"""Kunlun implementations for the community DeepSeek V4 mHC helpers."""

import torch

import logging as _logging

_MHC_LOG = _logging.getLogger("vllm_kunlun")

try:
    import kunlun_ops as _kunlun_ops
    _HAS_HC_PRE = hasattr(_kunlun_ops, "hc_pre_kunlun_impl")
except Exception:
    _HAS_HC_PRE = False

_mhc_pre_warned = False


def _rms_norm(x, weight, eps):
    if weight is None:
        return x
    try:
        out = torch.empty_like(x)
        torch.ops._C.rmsnorm(x, weight, out, eps)
        return out
    except Exception:
        # Fallback if native rmsnorm not available (e.g. pre-plugin-init)
        return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps).to(x.dtype) * weight


def _native_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                norm_weight=None, norm_eps=1e-6):
    outer_shape = residual.shape[:-2]
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if hc_mult != 4:
        raise NotImplementedError("Kunlun hc_pre_kunlun_impl supports only hc_mult=4")

    batch = residual.reshape(-1, hc_mult * hidden_size)
    layer_input = torch.empty(batch.shape[0], hidden_size, dtype=residual.dtype, device=residual.device)
    post_mix = torch.empty(batch.shape[0], hc_mult, dtype=residual.dtype, device=residual.device)
    comb_mix = torch.empty(batch.shape[0], hc_mult * hc_mult, dtype=residual.dtype, device=residual.device)
    import kunlun_ops
    kunlun_ops.hc_pre_kunlun_impl(
        batch.contiguous(), fn.contiguous(), hc_base, hc_scale,
        layer_input, post_mix, comb_mix, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
    )
    layer_input = _rms_norm(layer_input, norm_weight, norm_eps)
    return (
        post_mix.float().reshape(*outer_shape, hc_mult),
        comb_mix.float().reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def _torch_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
               hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
               norm_weight=None, norm_eps=1e-6):
    # This fallback preserves the same tensor contract. It is used only when
    # the native Kunlun composite is unavailable or rejects the input shape.
    outer_shape = residual.shape[:-2]
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    x = residual.reshape(-1, hc_mult * hidden_size)
    mixes = torch.nn.functional.linear(x.float(), fn.float())
    mixes = mixes * torch.rsqrt(
        x.float().square().mean(dim=-1, keepdim=True) + rms_eps
    )
    hc_mult2 = hc_mult * hc_mult
    pre_raw, post_raw, comb_raw = torch.split(mixes, (hc_mult, hc_mult, hc_mult2), dim=-1)
    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    post = torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
    comb_logits = (
        comb_raw * hc_scale[2] + hc_base[2 * hc_mult:]
    ).reshape(-1, hc_mult, hc_mult)
    comb = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    post = post * hc_post_mult_value
    layer_input = torch.sum(
        pre.unsqueeze(-1) * residual.reshape(-1, hc_mult, hidden_size).float(),
        dim=1,
    ).to(residual.dtype)
    layer_input = _rms_norm(layer_input, norm_weight, norm_eps)
    return (
        post.reshape(*outer_shape, hc_mult),
        comb.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def mhc_pre_tilelang(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                     hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                     n_splits=1, norm_weight=None, norm_eps=1e-6):
    """mHC pre-layer transform: sinkhorn-normalized connection matrices.

    Returns (post_mix [.., hc_mult], comb_mix [.., hc_mult, hc_mult],
    layer_input [.., hidden]). Uses the native composite kernel for
    hc_mult=4, falling back to the torch formula (warn-once) otherwise.
    """
    global _mhc_pre_warned
    hc_mult = residual.shape[-2]
    if _HAS_HC_PRE and hc_mult == 4:
        try:
            return _native_pre(
                residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                norm_weight, norm_eps,
            )
        except Exception as e:  # noqa: BLE001
            if not _mhc_pre_warned:
                _MHC_LOG.warning(
                    "native mhc_pre (hc_pre_kunlun_impl) failed (%s); "
                    "falling back to torch mhc_pre for the rest of the run", e)
                _mhc_pre_warned = True
    return _torch_pre(
        residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
        norm_weight, norm_eps,
    )


def mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix):
    """mHC post-layer merge: returns the next residual [.., hc_mult, hidden]."""
    import kunlun_ops

    hc_mult = residual.shape[-2]
    x_flat = x.reshape(-1, x.shape[-1]).contiguous()
    residual_flat = residual.reshape(-1, hc_mult, residual.shape[-1]).contiguous()
    post_mix_flat = post_layer_mix.reshape(-1, hc_mult).contiguous()
    comb_mix_flat = comb_res_mix.reshape(-1, hc_mult, hc_mult).contiguous()
    if residual_flat.shape[0] != x_flat.shape[0]:
        raise RuntimeError(
            "mHC post shape mismatch: "
            f"x={tuple(x.shape)} residual={tuple(residual.shape)} "
            f"post={tuple(post_layer_mix.shape)} comb={tuple(comb_res_mix.shape)} "
            f"normalized={tuple(x_flat.shape)}/{tuple(residual_flat.shape)}/"
            f"{tuple(post_mix_flat.shape)}/{tuple(comb_mix_flat.shape)}"
        )
    return kunlun_ops.mhc_post_fusion(
        x_flat, residual_flat, post_mix_flat, comb_mix_flat
    ).reshape_as(residual)


def mhc_fused_post_pre_tilelang(
    x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
    rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
    sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None, norm_eps=1e-6,
):
    """Post-then-pre fusion: returns (residual_next, post_mix, comb_mix,
    layer_input), i.e. mhc_post followed by mhc_pre in one call."""
    residual_next = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
    post_mix, comb_mix, layer_input = mhc_pre_tilelang(
        residual_next, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
        n_splits, norm_weight, norm_eps,
    )
    return residual_next, post_mix.unsqueeze(-1), comb_mix, layer_input


def hc_head_fused_kernel_kunlun(
    hs_flat,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_eps,
):
    """Final-layer mHC head: returns [num_tokens, hidden] from the fused
    kunlun_ops.fused_dpsk_v4_hc_head_nofc kernel."""
    num_tokens, hc_mult, hidden_size = hs_flat.shape
    out = torch.empty(
        num_tokens, hidden_size, dtype=hs_flat.dtype, device=hs_flat.device
    )
    if num_tokens == 0:
        return out

    import kunlun_ops

    gemm_out = torch.mm(
        hs_flat.reshape(num_tokens, hc_mult * hidden_size).float(), fn.t()
    )
    kunlun_ops.fused_dpsk_v4_hc_head_nofc(
        hs_flat,
        gemm_out,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )
    return out
