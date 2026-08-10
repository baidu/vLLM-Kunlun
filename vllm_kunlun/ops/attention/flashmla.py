# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py
import os
from typing import Optional, Tuple

import kunlun_ops
import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401

        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False

if current_platform.is_cuda():
    try:
        import vllm._flashmla_extension_C  # noqa: F401

        _flashmla_extension_C_AVAILABLE = True
    except ImportError:
        _flashmla_extension_C_AVAILABLE = False
else:
    _flashmla_extension_C_AVAILABLE = False


def is_flashmla_supported() -> Tuple[bool, Optional[str]]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    return True, None


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int = 1,
    num_heads_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Returns:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    # return flash_mla_cuda.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)
    cache_seqlens_cpu = cache_seqlens.cpu()
    return cache_seqlens_cpu, cache_seqlens


_SHAPE_SENTINEL_DONE = {}


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    head_dim_v: int = 512,
    tile_scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    # V4 kwargs
    topk_length: Optional[torch.Tensor] = None,
    attn_sink=None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    # V4 per-layer config (needed by native hybrid_attention)
    compress_ratio: int = 4,
    max_window_size: int = 128,
    com_topk: int = 512,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kunlun flash_mla_with_kvcache for DeepSeek-V4.
    Two paths:
    - Legacy V3: block_table provided -> kunlun_ops.paged_attention
    - V4: indices provided -> PyTorch naive MLA attention
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # ---- Legacy V3 path ----
    if block_table is not None and indices is None:
        output = torch.ones(
            q.size(0), q.size(1), q.size(2), head_dim_v, dtype=q.dtype, device=q.device
        )
        kv_lora_rank = head_dim_v
        qk_rope_head_dim = q.size(3) - head_dim_v
        head_dim = k_cache.shape[3]
        page_block_size = k_cache.shape[1]
        k_cache_view = k_cache.view(-1, 1, page_block_size, head_dim)

        kunlun_ops.paged_attention(
            output, q, k_cache_view, None, block_table,
            tile_scheduler_metadata, num_splits,
            False, causal, -1,
            kv_lora_rank, qk_rope_head_dim, softmax_scale, q_r=q,
        )
        return output, None

    # ---- one-shot shape sentinel (env-gated, default OFF) ----
    _sentinel_kind = "swa_only" if extra_k_cache is None else "dual"
    if (os.environ.get("KUNLUN_DSV4_SHAPE_SENTINEL", "0") == "1"
            and _sentinel_kind not in _SHAPE_SENTINEL_DONE):
        _SHAPE_SENTINEL_DONE[_sentinel_kind] = True

        def _d(name, t):
            if t is None:
                return f"    {name}: None"
            if isinstance(t, torch.Tensor):
                extra = ""
                try:
                    _st = t.untyped_storage()
                    _elems = _st.size() // t.element_size()
                    _base = t._base
                    extra = (f" storage_elems={_elems}"
                             f" numel={t.numel()}"
                             f" base_shape="
                             f"{tuple(_base.shape) if _base is not None else None}")
                except Exception as _e:
                    extra = f" (storage probe failed: {_e})"
                return (f"    {name}: shape={tuple(t.shape)} dtype={t.dtype} "
                        f"stride={t.stride()} contig={t.is_contiguous()}"
                        + extra)
            return f"    {name}: {type(t).__name__}={t!r}"

        lines = ["[DSv4-SHAPE-SENTINEL] flash_mla_with_kvcache route=" + _sentinel_kind + ":"]
        for _n, _t in (
                ("q", q), ("k_cache", k_cache), ("block_table", block_table),
                ("cache_seqlens", cache_seqlens),
                ("indices", indices), ("topk_length", topk_length),
                ("attn_sink", attn_sink),
                ("extra_k_cache", extra_k_cache),
                ("extra_indices_in_kvcache", extra_indices_in_kvcache),
                ("extra_topk_length", extra_topk_length),
                ("out", out),
                ("head_dim_v", head_dim_v),
                ("softmax_scale", softmax_scale),
                ("is_fp8_kvcache", is_fp8_kvcache),
        ):
            lines.append(_d(_n, _t))
        # Value summaries for the length/index tensors decide whether the
        # native op's kvseqlen-derived masking can reproduce these.
        for _n, _t in (("topk_length", topk_length),
                       ("extra_topk_length", extra_topk_length)):
            if isinstance(_t, torch.Tensor) and _t.numel() > 0:
                lines.append(f"    {_n} values: {_t.flatten()[:16].tolist()}")
        for _n, _t in (("indices", indices),
                       ("extra_indices_in_kvcache", extra_indices_in_kvcache)):
            if isinstance(_t, torch.Tensor) and _t.numel() > 0:
                _f = _t.flatten()
                lines.append(
                    f"    {_n}: min={int(_f.min())} max={int(_f.max())} "
                    f"n_neg={int((_f < 0).sum())} head={_f[:12].tolist()}")
        # Row 0 of each index tensor, so the padding convention (-1 vs
        # clamped vs repeated) is unambiguous.
        for _n, _t in (("indices", indices),
                       ("extra_indices_in_kvcache", extra_indices_in_kvcache)):
            if isinstance(_t, torch.Tensor) and _t.numel() > 0:
                _r = _t.reshape(_t.shape[0], -1)[0]
                lines.append(f"    {_n} row0[:32]: {_r[:32].tolist()}")
                lines.append(f"    {_n} row0[-8:]: {_r[-8:].tolist()}")
        try:
            import time as _time

            _nb, _bs = k_cache.shape[0], k_cache.shape[1]

            def _timeit(fn, reps=5):
                fn()
                torch.cuda.synchronize()
                _t0 = _time.perf_counter()
                for _ in range(reps):
                    fn()
                torch.cuda.synchronize()
                return (_time.perf_counter() - _t0) * 1e3 / reps

            # (a) current: full flatten (materializes a copy), then gather.
            _slots = indices.reshape(-1).long().clamp(0, _nb * _bs - 1)

            def _cur():
                _flat = k_cache.reshape(_nb * _bs, -1)
                return _flat[_slots]

            # (b) proposed: decompose slot -> (block, offset), gather in place.
            def _new():
                _blk = torch.div(_slots, _bs, rounding_mode="floor")
                _off = _slots - _blk * _bs
                return k_cache[_blk, _off, 0, :]

            _ms_cur = _timeit(_cur)
            _ms_new = _timeit(_new)
            _same = torch.equal(_cur().to(torch.float32),
                                _new().to(torch.float32))
            lines.append(
                f"    GATHER BENCH  flatten+gather={_ms_cur:.3f} ms  "
                f"strided_gather={_ms_new:.3f} ms  "
                f"speedup={_ms_cur / max(_ms_new, 1e-9):.1f}x  "
                f"results_identical={_same}  "
                f"n_slots={_slots.numel()}")
        except Exception as _e:
            lines.append(f"    gather bench failed: {type(_e).__name__}: {_e}")
        logger.info("\n".join(lines))

    # ---- Allocate output buffer (used by both native and torch paths) ----
    if out is not None:
        output = out
    else:
        output = torch.zeros(
            q.shape[0], q.shape[1], q.shape[2], head_dim_v,
            dtype=q.dtype, device=q.device,
        )

    # ---- Native hybrid_attention short-circuit (env-gated, default OFF,
    #        DUAL path only). For SWA-only layers the torch path is fast
    #        enough and the native op's host-side dim check rejects the
    #        1D empty tensors we'd have to pass. ----
    _native_ok = (
        os.environ.get("KUNLUN_DSV4_HYBRID_NATIVE", "0") == "1"
        and hasattr(kunlun_ops, "hybrid_attention")
        and extra_k_cache is not None
        and extra_indices_in_kvcache is not None
        and extra_topk_length is not None
    )
    if _native_ok:
        try:
            _B, _S, _H, _D = q.shape
            assert _S == 1, (
                f"native hybrid_attention decode-only path; got seq_q={_S}"
            )
            # ---- 2D view of caches: (NB, BS, 1, D) -> (NB*BS, D) ----
            _win_kv = k_cache.reshape(-1, k_cache.shape[-1])
            _com_kv = (
                extra_k_cache.reshape(-1, extra_k_cache.shape[-1])
                if extra_k_cache is not None
                else torch.empty(0, dtype=k_cache.dtype, device=q.device)
            )
            # ---- 2D view of indices: (B, 1, N) -> (B, N) ----
            # SGLang pattern: clamp to valid range so kernel never reads OOB
            # memory; the kernel uses kvseqlen/topk_length to mask invalid
            # positions out of attention.
            _win_idx = (
                indices.squeeze(1) if indices.dim() == 3 else indices
            ).contiguous().clamp_(min=0, max=_win_kv.shape[0] - 1)
            _com_idx = (
                extra_indices_in_kvcache.squeeze(1)
                if (extra_indices_in_kvcache is not None
                    and extra_indices_in_kvcache.dim() == 3)
                else extra_indices_in_kvcache
            )
            if _com_idx is not None:
                _com_idx = _com_idx.contiguous().clamp_(min=0, max=_com_kv.shape[0] - 1)
            # ---- 2D view of out: (B, 1, H, head_dim_v) -> (B, H, head_dim_v) ----
            _out2d = output[:, 0, :, :]
            # ---- qlod: cumulative q-token count, length B+1 ----
            # seq_q=1 for decode, so qlod = [0, 1, 2, ..., B]
            _qlod_xpu = torch.arange(
                _B + 1, dtype=torch.int32, device=q.device
            )
            # ---- kvseqlen: per-seq uncompressed kv length ----
            # Window tokens contribute topk_length positions.
            # Compressed tokens contribute extra_topk_length * compress_ratio
            # positions each (per wrapper docstring).
            if topk_length is not None:
                _win_len = topk_length[:_B].to(torch.int32)
                _com_len = extra_topk_length[:_B].to(torch.int32) * int(compress_ratio)
                _kvseqlen_xpu = (_win_len + _com_len).contiguous()
            else:
                _kvseqlen_xpu = torch.full(
                    (_B,), max_window_size + com_topk,
                    dtype=torch.int32, device=q.device,
                )
            _max_logits = torch.zeros(
                _B, _H, dtype=torch.float32, device=q.device,
            )
            _lse = torch.zeros(
                _B, _H, dtype=torch.float32, device=q.device,
            )
            _sink = (
                attn_sink
                if isinstance(attn_sink, torch.Tensor)
                else torch.empty(0, dtype=torch.float32, device=q.device)
            )
            # NOTE: qlod_cpu and kvseqlen_cpu must be DISTINCT CPU tensors
            # (not the same storage as the xpu ones). qlod_cpu=qlod_xpu was
            # the root cause of the native worker crash -- isolation test
            # variant B proved it. .cpu() forces a host-side copy.
            _qlod_cpu = _qlod_xpu.cpu()
            _kvseqlen_cpu = _kvseqlen_xpu.cpu()

            kunlun_ops.hybrid_attention(
                q=q[:, 0, :, :],
                win_kv_cache=_win_kv,
                win_indices=_win_idx,
                com_kv_cache=_com_kv,
                com_indices=_com_idx if _com_idx is not None
                             else torch.empty(
                                 0, dtype=_win_idx.dtype, device=q.device),
                o=_out2d,
                max_logits=_max_logits,
                lse=_lse,
                qlod_cpu=_qlod_cpu,
                qlod_xpu=_qlod_xpu,
                kvseqlen_cpu=_kvseqlen_cpu,
                kvseqlen_xpu=_kvseqlen_xpu,
                sm_scale=float(softmax_scale),
                is_causal=bool(causal),
                max_window_size=int(max_window_size),
                compress_ratio=int(compress_ratio),
                com_topk=int(com_topk) if _com_idx is not None else 0,
                attn_sink=_sink,
                side_stream=-1,
                use_xfa_boost=False,
            )
            return output, _max_logits
        except Exception as _e:
            if not getattr(flash_mla_with_kvcache, "_native_warned", False):
                logger.warning(
                    "[DSv4-NATIVE-ATTN] hybrid_attention failed, "
                    "falling back to torch: %s: %s",
                    type(_e).__name__, _e,
                )
                flash_mla_with_kvcache._native_warned = True

    # ---- V4 path: Vectorized sparse MLA (BF16 cache only; no per-token loop) ----
    batch_size = q.shape[0]
    seq_q = q.shape[1]
    num_heads = q.shape[2]
    d_qk = q.shape[3]
    kv_lora_rank = head_dim_v  # 512
    qk_rope_head_dim = d_qk - kv_lora_rank  # 64 (or 0 if d_qk==512)

    if indices is None or topk_length is None:
        # Returning zeros here silently produces wrong attention output for the
        # rest of the run (a single warning is easy to miss). Fail loudly.
        raise RuntimeError(
            "[KunlunFlashMLA] V4 sparse MLA needs both `indices` and "
            f"`topk_length` (got indices={indices is not None}, "
            f"topk_length={topk_length is not None}); refusing to return zeros."
        )

    # Flatten paged caches to [total_slots, dim] bf16 (device-side, no D2H)
    def _flatten_cache(cache_tensor, target_dim):
        total = cache_tensor.shape[0] * cache_tensor.shape[1]
        flat = cache_tensor.reshape(total, -1)

        if cache_tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
            flat_kv = flat.to(torch.bfloat16)
            if flat_kv.shape[1] > target_dim:
                return flat_kv[:, :target_dim]
            elif flat_kv.shape[1] < target_dim:
                pad = torch.zeros(total, target_dim - flat_kv.shape[1],
                                  dtype=torch.bfloat16, device=flat.device)
                return torch.cat([flat_kv, pad], dim=1)
            return flat_kv
        elif cache_tensor.dtype == torch.uint8:
            # FP8 e4m3fn dequant (vectorized, no Python loop)
            fp8_data = flat[:, :target_dim].contiguous()
            sign = ((fp8_data >> 7) & 1).to(torch.int32)
            exp_bits = ((fp8_data >> 3) & 0x0F).to(torch.int32)
            mant_bits = (fp8_data & 0x07).to(torch.int32)
            is_normal = (exp_bits > 0)
            is_zero = (fp8_data == 0) | (fp8_data == 0x80)
            mantissa_n = (8 + mant_bits).float() / 8.0
            exponent_n = (exp_bits - 7).float()
            val_normal = mantissa_n * torch.pow(2.0, exponent_n)
            val_subnormal = mant_bits.float() / 8.0 * (2.0 ** -6.0)
            result = torch.where(is_normal, val_normal, val_subnormal)
            result = torch.where(is_zero, torch.zeros_like(result), result)
            result = result * (1 - 2 * sign.float())
            return result.to(torch.bfloat16)
        else:
            # Unknown dtype: try to cast
            flat_kv = flat.to(torch.bfloat16)
            if flat_kv.shape[1] > target_dim:
                return flat_kv[:, :target_dim]
            elif flat_kv.shape[1] < target_dim:
                pad = torch.zeros(total, target_dim - flat_kv.shape[1],
                                  dtype=torch.bfloat16, device=flat.device)
                return torch.cat([flat_kv, pad], dim=1)
            return flat_kv

    flat_swa = _flatten_cache(k_cache, d_qk)
    swa_total = flat_swa.shape[0]

    flat_comp = None
    if extra_k_cache is not None:
        flat_comp = _flatten_cache(extra_k_cache, d_qk)

    # Parse indices: [B, 1, N] -> [B, N]
    swa_idx = indices.squeeze(1) if indices.dim() == 3 else indices
    comp_idx = None
    if extra_indices_in_kvcache is not None:
        comp_idx = extra_indices_in_kvcache.squeeze(1) if extra_indices_in_kvcache.dim() == 3 else extra_indices_in_kvcache

    # Vectorized per-batch gather + attention (no .item(), no per-token loop)
    # For decode: seq_q=1, so we process q[:, 0] directly
    q_flat = q[:, 0].float()  # [B, H, d_qk]
    q_c = q_flat[:, :, :kv_lora_rank]  # [B, H, 512]
    q_r = q_flat[:, :, kv_lora_rank:kv_lora_rank + qk_rope_head_dim] if qk_rope_head_dim > 0 else None  # [B, H, 64]

    for b in range(batch_size):
        # Gather using tensor indexing (no .item() for lengths - use full index width)
        swa_len = topk_length[b]  # scalar tensor on device
        parts = []
        valid_parts = []

        # SWA gather: padded indices are clamped for safe memory access, then
        # excluded from attention by the per-request valid-length mask.
        swa_indices_b = swa_idx[b].long().clamp(0, swa_total - 1)
        parts.append(flat_swa[swa_indices_b])  # [swa_width, d_qk]
        valid_parts.append(
            torch.arange(swa_indices_b.shape[0], device=q.device) < swa_len
        )

        # Compressed gather
        if flat_comp is not None and comp_idx is not None:
            if extra_topk_length is None:
                raise RuntimeError(
                    "extra_topk_length is required with compressed sparse indices"
                )
            comp_indices_b = comp_idx[b].long().clamp(0, flat_comp.shape[0] - 1)
            parts.append(flat_comp[comp_indices_b])  # [comp_width, d_qk]
            valid_parts.append(
                torch.arange(comp_indices_b.shape[0], device=q.device)
                < extra_topk_length[b]
            )

        # Concatenate: [total_kv, d_qk]
        kv_all = torch.cat(parts, dim=0).float()
        valid_mask = torch.cat(valid_parts)
        kv_c = kv_all[:, :kv_lora_rank]  # [K, 512]
        kv_r = kv_all[:, kv_lora_rank:kv_lora_rank + qk_rope_head_dim] if qk_rope_head_dim > 0 else None

        # MLA score: [H, K] = [H, 512] @ [512, K] + [H, 64] @ [64, K]
        scores = torch.mm(q_c[b], kv_c.T)  # [H, K]
        if q_r is not None and kv_r is not None:
            scores = scores + torch.mm(q_r[b], kv_r.T)
        scores = scores * softmax_scale

        # Keep one safe slot for all-padding rows so softmax stays finite, then
        # explicitly zero the output. For non-empty rows, only valid slots
        # participate in softmax.
        has_valid = valid_mask.any()
        safe_valid_mask = valid_mask | (
            (~has_valid)
            & (torch.arange(valid_mask.shape[0], device=q.device) == 0)
        )
        scores = scores.masked_fill(~safe_valid_mask.unsqueeze(0), float("-inf"))

        # Softmax + output
        attn_weights = torch.softmax(scores, dim=-1)  # [H, K]
        attn_out = torch.mm(attn_weights, kv_c)  # [H, 512]
        attn_out = attn_out * has_valid.to(attn_out.dtype)

        # Attention sink
        if attn_sink is not None:
            lse = torch.logsumexp(scores, dim=-1)  # [H]
            sink_scale = torch.sigmoid(lse - attn_sink[:num_heads].float())
            attn_out = attn_out * sink_scale.unsqueeze(-1)

        output[b, 0, :, :kv_lora_rank] = attn_out.to(output.dtype)

    return output, None


def kunlun_flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_seqlens_cpu: torch.Tensor,
    head_dim_v: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    max_seq_kv: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_tokens_kv, head_dim).
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.
        is_fp8_kvcache: bool. Whether the k_cache and v_cache are in fp8 format.
        indices: (batch_size, seq_len_q, topk), torch.int32. If not None, sparse attention will be enabled, and only tokens in the `indices` array will be attended to. Invalid indices should be set to -1 or numbers >= total_seq_len_kv.
        max_seq_kv: seq中最大的kv长度

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        max_logits:  (batch_size, seq_len_q, num_heads_q), torch.float32.
        p_sums:  (batch_size, seq_len_q, num_heads_q), torch.float32.
    """
    assert not is_fp8_kvcache, "By now, the kernel does not support uint8 kv cache."
    assert (
        q.shape[1] <= 2
    ), "kunlun_ops.fwd_kvcache_mla only support seq_len_q <= 2 for now."
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if indices is not None:
        # NOTE (zyongye): sparse attention is also causal
        # since it only attend to the tokens before
        # but here `causal` should not be specified
        assert not causal, "causal must be `false` if sparse attention is enabled."

    batch_size, seq_len_q, num_heads_q, head_dim = q.shape
    kv_lora_rank = head_dim_v

    out = torch.zeros(
        [batch_size, seq_len_q, num_heads_q, kv_lora_rank],
        dtype=q.dtype,
        device=q.device,
    )
    max_logits = torch.zeros(
        [batch_size, seq_len_q, num_heads_q], dtype=torch.float32, device=q.device
    )
    p_sums = torch.zeros(
        [batch_size, seq_len_q, num_heads_q], dtype=torch.float32, device=q.device
    )

    torch.ops._C.fwd_kvcache_mla(
        q_c=q,
        kv_cache=k_cache,
        indices=indices,
        kv_lod_cpu=cache_seqlens_cpu,
        max_seq_kv=max_seq_kv,
        softmax_scale=softmax_scale,
        # q_r=q_r,
        # pe_cache=pe_cache,
        out=out,
        max_logits=max_logits,
        p_sums=p_sums,
        kv_lod_xpu=cache_seqlens,
    )

    return out, max_logits, p_sums


def flash_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    q_lod_xpu: Optional[torch.Tensor] = None,
    q_lod_cpu: Optional[torch.Tensor] = None,
    d_v: int = 512,
    # V4 new kwargs
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
    - q: [s_q, h_q, d_qk], bfloat16
    - kv: [s_kv, d_qk], bfloat16
    - indices: [s_q, h_kv, topk], int32.
        Invalid indices should be set to -1 or numbers >= s_kv
    - sm_scale: float
    - q_lod_xpu: [batch+1], int32, q的每个seq长度的累加信息, 长度为batch_num + 1 (为空则表示q定长).
    - d_v: The dimension of value vectors. Can only be 512

    Returns:
    - (output, max_logits, lse)
        About the definition of output,
        max_logits and lse, please refer to README.md
    - output: [s_q, h_q, d_v], bfloat16
    - max_logits:  [s_q, h_q], float
    - lse: [s_q, h_q], float, 2-based log-sum-exp
    """
    s_q, h_q, d_qk = q.shape

    # [kunlun-patch] 3D-kv naive PyTorch MLA fallback
    #
    # DSv4 nvidia._forward_prefill calls this wrapper with
    #   kv     = kv_workspace.view(-1, 1, d_qk)   # 3-D
    #   indices= combined_indices.unsqueeze(1)    # [s_q, 1, topk]
    #   topk_length = combined_lens
    # The kunlun sparse_prefill_fwd_opt kernel asserts kv.dim() == 2 and
    # historically rejected this V4 layout on Kunlun, so we fall back to a
    # naive PyTorch MLA attention over the gathered bf16 workspace. This
    # matches the source-patched implementation that produced tokens on the
    # old pod. The 2-D fast path (V3.2 indexer sparse prefill) is unchanged.
    if kv.dim() == 3:
        import torch as _torch
        d_qk_v = q.shape[-1]
        s_q_v = q.shape[0]
        h_q_v = q.shape[1]
        if out is None:
            out = _torch.zeros(
                [s_q_v, h_q_v, d_v], dtype=q.dtype, device=q.device
            )
        kv_flat = kv.reshape(-1, d_qk_v)
        idx2d = indices.reshape(s_q_v, -1)
        kv_lora_rank = d_v
        rope_dim = d_qk_v - kv_lora_rank
        lens = topk_length
        lse_for_sink = _torch.zeros(
            [s_q_v, h_q_v], dtype=_torch.float32, device=q.device
        )
        for _t in range(s_q_v):
            valid = int(lens[_t].item()) if lens is not None else idx2d.shape[1]
            if valid <= 0:
                continue
            vi = idx2d[_t, :valid].long().clamp(0, kv_flat.shape[0] - 1)
            kvg = kv_flat[vi].float()
            kv_c = kvg[:, :kv_lora_rank]
            kv_r = kvg[:, kv_lora_rank:kv_lora_rank + rope_dim]
            q_tok = q[_t].float()
            q_c = q_tok[:, :kv_lora_rank]
            q_r = q_tok[:, kv_lora_rank:kv_lora_rank + rope_dim]
            scores = (q_c @ kv_c.T + q_r @ kv_r.T) * sm_scale
            w = _torch.softmax(scores, dim=-1)
            lse_for_sink[_t] = _torch.logsumexp(scores, dim=-1)
            ao = w @ kv_c
            out[_t, :, :kv_lora_rank] = ao.to(out.dtype)
        if isinstance(attn_sink, _torch.Tensor):
            sink = attn_sink[:h_q_v].to(
                device=out.device, dtype=_torch.float32
            )
            sink_scale = _torch.sigmoid(lse_for_sink - sink.unsqueeze(0))
            out.mul_(sink_scale.unsqueeze(-1).to(out.dtype))
        max_logits = _torch.zeros(
            [s_q_v, h_q_v], dtype=_torch.float32, device=q.device
        )
        lse = lse_for_sink
        return out, max_logits, lse

    if out is None:
        out = torch.zeros([s_q, h_q, d_v], dtype=q.dtype, device=q.device)
    max_logits = torch.zeros([s_q, h_q], dtype=torch.float32, device=q.device)
    lse = torch.zeros([s_q, h_q], dtype=torch.float32, device=q.device)

    # If q_lod not provided (V4 path), create a simple [0, s_q] lod
    if q_lod_xpu is None:
        q_lod_cpu = torch.tensor([0, s_q], dtype=torch.int32)
        q_lod_xpu = q_lod_cpu.to(q.device)
    if q_lod_cpu is None:
        q_lod_cpu = q_lod_xpu.cpu()

    torch.ops._C.sparse_prefill_fwd_opt(
        q=q,
        kv=kv,
        indices=indices,
        qlod_cpu=q_lod_cpu,
        qlod_xpu=q_lod_xpu,
        kvlod_cpu=q_lod_cpu,
        kvlod_xpu=q_lod_xpu,
        sm_scale=sm_scale,
        d_v=d_v,
        is_causal=True,  # aiak这个值为true，这是为啥
        out=out,
        max_logits=max_logits,
        lse=lse,
    )

    # NOTE: Compared with torch.ops._flashmla_C.sparse_prefill_fwd,
    # out_scale = 1 / math.log2(math.e)
    # gpu_max_logits * out_scale = kunlun_lse
    # gpu_lse * out_scale = kunlun_lse
    lse = lse.float()
    if isinstance(attn_sink, torch.Tensor):
        sink = attn_sink[:h_q].to(device=lse.device, dtype=torch.float32)
        sink_scale = torch.sigmoid(lse - sink.unsqueeze(0))
        out.mul_(sink_scale.unsqueeze(-1).to(out.dtype))
    return out, max_logits.float(), lse


#
# TODO: Add fake functions
#
# @register_fake("_flashmla_C::get_mla_metadata")
# def _get_mla_metadata_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#
# @register_fake("_flashmla_C::fwd_kvcache_mla")
# def _fwd_kvcache_mla_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#
