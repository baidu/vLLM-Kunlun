# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py
import os
import time
from typing import Optional, Tuple

import kunlun_ops
import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_DSV4_DEBUG_ENABLED = os.getenv("KUNLUN_DSV4_DEBUG", "0") == "1"
_DSV4_DEBUG_EVERY = max(1, int(os.getenv("KUNLUN_DSV4_DEBUG_EVERY", "1")))
_DSV4_DEBUG_CALL_ID = 0


def _dsv4_debug(stage: str, event: str, call_id: int, **fields: object) -> None:
    if not _DSV4_DEBUG_ENABLED:
        return
    if event == "end" and call_id % _DSV4_DEBUG_EVERY:
        return
    rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "?"))
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"[DSV4_DEBUG] rank={rank} pid={os.getpid()} call={call_id} "
        f"stage={stage} event={event} {details}",
        flush=True,
    )


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


# ---- Static CPU lod cache (cudagraph FULL capture safety) ----
# kunlun_ops.hybrid_attention's cpp wrapper reads vsl.kv_lod_vp.cpu[i] in a
# host-side for-loop to compute max_seqlen. Reading CPU memory is fine, but
# the .cpu() call that produces the CPU tensor triggers a D2H sync, which
# breaks cudagraph capture.
#
# Workaround: pre-allocate the CPU lod tensor once per shape; update its
# contents via .copy_() OUTSIDE capture (during warmup); during capture/replay
# the static tensor's data_ptr is read directly by the cpp wrapper -- no D2H
# needed. This mirrors SGLang's per-BS metadata cache + copy_() replay pattern.
_static_qlod_cpu_cache: dict = {}
_static_kvseqlen_cpu_cache: dict = {}


def _get_static_cpu_lod(xpu_lod, cache, bound=None):
    """Return pre-allocated CPU lod; update from xpu only if NOT capturing.

    `bound` is a static upper bound on the per-seq kv length (the topk
    width). During capture we write the bound via a host memset (legal inside
    capture): the kernel honours the per-row bounds from the xpu lod,
    refreshed on every replay, so the cpu copy is only a scheduling hint.
    """
    key = (xpu_lod.shape[0], xpu_lod.numel(), xpu_lod.dtype)
    if key not in cache:
        cache[key] = torch.empty(
            xpu_lod.shape, dtype=xpu_lod.dtype,
            device='cpu', pin_memory=True,
        )
    static = cache[key]
    _cap = torch.cuda.is_current_stream_capturing()
    if not _cap:
        # D2H sync -- OK outside capture (warmup, decode non-capture path)
        static.copy_(xpu_lod)
    elif bound is not None:
        static.fill_(int(bound))
    return static


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


# ---- V4 capture-safe native path (SGLang-style: fwd_kvcache_mla family + LSE merge) ----
# NOTE: attention_merge_stage's docstring says log2/exp2, but empirical testing
# (v_a=v_b=v, s_a=s_b=s -> s_merged = s + ln(2)) shows it uses natural log
# (exp/log) internally, NOT log2. Do not apply a base-2 conversion here.
#
# Per-(B,) static kv_lod caches. Shape [batch_num] (per-seq lengths, not cumsum) per
# kunlun_ops._attention.py:fwd_kvcache_mla docstring. Each entry holds a pre-allocated
# int32 [B] tensor on the same device as the source lens.
#
# IMPLEMENTATION NOTE — why we DO NOT gate on `is_current_stream_capturing`:
# only the static buffer is captured, but the COPY kernel that writes into it
# is also captured. At replay, the copy_ reads `lens` from its CURRENT data_ptr,
# so the static buffer is refreshed every step (not pinned to the first capture).
# This is the only way to keep per-row lod values fresh as the sliding-window
# counter `topk_length` grows by 1 per decode step.
_static_kvlod_swa_xpu_cache: dict = {}
_static_kvlod_swa_cpu_cache: dict = {}
_static_kvlod_com_xpu_cache: dict = {}
_static_kvlod_com_cpu_cache: dict = {}

# Pre-allocated int32 [B, 1, topk] clamped-index buffers.
#
# `kv_lod` is a per-row mask (slots j >= topk_length[i] are never read), so
# this pre-fill is belt-and-braces only: it keeps the buffer free of the -1
# padding the kernel would treat as an OOB row index.
_static_clamped_swa_idx_cache: dict = {}
_static_clamped_com_idx_cache: dict = {}

# Pre-allocated output buffers for kunlun_flash_mla_with_kvcache (V4 production decode).
# Same shape -> same tensor -> stable data_ptr across cudagraph FULL replays.
_kunlun_mla_out_cache: dict = {}

# Pre-allocated contiguous [rows, D] staging buffers for the kv-cache.
#
# WHY. The production kv-cache view is [NB, BS, 1, D] whose block pitch is larger
# than BS*D (each layer is a column slice of a shared pool), so
# `k_cache.reshape(-1, D)` MATERIALISES a copy -- 1.80 GiB for the SWA cache at
# NB=29515. Outside capture the caching allocator recycles that block; during
# cudagraph capture the allocation is served from the graph private pool, which
# cannot reuse it, and FULL capture died with `OutOfMemoryError: Tried to
# allocate 1.80 GiB ... 5.80 GiB allocated in private pools`.
#
# WHY NOT SOMETHING CHEAPER (measured at production shapes):
#   reshape(-1, D)                   2.079 ms   <- allocates
#   static.copy_(cache_4d)           2.077 ms   <- allocates nothing
#   gather only the ~32 MiB of rows
#   the kernel reads, then arange
#   indices                          2.188 ms + kernel 0.230 ms
# The bulk copy already runs at full HBM bandwidth; the strided advanced-index
# gather does not (~0.9 GB/s), so materialising less is not faster. And
# `fwd_kvcache_mla` ignores `kv_cache` strides (verified: a strided view is
# accepted but silently read as if contiguous), so the copy cannot be skipped
# by passing the paged view. Hence: keep the copy, move only its ALLOCATION out
# of the graph. Eliminating it needs a kernel that takes the paged layout.
_static_flat_cache: dict = {}


def _flatten_cache_static(cache_4d: torch.Tensor) -> torch.Tensor:
    """Return a contiguous [rows, D] view of a [NB, BS, 1, D] kv-cache.

    Free when `cache_4d` is contiguous. Otherwise copies into a persistent
    buffer, so nothing is allocated while a graph is being captured. The copy_
    is recorded in the graph and reads `cache_4d` at replay time, so the
    staging buffer holds fresh tokens on every step.
    """
    last = cache_4d.shape[-1]
    rows = cache_4d.numel() // last
    if cache_4d.is_contiguous():
        return cache_4d.view(rows, last)
    key = (rows, last, cache_4d.dtype, cache_4d.device.index)
    buf = _static_flat_cache.get(key)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "[DSv4-NATIVE-ATTN] no kv-cache staging buffer for shape "
                f"{tuple(cache_4d.shape)}; it must be allocated before capture"
            )
        buf = torch.empty(rows, last, dtype=cache_4d.dtype,
                          device=cache_4d.device)
        _static_flat_cache[key] = buf
    buf.view(cache_4d.shape).copy_(cache_4d)
    return buf


def _get_kunlun_mla_output_buffers(
    batch_size: int,
    seq_len_q: int,
    num_heads_q: int,
    head_dim_v: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple:
    key = (batch_size, seq_len_q, num_heads_q, head_dim_v, dtype, str(device))
    bufs = _kunlun_mla_out_cache.get(key)
    if bufs is None:
        out = torch.empty([batch_size, seq_len_q, num_heads_q, head_dim_v], dtype=dtype, device=device)
        max_logits = torch.empty([batch_size, seq_len_q, num_heads_q], dtype=torch.float32, device=device)
        p_sums = torch.empty([batch_size, seq_len_q, num_heads_q], dtype=torch.float32, device=device)
        bufs = (out, max_logits, p_sums)
        _kunlun_mla_out_cache[key] = bufs
    return bufs


def _build_static_per_seq_lens(lens: torch.Tensor, cache: dict) -> torch.Tensor:
    """Return a static int32 [B] tensor of per-seq kv lengths (NOT cumsum).

    Always refreshes contents from `lens` (GPU-side copy_; no D2H sync, so it
    is capture-safe). During capture, the copy_ kernel is recorded as part of
    the graph; at replay, the kernel reads the CURRENT `lens` data_ptr and
    writes fresh values into the static buffer, so the downstream kernel call
    sees up-to-date per-seq lengths on every replay step (not the stale values
    from the first capture). See `_prepare_static_clamped_indices` for the
    same pattern applied to indices.
    """
    key = (lens.shape[0],)
    if key not in cache:
        cache[key] = torch.empty(
            lens.shape, dtype=torch.int32, device=lens.device,
        )
    static = cache[key]
    # GPU-side copy_; capture-safe. The captured copy_ reads `lens` at REPLAY
    # time, so the static buffer is refreshed per step (not pinned to the
    # capture-time value).
    static.copy_(lens.to(torch.int32))
    return static


def _prepare_static_clamped_indices(
    src_indices_3d: torch.Tensor,  # [B, 1, topk] int32 (already clamped to cache size)
    lens: torch.Tensor,            # [B] int32 per-seq valid length
    cache: dict,
) -> torch.Tensor:
    """Pre-fill j >= lens[i] slots with the last-valid-index for that row.

    Returns static int32 [B, 1, topk] tensor.

    The clamp computation is GPU-side only (clamp, gather, where, copy_); it is
    capture-safe. We DELIBERATELY do not gate on `is_current_stream_capturing`
    so the captured clamp kernel replays with fresh `src_indices_3d` and writes
    fresh values into the static buffer. The first decode (where the cmp
    values happen to be captured) and the second decode (where they must
    change because topk_length grew by 1) both produce the correct result.
    """
    B, _, TOPK = src_indices_3d.shape
    key = (B, TOPK)
    if key not in cache:
        cache[key] = torch.empty(
            src_indices_3d.shape, dtype=torch.int32, device=src_indices_3d.device,
        )
    static = cache[key]
    src2d = src_indices_3d.squeeze(1)  # [B, TOPK]
    safe = src2d.clamp(min=0)  # safety; assume upstream already clamped to cache size
    # last valid index per row: clamp to 0 when lens==0 so the gather is well-defined
    last = (lens.long() - 1).clamp(min=0)  # [B]
    tail_fill = safe.gather(1, last.unsqueeze(1).expand(-1, TOPK))  # [B, TOPK]
    arange = torch.arange(TOPK, device=safe.device).unsqueeze(0)  # [1, TOPK]
    valid = arange < lens.long().unsqueeze(1)  # [B, TOPK]
    clamped = torch.where(valid, safe, tail_fill).to(torch.int32)
    static.copy_(clamped.unsqueeze(1))
    return static


# DSv4 packs RoPE as a *sub-slice* of the 512-dim latent (config: head_dim=512,
# qk_rope_head_dim=64), so the attention head dim equals head_dim_v and there is
# no RoPE tail to peel off. fwd_kvcache_mla's packed mode requires
# rope_head_dim = q.shape[-1] - out.shape[-1] > 0 and rejects the V4 geometry
# with "decode_attention_dsa failed ret=1". Its non-packed mode takes the RoPE
# part as a separate (q_r, pe_cache) pair, so we feed a zero-filled 1-wide RoPE
# side channel: it contributes exactly 0 to every score while letting the kernel
# treat the full 512 dims as the "nope" part for both scores and output.
_ZERO_ROPE_WIDTH = 1
_zero_pe_cache: dict = {}
_zero_q_rope_cache: dict = {}


def _zero_rope_side_channel(q: torch.Tensor, kv_rows: int, kv_lora: int) -> dict:
    """fwd_kvcache_mla kwargs for the zero RoPE side channel.

    Empty dict when q already carries a real RoPE tail (q.shape[-1] > kv_lora),
    i.e. the kernel's packed mode applies as-is.
    """
    if q.shape[-1] > kv_lora:
        return {}
    qkey = (q.shape[0], q.shape[1], q.shape[2], q.dtype, q.device)
    q_r = _zero_q_rope_cache.get(qkey)
    if q_r is None:
        q_r = torch.zeros(
            q.shape[0], q.shape[1], q.shape[2], _ZERO_ROPE_WIDTH,
            dtype=q.dtype, device=q.device,
        )
        _zero_q_rope_cache[qkey] = q_r
    ckey = (kv_rows, q.dtype, q.device)
    pe = _zero_pe_cache.get(ckey)
    if pe is None:
        pe = torch.zeros(
            kv_rows, _ZERO_ROPE_WIDTH, dtype=q.dtype, device=q.device)
        _zero_pe_cache[ckey] = pe
    return {"q_r": q_r, "pe_cache": pe}


# ---- Static auxiliary buffer caches for hybrid_attention (cudagraph-safe) ----
_static_qlod_cpu_ha: dict = {}
_static_qlod_xpu_ha: dict = {}
_static_kvseqlen_cpu_ha: dict = {}
_static_kvseqlen_xpu_ha: dict = {}
_static_out_ha: dict = {}
_static_ml_ha: dict = {}
_static_lse_ha: dict = {}


def _v4_hybrid_attention_fused(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink,
    extra_k_cache,
    extra_indices_in_kvcache,
    extra_topk_length,
    head_dim_v: int,
    softmax_scale: float,
    output: torch.Tensor,
    compress_ratio: int,
    max_window_size: int,
    com_topk: int,
):
    """Fused V4 decode attention via kunlun_ops.hybrid_attention (single kernel)."""
    _B = q.shape[0]
    _H = q.shape[2]
    _D = q.shape[3]
    KV_LORA_RANK = head_dim_v
    q_3d = q.squeeze(1)
    flat_swa = _flatten_cache_static(k_cache)
    swa_idx = indices.squeeze(1) if indices.dim() == 3 else indices
    swa_idx = swa_idx[:, :max_window_size].contiguous()
    swa_idx = swa_idx.clamp(min=0, max=flat_swa.shape[0] - 1)
    if extra_k_cache is not None and extra_topk_length is not None:
        flat_com = _flatten_cache_static(extra_k_cache)
        com_idx = (extra_indices_in_kvcache.squeeze(1)
                   if extra_indices_in_kvcache.dim() == 3
                   else extra_indices_in_kvcache)
        actual_com_topk = com_topk
        if actual_com_topk > 0 and actual_com_topk % 2 != 0:
            actual_com_topk = actual_com_topk + 1
            if actual_com_topk > com_idx.shape[-1]:
                actual_com_topk = actual_com_topk - 2
        if com_idx.shape[-1] > actual_com_topk:
            com_idx = com_idx[:, :actual_com_topk].contiguous()
        com_idx = com_idx.clamp(min=0, max=flat_com.shape[0] - 1)
        effective_compress_ratio = compress_ratio
    else:
        flat_com = torch.empty((0, _D), dtype=k_cache.dtype, device=q.device)
        com_idx = torch.empty((_B, 0), dtype=torch.int32, device=q.device)
        actual_com_topk = 0
        effective_compress_ratio = max(compress_ratio, 1)
    out_key = (_B, _H, KV_LORA_RANK, q.dtype, str(q.device))
    if out_key not in _static_out_ha:
        _static_out_ha[out_key] = torch.empty(_B, _H, KV_LORA_RANK, dtype=q.dtype, device=q.device)
        _static_ml_ha[out_key] = torch.empty(_B, _H, dtype=torch.float32, device=q.device)
        _static_lse_ha[out_key] = torch.empty(_B, _H, dtype=torch.float32, device=q.device)
    out_3d = _static_out_ha[out_key]
    max_logits = _static_ml_ha[out_key]
    lse_buf = _static_lse_ha[out_key]
    qlod_key = (_B,)
    if qlod_key not in _static_qlod_cpu_ha:
        _static_qlod_cpu_ha[qlod_key] = torch.arange(_B + 1, dtype=torch.int32)
        _static_qlod_xpu_ha[qlod_key] = _static_qlod_cpu_ha[qlod_key].to(q.device)
    qlod_cpu = _static_qlod_cpu_ha[qlod_key]
    qlod_xpu = _static_qlod_xpu_ha[qlod_key]
    if extra_topk_length is not None:
        kvseqlen_xpu = topk_length[:_B].int() + extra_topk_length[:_B].int() * compress_ratio
    else:
        kvseqlen_xpu = topk_length[:_B].int()
    kvseqlen_key = (_B,)
    if kvseqlen_key not in _static_kvseqlen_cpu_ha:
        _static_kvseqlen_cpu_ha[kvseqlen_key] = torch.empty(_B, dtype=torch.int32)
    kvseqlen_cpu = _static_kvseqlen_cpu_ha[kvseqlen_key]
    # kvseqlen_cpu = scheduling hint for C++ wrapper; always fill with upper bound
    # to avoid D2H sync issues during cudagraph capture/warmup.
    kvseqlen_cpu.fill_(max_window_size + actual_com_topk * compress_ratio)
    kunlun_ops.hybrid_attention(
        q=q_3d, win_kv_cache=flat_swa, win_indices=swa_idx,
        com_kv_cache=flat_com, com_indices=com_idx,
        o=out_3d, max_logits=max_logits, lse=lse_buf,
        qlod_cpu=qlod_cpu, qlod_xpu=qlod_xpu,
        kvseqlen_cpu=kvseqlen_cpu, kvseqlen_xpu=kvseqlen_xpu,
        sm_scale=float(softmax_scale), is_causal=True,
        max_window_size=max_window_size, compress_ratio=effective_compress_ratio,
        com_topk=actual_com_topk,
        attn_sink=attn_sink if isinstance(attn_sink, torch.Tensor) else torch.empty(0),
    )
    output[:, 0, :, :KV_LORA_RANK] = out_3d
    return output, None



def _v4_sparse_native_path(
    q: torch.Tensor,                 # [B, 1, H, D] (B=num_decode_tokens)
    k_cache: torch.Tensor,           # [NB_swa, BS_swa, 1, D]
    indices: torch.Tensor,           # [B, 1, window_size] int32
    topk_length: torch.Tensor,       # [B] int32, valid SWA slots per row
    attn_sink,                       # None | [H] fp32
    extra_k_cache,                   # None | [NB_com, BS_com, 1, D] (only when compress_ratio>1)
    extra_indices_in_kvcache,        # None | [B, 1, com_topk] int32
    extra_topk_length,               # None | [B] int32
    head_dim_v: int,
    softmax_scale: float,
    output: torch.Tensor,            # [B, 1, H, head_dim_v] pre-allocated by caller
    compress_ratio: int,
    max_window_size: int,
    com_topk: int,
) -> Tuple[torch.Tensor, None]:
    """Capture-safe V4 sparse MLA decode path.

    Mirrors SGLang DeepseekV4BackendRadix.forward which routes through the
    capture-safe `paged_attention` family. Our Kunlun equivalent is:
        SWA part   -> kunlun_ops.fwd_kvcache_mla   (capture-safe, sparse via indices)
        Compressed -> kunlun_ops.fwd_kvcache_mla   (same op, second call)
        Combine    -> kunlun_ops.attention_merge_stage (LSE-based merge)
    All metadata tensors are static pre-allocated buffers whose contents are
    refreshed only OUTSIDE capture (mirrors SGLang's per-BS metadata cache).
    """
    global _DSV4_DEBUG_CALL_ID
    _DSV4_DEBUG_CALL_ID += 1
    debug_call_id = _DSV4_DEBUG_CALL_ID
    _B, _S, _H, _D = q.shape
    assert _S == 1, f"V4 decode-only path; got seq_q={_S}"
    native_started = time.monotonic()
    _dsv4_debug(
        "native_attention", "begin", debug_call_id,
        batch=_B, heads=_H, head_dim=_D, window=max_window_size,
        topk=indices.shape[-1], has_compressed=int(extra_k_cache is not None),
    )
    KV_LORA = head_dim_v  # 512

    # ---- SWA cache as flat [NB_swa*BS_swa, D] (static staging buffer; the
    # production view is strided, so this is a copy -- see _flatten_cache_static
    # for why the copy is unavoidable and why it must not allocate here) ----
    flat_swa = _flatten_cache_static(k_cache)
    swa_size = flat_swa.shape[0]
    kv_lod_swa_xpu = _build_static_per_seq_lens(
        topk_length[:_B], _static_kvlod_swa_xpu_cache)
    kv_lod_swa_cpu = _get_static_cpu_lod(
        kv_lod_swa_xpu, _static_kvlod_swa_cpu_cache,
        bound=indices.shape[-1])

    swa_idx_2d = indices.squeeze(1) if indices.dim() == 3 else indices
    if not swa_idx_2d.is_contiguous():
        swa_idx_2d = swa_idx_2d.contiguous()
    # Defense in depth: bound the upper end too. `_prepare_static_clamped_indices`
    # only clamps min=0, so a corrupted upstream index would reach fwd_kvcache_mla
    # as a wild row id and fault the device. The fused path already clamps both
    # ends -- mirror it here.
    swa_idx_2d = swa_idx_2d.clamp(max=swa_size - 1)
    clamped_swa = _prepare_static_clamped_indices(
        swa_idx_2d.unsqueeze(1), topk_length[:_B],
        _static_clamped_swa_idx_cache,
    )

    out_swa = torch.empty(_B, _S, _H, KV_LORA, dtype=q.dtype, device=q.device)
    ml_swa = torch.empty(_B, _S, _H, dtype=torch.float32, device=q.device)
    ps_swa = torch.empty(_B, _S, _H, dtype=torch.float32, device=q.device)

    _dsv4_debug(
        "swa_fwd_kvcache_mla", "begin", debug_call_id,
        cache_rows=swa_size, kv_lod_bound=indices.shape[-1],
    )
    swa_started = time.monotonic()
    kunlun_ops.fwd_kvcache_mla(
        q_c=q, kv_cache=flat_swa, indices=clamped_swa,
        kv_lod_cpu=kv_lod_swa_cpu,
        out=out_swa, max_logits=ml_swa, p_sums=ps_swa,
        softmax_scale=float(softmax_scale),
        max_seq_kv=int(max_window_size),
        kv_lod_xpu=kv_lod_swa_xpu,
        **_zero_rope_side_channel(q, swa_size, KV_LORA),
    )
    _dsv4_debug(
        "swa_fwd_kvcache_mla", "end", debug_call_id,
        elapsed_ms=round((time.monotonic() - swa_started) * 1000, 3),
    )
    # LSE in natural log (units of scaled logits, same as `sink`):
    #   lse = log(ps) + ml   (where ps = sum exp(scores_scaled - ml))
    # NOTE: attention_merge_stage uses natural log internally (verified
    # empirically: with same v and same lse inputs, s_merged - lse = ln(2),
    # which is `log(d) + s_max` only when d=2 and `log` is natural).
    lse_swa_e = torch.log(ps_swa.clamp_min(1e-30)) + ml_swa

    if extra_k_cache is None:
        out_merged = out_swa
        lse_merged_e = lse_swa_e
    else:
        flat_com = _flatten_cache_static(extra_k_cache)
        com_size = flat_com.shape[0]
        kv_lod_com_xpu = _build_static_per_seq_lens(
            extra_topk_length[:_B], _static_kvlod_com_xpu_cache)
        kv_lod_com_cpu = _get_static_cpu_lod(
            kv_lod_com_xpu, _static_kvlod_com_cpu_cache,
            bound=extra_indices_in_kvcache.shape[-1])
        com_idx_2d = (extra_indices_in_kvcache.squeeze(1)
                      if extra_indices_in_kvcache.dim() == 3
                      else extra_indices_in_kvcache)
        if not com_idx_2d.is_contiguous():
            com_idx_2d = com_idx_2d.contiguous()
        com_idx_2d = com_idx_2d.clamp(max=com_size - 1)
        clamped_com = _prepare_static_clamped_indices(
            com_idx_2d.unsqueeze(1), extra_topk_length[:_B],
            _static_clamped_com_idx_cache,
        )

        out_com = torch.empty(_B, _S, _H, KV_LORA, dtype=q.dtype, device=q.device)
        ml_com = torch.empty(_B, _S, _H, dtype=torch.float32, device=q.device)
        ps_com = torch.empty(_B, _S, _H, dtype=torch.float32, device=q.device)

        _dsv4_debug(
            "compressed_fwd_kvcache_mla", "begin", debug_call_id,
            cache_rows=com_size,
            kv_lod_bound=extra_indices_in_kvcache.shape[-1],
        )
        compressed_started = time.monotonic()
        kunlun_ops.fwd_kvcache_mla(
            q_c=q, kv_cache=flat_com, indices=clamped_com,
            kv_lod_cpu=kv_lod_com_cpu,
            out=out_com, max_logits=ml_com, p_sums=ps_com,
            softmax_scale=float(softmax_scale),
            max_seq_kv=int(com_topk),
            kv_lod_xpu=kv_lod_com_xpu,
            **_zero_rope_side_channel(q, com_size, KV_LORA),
        )
        _dsv4_debug(
            "compressed_fwd_kvcache_mla", "end", debug_call_id,
            elapsed_ms=round((time.monotonic() - compressed_started) * 1000, 3),
        )
        lse_com_e = torch.log(ps_com.clamp_min(1e-30)) + ml_com

        # ---- The case this two-call+merge decomposition does not express ----
        # The op this path replaced (`kunlun_ops.hybrid_attention`) took BOTH
        # `topk_length` and `extra_topk_length` into ONE kernel, so a row with no
        # compressed slots was handled inside the kernel and degenerated to
        # SWA-only by construction.  Splitting into two `fwd_kvcache_mla` calls
        # plus `attention_merge_stage` loses that: the merge has no notion of "this
        # branch is empty for this row", and `extra_topk_length[i] == 0` happens for
        # every row whose context is shorter than that layer's compress_ratio
        # (config compress_ratios alternate 4 / 128).
        #
        # So state it explicitly, from the METADATA rather than from the kernel's
        # output: an empty branch contributes a zero payload at a log-weight of
        # -inf, which the merge turns into exactly the SWA-only answer
        # (docs/stepS_fix.py: torch.equal(v_merged, v_a) is True, s_merged == s_a).
        # `-1e30` is the finite stand-in for -inf: exp() of it underflows to 0
        # while keeping every value the merge sees finite.
        #
        # Not done by scrubbing NaN: the kernel's kv_lod==0 output happens to be
        # out=NaN / max_logits=-1e12 / p_sums=0 (docs/stepT_kernel.py), but that is
        # undocumented behaviour, and masking NaN would also silently swallow a
        # NaN arriving for any other reason.  The condition that is actually
        # meaningful is `extra_topk_length == 0`.
        _com_empty = (extra_topk_length[:_B] == 0).view(_B, 1, 1)   # [B,1,1]
        out_com = torch.where(_com_empty.unsqueeze(-1),
                              torch.zeros((), dtype=out_com.dtype,
                                          device=out_com.device),
                              out_com)
        lse_com_e = torch.where(_com_empty, torch.full((), -1e30,
                                                       dtype=lse_com_e.dtype,
                                                       device=lse_com_e.device),
                                lse_com_e)

        # attention_merge_stage expects 3-D (tokens, heads, dim) and operates on
        # natural-log LSE (s_a, s_b in natural log units; s_merged also natural).
        # Empirical: with v_a=v_b and s_a=s_b=s, v_merged = v (max|diff|=0) and
        # s_merged = s + ln(2) (consistent with `s_merged = log(d) + s_max`
        # where `log` is natural, despite the kernel docstring saying `log2`).
        out_merged = torch.empty(_B, _H, KV_LORA, dtype=q.dtype, device=q.device)
        lse_merged_e = torch.empty(_B, _H, dtype=torch.float32, device=q.device)
        _dsv4_debug("attention_merge_stage", "begin", debug_call_id)
        merge_started = time.monotonic()
        kunlun_ops.attention_merge_stage(
            v_a=out_swa.squeeze(1), s_a=lse_swa_e.squeeze(1),
            v_b=out_com.squeeze(1), s_b=lse_com_e.squeeze(1),
            v_merged=out_merged, s_merged=lse_merged_e,
        )
        _dsv4_debug(
            "attention_merge_stage", "end", debug_call_id,
            elapsed_ms=round((time.monotonic() - merge_started) * 1000, 3),
        )
        # attention_merge_stage returns 2-D outputs [B, H]; restore the seq
        # axis so downstream sink broadcast matches out_swa/out_com layout.
        out_merged = out_merged.unsqueeze(1)            # [B, 1, H, D]
        lse_merged_e = lse_merged_e.unsqueeze(1)        # [B, 1, H]
        # s_merged already in natural log units — no further conversion needed.

    # ---- Attention sink post-processing: out *= sigmoid(lse - sink) ----
    # lse_merged_e is [B, 1, H] (matches out_merged's [B, 1, H, D] layout);
    # sink is [H]; broadcast to [B, 1, H, 1] for per-head, per-token scale.
    if isinstance(attn_sink, torch.Tensor):
        sink = attn_sink[:_H].to(device=q.device, dtype=torch.float32)
        scale = torch.sigmoid(lse_merged_e - sink.view(1, 1, _H))
        out_merged = out_merged * scale.unsqueeze(-1).to(out_merged.dtype)

    output.copy_(out_merged)
    _dsv4_debug(
        "native_attention", "end", debug_call_id,
        elapsed_ms=round((time.monotonic() - native_started) * 1000, 3),
    )
    return output, None


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

    # ---- Allocate output buffer (used by both native and torch paths) ----
    if out is not None:
        output = out
    else:
        output = torch.zeros(
            q.shape[0], q.shape[1], q.shape[2], head_dim_v,
            dtype=q.dtype, device=q.device,
        )

    # ---- Fused V4 decode: kunlun_ops.hybrid_attention (single kernel) ----
    _fused_ok = (
        os.environ.get("KUNLUN_DSV4_ATTN_DECODE_FUSED", "0") == "1"
        and hasattr(kunlun_ops, "hybrid_attention")
        and indices is not None
        and topk_length is not None
        and (k_cache.dtype == torch.bfloat16)
        and (extra_k_cache is None or extra_k_cache.dtype == torch.bfloat16)
    )
    if _fused_ok:
        try:
            return _v4_hybrid_attention_fused(
                q=q, k_cache=k_cache, indices=indices,
                topk_length=topk_length, attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices_in_kvcache,
                extra_topk_length=extra_topk_length,
                head_dim_v=head_dim_v, softmax_scale=softmax_scale,
                output=output,
                compress_ratio=compress_ratio,
                max_window_size=max_window_size, com_topk=com_topk,
            )
        except Exception as _e:
            if torch.cuda.is_current_stream_capturing():
                raise
            if not getattr(flash_mla_with_kvcache, "_fused_warned", False):
                logger.warning(
                    "[DSv4-FUSED-ATTN] hybrid_attention failed, "
                    "falling back to decomposed: %s: %s",
                    type(_e).__name__, _e,
                )
                flash_mla_with_kvcache._fused_warned = True


    # ---- Native V4 path (capture-safe; fwd_kvcache_mla + attention_merge_stage).
    # Replaces the previous kunlun_ops.hybrid_attention short-circuit which
    # allocated L3 scratch per call (capture-unsafe) and the per-batch torch
    # loop fallback (also capture-unsafe). Both old paths are removed.
    # BF16-only first iteration: FP8 KV cache layers still go through the
    # torch fallback further below (which handles uint8 dequant). ----
    _native_ok = (
        os.environ.get("KUNLUN_DSV4_HYBRID_NATIVE", "1") == "1"
        and hasattr(kunlun_ops, "fwd_kvcache_mla")
        and hasattr(kunlun_ops, "attention_merge_stage")
        and indices is not None
        and topk_length is not None
        and (k_cache.dtype == torch.bfloat16)
        and (extra_k_cache is None or extra_k_cache.dtype == torch.bfloat16)
    )
    if _native_ok:
        try:
            return _v4_sparse_native_path(
                q=q, k_cache=k_cache, indices=indices,
                topk_length=topk_length, attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices_in_kvcache,
                extra_topk_length=extra_topk_length,
                head_dim_v=head_dim_v, softmax_scale=softmax_scale,
                output=output,
                compress_ratio=compress_ratio,
                max_window_size=max_window_size, com_topk=com_topk,
            )
        except Exception as _e:
            # C5 -- no silent fallback during capture. The code below this
            # `except` is the torch loop path, which is capture-unsafe: if it
            # were recorded we would get a graph that is wrong AND silent.
            # The diagnostic `int(tensor.min())` reads are themselves illegal
            # during capture (on Kunlun they return garbage instead of raising:
            # a real capture-time failure printed
            # `idx min/max=1016332288/1016332288`), so they must not run either.
            if torch.cuda.is_current_stream_capturing():
                raise
            if not getattr(flash_mla_with_kvcache, "_native_warned", False):
                def _d(t):
                    if not isinstance(t, torch.Tensor):
                        return repr(t)
                    return "%s%s" % (tuple(t.shape), t.dtype)
                logger.warning(
                    "[DSv4-NATIVE-ATTN] fwd_kvcache_mla path failed, "
                    "falling back to torch: %s: %s\n"
                    "  q=%s k_cache=%s indices=%s topk_length=%s "
                    "extra_k=%s extra_idx=%s extra_len=%s\n"
                    "  head_dim_v=%s scale=%s cr=%s mws=%s com_topk=%s\n"
                    "  swa idx min/max=%s/%s len min/max=%s/%s cache_rows=%s\n"
                    "  com idx min/max=%s/%s len min/max=%s/%s cache_rows=%s",
                    type(_e).__name__, _e,
                    _d(q), _d(k_cache), _d(indices), _d(topk_length),
                    _d(extra_k_cache), _d(extra_indices_in_kvcache),
                    _d(extra_topk_length),
                    head_dim_v, softmax_scale, compress_ratio,
                    max_window_size, com_topk,
                    int(indices.min()), int(indices.max()),
                    int(topk_length.min()), int(topk_length.max()),
                    k_cache.numel() // k_cache.shape[-1],
                    int(extra_indices_in_kvcache.min())
                    if extra_indices_in_kvcache is not None else None,
                    int(extra_indices_in_kvcache.max())
                    if extra_indices_in_kvcache is not None else None,
                    int(extra_topk_length.min())
                    if extra_topk_length is not None else None,
                    int(extra_topk_length.max())
                    if extra_topk_length is not None else None,
                    (extra_k_cache.numel() // extra_k_cache.shape[-1])
                    if extra_k_cache is not None else None,
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

    out, max_logits, p_sums = _get_kunlun_mla_output_buffers(
        batch_size, seq_len_q, num_heads_q, kv_lora_rank, q.dtype, q.device,
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
    # rejects this V4 layout, so fall back to a naive PyTorch MLA attention
    # over the gathered bf16 workspace. The 2-D fast path (V3.2 indexer
    # sparse prefill) is unchanged.
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

