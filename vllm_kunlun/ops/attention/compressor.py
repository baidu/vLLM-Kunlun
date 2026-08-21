"""DeepSeek-V4 KV-compressor on Kunlun XPU.

Provides Kunlun implementations of the three compressor entry points --
``get_compressed_slot_mapping``, ``save_partial_states`` and
``compress_norm_rope_store`` -- replacing the upstream Triton kernels.
The hot path probes and uses the XSpeedGate kernels
(``fused_kv_compress_gather``, ``dpsk_v4_norm_rope_gptj``) when available,
with PyTorch fallbacks otherwise.
"""
import logging
from typing import List

import os

import torch

from vllm_kunlun.adapter_utils import WarningOnce, record_wired
from vllm_kunlun.patches.registry import _register_lazy

LOGGER = logging.getLogger("vllm_kunlun.ops.attention.compressor")
_APPLIED_SENTINEL = "_dsv4_compressor_applied"
_FALSE = object()
_slot_map_logged: set = set()

# TODO(perf): get_compressed_slot_mapping below is ~10 elementwise kernels plus a
# [num_reqs, num_tokens] bool reduction, where upstream runs one triton kernel over
# num_reqs programs. XSpeedGate has no equivalent yet -- init_compressed_attn_metadata
# does not fit (per-request output, no page-table lookup, 0 instead of -1 as the
# skip sentinel, decode-only shape), so this wants a new native op.


def _applied(mod: object) -> bool:
    return bool(getattr(mod, _APPLIED_SENTINEL, False))


def _is_fn_patched(fn_obj: object) -> bool:
    """Check legacy marker *and* our unified marker on installed functions."""
    if fn_obj is None:
        return False
    return getattr(fn_obj, "_dsv4_wired", False) or getattr(
        fn_obj, "_kunlun_patched", False
    )


def _mark_wired(obj: object) -> None:
    setattr(obj, "_dsv4_wired", True)


# Paged full-row store: xspeedgate_ops.set_k_and_s_v4 replaces the torch scatter
# when the cache layout is expressible for it (see _paged_store_usable).
_PAGED_STORE_OFF = os.environ.get("KUNLUN_DSV4_PAGED_STORE_NATIVE", "1") != "1"
_PAGED_STORE_DEBUG = os.environ.get("KUNLUN_DSV4_PAGED_STORE_DEBUG", "0") == "1"
_paged_store_op: object = _FALSE
_paged_store_seen: set = set()
_paged_store_warm: set = set()


def _log_paged_store_route(cache, data, col_start, fast) -> None:
    key = (tuple(cache.shape), str(cache.dtype), int(col_start), int(data.shape[1]))
    if key in _paged_store_seen:
        return
    _paged_store_seen.add(key)
    LOGGER.info(
        "[paged-store] cache=%s %s strides=%s col_start=%d D=%d -> %s",
        tuple(cache.shape), cache.dtype, tuple(cache.stride()), col_start,
        data.shape[1], "native" if fast else "torch",
    )


def _find_paged_store_op():
    global _paged_store_op
    if _paged_store_op is not _FALSE:
        return _paged_store_op
    handle = None
    source = "skip" if _PAGED_STORE_OFF else "torch_fallback"
    if not _PAGED_STORE_OFF:
        try:
            __import__("xspeedgate_ops")
        except Exception:  # noqa: BLE001
            pass
        ns = getattr(torch.ops, "xspeedgate_ops", None)
        candidate = getattr(ns, "set_k_and_s_v4", None) if ns is not None else None
        if candidate is not None:
            handle = candidate
            source = "torch.ops.xspeedgate_ops"
    _paged_store_op = handle
    record_wired("set_k_and_s_v4", source)
    return handle


def _paged_store_usable(cache, data, col_start) -> bool:
    """Whether ``set_k_and_s_v4`` can express this write.

    3D mode addresses ``[num_pages, page_size, k_dim]`` with the page stride
    taken from ``stride(0)``, so pages may be spaced apart (the compressor
    caches are strided views into the shared KV pool) but rows must be dense
    and the write must cover a full row of a BF16/FP16 cache.
    """
    return (
        col_start == 0
        and cache.dim() == 3
        and data.dim() == 2
        and cache.dtype in (torch.bfloat16, torch.float16)
        and data.shape[1] == cache.shape[2]
        and cache.stride(2) == 1
        and cache.stride(1) == cache.shape[2]
        and cache.stride(0) >= cache.shape[1] * cache.shape[2]
    )


def _warm_paged_store_once(cache) -> None:
    """Launch ``set_k_and_s_v4`` once per cache, outside any cudagraph capture.

    A kernel module is loaded on its first launch and that load is illegal
    inside stream capture on this XPU (same failure mode documented for
    ``fused_kv_compress_gather`` in ``_warm_native_once``). The warm write puts
    zeros into slot 0, the null block vLLM's BlockPool keeps zeroed.
    """
    op = _find_paged_store_op()
    if op is not None and cache.dim() == 3:
        key = (cache.data_ptr(), int(cache.shape[1]), int(cache.shape[2]))
        if key not in _paged_store_warm and _paged_store_usable(
            cache, cache[0, :1], 0
        ):
            _paged_store_warm.add(key)
            device = cache.device
            op(
                cache,
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.zeros((1, cache.shape[2]), dtype=cache.dtype, device=device),
                cache.shape[1],
            )
    _probe_indexer_store_once(cache)


# Int8 fused quant store: kunlun_ops.indexer_k_quant_and_cache quantizes bf16 k
# rows (scale = amax/127) and writes the per-page planar layout
# [block_size*128 values | block_size*fp32 scales] that I8_paged_mqa_logits
# reads. The kernel derives each page base as block * block_size * stride(1),
# so a padded cache (stride(0) != block_size * stride(1)) must be re-viewed
# with row stride stride(0) // block_size.
_INDEXER_STORE_OFF = os.environ.get("KUNLUN_DSV4_INDEXER_STORE_NATIVE", "1") != "1"
_indexer_store_op: object = _FALSE
_indexer_store_ok: set = set()
_indexer_store_bad: set = set()


def _find_indexer_store_op():
    global _indexer_store_op
    if _indexer_store_op is not _FALSE:
        return _indexer_store_op
    handle = None
    source = "skip" if _INDEXER_STORE_OFF else "torch_fallback"
    if not _INDEXER_STORE_OFF:
        try:
            mod = __import__("kunlun_ops", fromlist=["indexer_k_quant_and_cache"])
            candidate = getattr(mod, "indexer_k_quant_and_cache", None)
            if callable(candidate):
                handle = candidate
                source = "kunlun_ops_module"
        except Exception:  # noqa: BLE001
            pass
    _indexer_store_op = handle
    record_wired("indexer_k_quant_and_cache", source)
    return handle


def indexer_cache_planar(cache) -> bool:
    """Whether ``cache`` is maintained in the planar layout the native store
    produces (prefill-side gather must match the write layout)."""
    return cache.data_ptr() in _indexer_store_ok


def _indexer_store_view(cache):
    """Re-view a padded indexer cache as dense rows for the native store.

    ``indexer_k_quant_and_cache`` derives each page base from ``stride(1)``
    and cannot see the 512B-alignment padding between pages (stride(0) >
    page_size * stride(1)), so the cache is re-exposed as
    [num_pages, page_size, row_stride] with row_stride = stride(0)//page_size
    covering value bytes plus padding. Returns None when the geometry does
    not allow such a view.
    """
    page = cache.stride(0)
    bs = cache.shape[1]
    if page % bs:
        return None
    row = page // bs
    if row <= cache.shape[2]:
        return None
    return cache.as_strided(cache.shape[:2] + (row,), (page, row, 1),
                            cache.storage_offset())


def _indexer_store_usable(cache, data, col_start) -> bool:
    """Whether the fused int8 quant+planar store can take this write.

    Requires the warm-time probe (``_probe_indexer_store_once``) to have
    verified the planar layout on this exact cache, plus full-width rows:
    cache.shape[2] == data.shape[1] + 4 is the [values | fp32 scale] page
    geometry the decoder reads back.
    """
    return (
        col_start == 0
        and cache.dtype == torch.int8
        and cache.dim() == 3
        and data.dim() == 2
        and cache.shape[2] == data.shape[1] + 4
        and cache.stride(2) == 1
        and cache.data_ptr() in _indexer_store_ok
    )


def _probe_indexer_store_once(cache) -> None:
    """Verify the native store's addressing on this cache, once, outside capture.

    Writes rows with absmax exactly 127 (scale = 127/127 = 1.0f, so int8 bytes
    equal the source values) into blocks 1 and 2, then reads them back through
    the planar offsets the decode reader uses. Any mismatch keeps the torch
    fallback (and the interleaved prefill read) active for this cache.
    """
    op = _find_indexer_store_op()
    if op is None or cache.dim() != 3 or cache.dtype != torch.int8:
        return
    key = cache.data_ptr()
    if key in _indexer_store_ok or key in _indexer_store_bad:
        return
    if cache.shape[0] < 3 or cache.shape[2] != 132 or cache.shape[1] < 4:
        _indexer_store_bad.add(key)
        return
    try:
        view = _indexer_store_view(cache)
        if view is None:
            raise ValueError("page stride not divisible")
        dev = cache.device
        bs = cache.shape[1]
        k = torch.zeros(2, 128, dtype=torch.bfloat16, device=dev)
        k[:, 0], k[:, 1], k[:, 2] = 127.0, -127.0, 64.0
        slots = torch.tensor([bs, 2 * bs + 3], dtype=torch.int64, device=dev)
        op(k, view, slots, 128, True, False, False)
        # 符号重写栈：XPU sync 走 torch.cuda 入口
        torch.cuda.synchronize()
        page1 = cache[1].reshape(-1)
        page2 = cache[2].reshape(-1)
        ok = (
            int(page1[0]) == 127 and int(page1[1]) == -127 and int(page1[2]) == 64
            and page1[bs * 128:bs * 128 + 4].view(torch.float32).item() == 1.0
            and int(page2[3 * 128]) == 127 and int(page2[3 * 128 + 1]) == -127
            and page2[bs * 128 + 12:bs * 128 + 16].view(torch.float32).item() == 1.0
        )
        if not ok:
            raise ValueError("planar readback mismatch")
        _indexer_store_ok.add(key)
        LOGGER.info(
            "[indexer-store] native planar store verified on %s %s strides=%s",
            tuple(cache.shape), cache.dtype, tuple(cache.stride()),
        )
    except Exception as exc:  # noqa: BLE001
        _indexer_store_bad.add(key)
        LOGGER.info(
            "[indexer-store] native store rejected (%s); torch fallback active", exc,
        )


def _masked_paged_write(cache, dest, write_ok, data, col_start=0):
    """Scatter ``data`` into paged ``cache`` at flat slots ``dest``, skipping rows.

    Keeps every output shape static during cudagraph capture by redirecting
    skipped rows onto block 0 (the null block kept zeroed by vLLM BlockPool).
    Mask logic runs in fp32 because some xdnn capture ops reject uint8 masks.
    """
    if cache.dtype.itemsize == 1 and col_start not in _slot_map_logged:
        _slot_map_logged.add(col_start)
        probe = ""
        try:
            if cache.shape[-1] >= 132:
                s0 = cache[0, 0, 128:132].view(torch.float32).item()
                v0 = int(cache[0, 0, 0].item())
                probe = f" slot0: scale={s0:.6g} val0={v0}"
        except Exception:  # noqa: BLE001
            pass
        LOGGER.info(
            "[paged-write] cache=%s %s strides=%s col_start=%d data=%s %s%s",
            tuple(cache.shape), cache.dtype, tuple(cache.stride()), col_start,
            tuple(data.shape), data.dtype, probe,
        )
    op = _find_paged_store_op()
    fast = op is not None and _paged_store_usable(cache, data, col_start)
    iop = _find_indexer_store_op()
    ifast = iop is not None and _indexer_store_usable(cache, data, col_start)
    if _PAGED_STORE_DEBUG:
        _log_paged_store_route(cache, data, col_start, fast or ifast)
    if fast:
        dest_safe = torch.where(write_ok, dest, 0)
        data_ok = data.to(cache.dtype)
        if data_ok.data_ptr() == data.data_ptr():
            data_ok = data_ok.clone()
        data_ok.mul_(write_ok.unsqueeze(-1))
        op(cache, dest_safe, data_ok, cache.shape[1])
        return
    if ifast:
        # Fused quant + planar store; the kernel skips negative slots itself.
        dest_arg = torch.where(write_ok, dest.long(), torch.full_like(dest, -1))
        k_bf16 = data.to(torch.bfloat16)
        if not k_bf16.is_contiguous():
            k_bf16 = k_bf16.contiguous()
        iop(k_bf16, _indexer_store_view(cache), dest_arg, 128, True, False, False)
        return

    D = data.shape[1]
    block_size = cache.shape[1]
    data = data.float()
    dest_safe = torch.where(write_ok, dest.long(), torch.zeros_like(dest))
    data_ok = torch.where(write_ok.unsqueeze(-1), data, torch.zeros_like(data)).to(
        cache.dtype
    )
    cache[:, :, col_start : col_start + D].index_put_(
        (dest_safe // block_size, dest_safe % block_size),
        data_ok,
        accumulate=False,
    )
    

# Compressed-slot-mapping: vectorized page-table lookup.
_SLOT_FN_NAME = "get_compressed_slot_mapping"
_SLOT_TARGETS = (
    "vllm.v1.attention.backends.mla.compressor_utils",
    "vllm.v1.attention.backends.mla.indexer",
)


def _install_slot_mapping(utils_mod: object) -> None:
    if _is_fn_patched(getattr(utils_mod, _SLOT_FN_NAME, None)):
        return

    def get_compressed_slot_mapping(
        num_tokens,
        query_start_loc,
        seq_lens,
        block_table,
        block_size,
        compress_ratio,
        out=None,
    ):
        """Per-token slot in the compressor state cache, ``-1`` where unused.

        query_start_loc [num_reqs+1] int32/int64, seq_lens [num_reqs] int,
        block_table [num_reqs, max_blocks] int, both on device. A token at
        absolute position ``pos`` only occupies a slot when it closes a
        compression window (``(pos + 1) % compress_ratio == 0``); its slot is
        ``block_table[req, cpos // block_size] * block_size + cpos % block_size``
        with ``cpos = pos // compress_ratio``. Every other token, and every
        padding token past the last request, keeps the ``-1`` sentinel that
        downstream save/compress paths read as "skip".

        Returns ``out`` when given (filled in place), else a fresh
        [num_tokens] int64 tensor.

        When num_reqs == num_tokens (pure decode: exactly one query token
        per request) a reduced per-request path runs instead: it skips the
        token-to-request expansion and the [num_reqs, num_tokens] broadcast
        reduction, which at small batch dominate the kernel count.
        """
        device = query_start_loc.device
        if out is None:
            result = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)
        else:
            out.fill_(-1)
            result = out[:num_tokens]

        num_reqs = min(block_table.shape[0], query_start_loc.shape[0] - 1)
        if num_reqs <= 0 or num_tokens <= 0:
            return out if out is not None else result

        if compress_ratio not in _slot_map_logged:
            _slot_map_logged.add(compress_ratio)
            LOGGER.info(
                "[slot-map] ratio=%d num_reqs=%d num_tokens=%d block_size=%d",
                compress_ratio, num_reqs, num_tokens, block_size,
            )

        if num_reqs == num_tokens:
            # 纯 decode（每请求恰 1 个 query token）：token<->请求一一对应，
            # 跳过通用路径的 token 展开与 broadcast 归约。
            pos = seq_lens[:num_reqs].long() - 1
            cpos = pos // compress_ratio
            block_ids = (cpos // block_size).clamp_(
                min=0, max=block_table.shape[1] - 1
            )
            slots = (
                block_table.gather(1, block_ids.unsqueeze(1)).squeeze(1).long()
                * block_size + cpos % block_size
            )
            slots.masked_fill_((pos + 1) % compress_ratio != 0, -1)
            result.copy_(slots)
            return out if out is not None else result

        bounds = query_start_loc[: num_reqs + 1].long()
        req_start = bounds[:-1]
        query_lens = bounds[1:] - req_start

        token = torch.arange(num_tokens, device=device, dtype=torch.int64)
        # Request index per token, without the host round-trip a bucketize or a
        # repeat_interleave would need: count the request ends already passed.
        req = (token.unsqueeze(0) >= bounds[1:].unsqueeze(1)).sum(0)
        req.clamp_(max=num_reqs - 1)

        pos = seq_lens.long()[req] - query_lens[req] + (token - req_start[req])
        compressed_pos = pos // compress_ratio
        # Clamp before the gather because torch cannot mask it the way the
        # upstream triton kernel masks its block_table load; the out-of-range
        # rows are dropped by `keep` below anyway.
        block_ids = (compressed_pos // block_size).clamp_(
            min=0, max=block_table.shape[1] - 1
        )
        slots = (
            block_table[req, block_ids].long() * block_size
            + compressed_pos % block_size
        )

        keep = (
            (token < bounds[num_reqs])
            & (pos >= 0)
            & ((pos + 1) % compress_ratio == 0)
        )
        result.copy_(torch.where(keep, slots, torch.full_like(slots, -1)))
        return out if out is not None else result

    _mark_wired(get_compressed_slot_mapping)
    setattr(utils_mod, _SLOT_FN_NAME, get_compressed_slot_mapping)
    LOGGER.info("Installed compressed-slot-mapping into %s", utils_mod.__name__)


# save_partial_states dispatchers.
_SPS_TARGET_MODULE = "vllm.models.deepseek_v4.common.ops.save_partial_states"
_SPS_OP_CANDIDATE_NAMES = [
    ("kunlun_ops", "save_partial_states"),
]
_sps_op_handle: object = _FALSE
_sps_warning_key = "dsv4-save-partial-states-native-failed-once"


def _find_save_partial_states_op():
    global _sps_op_handle
    if _sps_op_handle is not _FALSE:
        return _sps_op_handle
    handle = None
    src = "torch_fallback"
    for mod_name, attr_name in _SPS_OP_CANDIDATE_NAMES:
        try:
            mod = __import__(mod_name, fromlist=[attr_name])
            candidate = getattr(mod, attr_name, None)
            if callable(candidate):
                handle = candidate
                src = f"{mod_name}_module"
                break
        except Exception:  # noqa: BLE001
            continue
    _sps_op_handle = handle
    record_wired("save_partial_states", src)
    return handle


def _torch_save(kv, score, ape, positions, state_cache, slot_mapping,
                block_size, state_width, compress_ratio):
    valid_mask = slot_mapping >= 0
    capturing = torch.cuda.is_current_stream_capturing() if hasattr(torch.cuda, "is_current_stream_capturing") else False
    if capturing:
        sel = torch.arange(slot_mapping.shape[0], device=kv.device)
        wv = valid_mask
    else:
        if not bool(valid_mask.any()):
            return
        sel = valid_mask.nonzero(as_tuple=True)[0]
        wv = torch.ones_like(sel, dtype=torch.bool)

    slots = slot_mapping.index_select(0, sel).long()
    write_ok = wv & (slots >= 0)
    _masked_paged_write(state_cache, dest=slots, write_ok=write_ok, data=kv.index_select(0, sel))
    ape_rows = positions.index_select(0, sel) % compress_ratio
    _masked_paged_write(
        state_cache,
        dest=slots,
        write_ok=write_ok,
        data=(score.index_select(0, sel) + ape.index_select(0, ape_rows)),
        col_start=state_width,
    )


def _install_save_partial_states(sps_module: object) -> None:
    fn_name = "save_partial_states"
    if _is_fn_patched(getattr(sps_module, fn_name, None)):
        return

    sps_native = _find_save_partial_states_op()

    def save_partial_states(
        kv, score, ape, positions, state_cache, slot_mapping,
        block_size, state_width, compress_ratio, pdl_kwargs=None,
    ):
        del pdl_kwargs
        global _c4_stashed_kv_score, _c4_stashed_ape
        if _C4_FUSED_ENABLED and compress_ratio == 4:
            _c4_stashed_kv_score = torch.cat([kv, score], dim=-1)
            _c4_stashed_ape = ape
            return
        if sps_native is not None:
            try:
                sps_native(
                    kv, score, ape, positions, slot_mapping, state_cache,
                    block_size, state_width, compress_ratio,
                )
                return
            except Exception as exc:  # noqa: BLE001
                WarningOnce.emit(
                    _sps_warning_key,
                    "Native kunlun_ops.save_partial_states failed (%s); falling back to PyTorch reference",
                    str(exc),
                )
        _torch_save(kv, score, ape, positions, state_cache, slot_mapping,
                    block_size, state_width, compress_ratio)

    _mark_wired(save_partial_states)
    setattr(sps_module, fn_name, save_partial_states)
    LOGGER.info(
        "Patched save_partial_states into %s (native=%s)",
        sps_module.__name__,
        sps_native is not None,
    )


# Vectorized compress/norm/rope/store dispatcher.
_VECT_TARGET_MODULE = "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache"
_COMPRESSED_FN_ATTR = "_dsv4_compress_core_installed"
_warned_missing_compress_op_kind: set[str] = set()


def _warn_once_compress_missing(kind: str):
    if kind in _warned_missing_compress_op_kind:
        return
    _warned_missing_compress_op_kind.add(kind)
    WarningOnce.emit(
        f"compress-missing:{kind}",
        "%s unavailable for V4 compressor fast-path; keeping cudagraph-safe torch fallback active",
        kind,
    )


def _find_xsg_compress_ops(native_allowed: bool):
    """Lookup fused_kv_compress_gather + dpsk_v4_norm_rope_gptj.

    Three lookup layers (§8-#0 fix, 2026-08-08):
      1. xspeedgate_ops Python module (only a shim in practice)
      2. torch.ops.xspeedgate_ops (registered via TORCH_LIBRARY)
      3. kunlun_ops Python module (these two ops are NOT registered to
         torch.ops but ARE in the kunlun_ops ctypes Python module —
         vllm_kunlun/ops/attention/compressor.py verification).

    Records what was found (or fell back to) into the vllm_kunlun
    wired inventory for the startup log.
    """
    desired = ["fused_kv_compress_gather", "dpsk_v4_norm_rope_gptj"]
    if not native_allowed:
        for n in desired:
            record_wired(n, "skip")
        return False, [], []
    handles: dict[str, object] = {}
    sources: dict[str, str] = {}
    # Layer 1: xspeedgate_ops Python module
    try:
        xspeedgate_module = __import__("xspeedgate_ops", fromlist=[desired[-1]])
        for n in desired:
            h = getattr(xspeedgate_module, n, None)
            if callable(h):
                handles[n] = h
                sources[n] = "xspeedgate_ops_module"
    except Exception:  # noqa: BLE001
        pass
    # Layer 2: torch.ops.xspeedgate_ops
    try:
        alt_ns = getattr(torch.ops, "xspeedgate_ops", None)
        if alt_ns is not None:
            for n in desired:
                if n in handles:
                    continue
                h2 = getattr(alt_ns, n, None)
                if callable(h2):
                    handles[n] = h2
                    sources[n] = "torch.ops.xspeedgate_ops"
    except Exception:  # noqa: BLE001
        pass
    # Layer 3 (§8-#0 fix): kunlun_ops Python module
    try:
        kunlun_module = __import__("kunlun_ops", fromlist=desired)
        for n in desired:
            if n in handles:
                continue
            h3 = getattr(kunlun_module, n, None)
            if callable(h3):
                handles[n] = h3
                sources[n] = "kunlun_ops_module"
    except Exception:  # noqa: BLE001
        pass
    # Record inventory (one entry per desired op)
    for n in desired:
        if n in handles:
            record_wired(n, sources[n])
        else:
            record_wired(n, "torch_fallback")
    if len(handles) == len(desired):
        return True, [handles["fused_kv_compress_gather"], handles["dpsk_v4_norm_rope_gptj"]], sources
    for k in set(desired) - set(handles.keys()):
        _warn_once_compress_missing(f"{k}")
    return False, [], sources


_NATIVE_OFF = os.environ.get("KUNLUN_DSV4_COMPRESS_NATIVE", "1") != "1"
# With both native kernel modules loaded before capture (`_warm_native_once`),
# the native compressor is usable inside a capture and the old "always fall back
# to torch while capturing" gate is not needed. KUNLUN_DSV4_COMPRESS_WARMUP=0
# restores that gate, i.e. the pre-warmup behaviour, exactly.
_WARMUP_ON = os.environ.get("KUNLUN_DSV4_COMPRESS_WARMUP", "1") != "0"
_WARMED: set = set()

# flash_compress_4_decode fused path (write + compress in one kernel).
_C4_FUSED_ENABLED = os.environ.get("KUNLUN_DSV4_C4_FUSED", "0") == "1"
_c4_fused_op: object = _FALSE
_c4_fused_warned_key = "dsv4-c4-fused-failed-once"
_c4_stashed_kv_score: object = None
_c4_stashed_ape: object = None


def _find_c4_fused_op():
    global _c4_fused_op
    if _c4_fused_op is not _FALSE:
        return _c4_fused_op
    handle = None
    src = "unavailable"
    try:
        kunlun_module = __import__("kunlun_ops", fromlist=["flash_compress_4_decode"])
        candidate = getattr(kunlun_module, "flash_compress_4_decode", None)
        if callable(candidate):
            handle = candidate
            src = "kunlun_ops_module"
    except Exception:
        pass
    _c4_fused_op = handle
    record_wired("flash_compress_4_decode", src)
    return handle


def _fused_c4_path_full(
    state_cache, num_actual, positions, slot_mapping, block_size, block_table,
    token_to_req_indices, cos_sin_cache, kv_cache, kv_slot_mapping,
    head_dim, rope_head_dim, compress_ratio, rms_norm_weight, rms_norm_eps,
    kv_score_input_full, ape_param,
):
    """Fused C4 decode: flash_compress_4_decode (write+compress) + norm_rope + write KV cache."""
    c4_op = _find_c4_fused_op()
    if c4_op is None:
        return False

    all_slots = slot_mapping[:num_actual]
    all_positions = positions[:num_actual]
    device = state_cache.device

    # seq_lens (1-based) and page indices
    seq_lens = (all_positions + 1).to(torch.int32)
    indices = (all_slots // block_size).clamp(min=0).to(torch.int32)

    # Compute extra (overlap page) for Page4Align mode
    req_indices = token_to_req_indices[:num_actual].long()
    overlap_block_id = ((all_positions - compress_ratio).clamp(min=0) // block_size).long()
    overlap_block_id_safe = overlap_block_id.clamp(max=block_table.shape[1] - 1)
    extra = block_table[req_indices, overlap_block_id_safe].to(torch.int32).unsqueeze(1)

    # APE: [4, 2*head_dim] fp32 -> [8, head_dim] same dtype as buffer
    ape_for_kernel = ape_param.view(8, head_dim).to(state_cache.dtype)

    # kv_score_input: [num_actual, 4*head_dim], same dtype as buffer
    kv_input = kv_score_input_full[:num_actual].to(state_cache.dtype)

    # Output buffer for compressed result
    compressed_out = torch.zeros(
        (num_actual, head_dim), dtype=state_cache.dtype, device=device
    )

    try:
        ret = c4_op(
            state_cache, kv_input, compressed_out, ape_for_kernel,
            indices, seq_lens, extra,
        )
        if ret != 0:
            raise RuntimeError(f"flash_compress_4_decode returned {ret}")
    except Exception as exc:
        WarningOnce.emit(
            _c4_fused_warned_key,
            "flash_compress_4_decode failed (%s); falling back to legacy path",
            str(exc),
        )
        return False

    # Identify boundary tokens (seq_len % 4 == 0)
    boundary_mask = (seq_lens.long() % compress_ratio == 0)

    capturing_func = getattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    capturing = capturing_func()
    if not capturing:
        if not bool(boundary_mask.any()):
            return True  # Write done, no compress needed
        boundary_sel = boundary_mask.nonzero(as_tuple=True)[0]
    else:
        boundary_sel = torch.arange(num_actual, device=device)

    # Compressed values for boundary tokens
    compressed_boundary = compressed_out.index_select(0, boundary_sel).float()

    # Apply norm + rope (reuse existing native op)
    result = _find_xsg_compress_ops(True)
    normrope_op = result[1][1] if len(result[1]) > 1 else None
    half_rope = rope_head_dim // 2

    if normrope_op is not None:
        boundary_positions = all_positions.index_select(0, boundary_sel).long()
        compressed_positions = (boundary_positions // compress_ratio) * compress_ratio
        cs = cos_sin_cache.index_select(0, compressed_positions)
        interleaved_freqs = (
            torch.stack((cs[:, :half_rope], cs[:, half_rope:]), dim=-1)
            .reshape(boundary_sel.shape[0], rope_head_dim)
            .float()
            .contiguous()
        )
        local_pos = torch.arange(boundary_sel.shape[0], dtype=torch.int64, device=device)
        status = normrope_op(
            compressed_boundary,
            rms_norm_weight.float().contiguous(),
            local_pos,
            interleaved_freqs,
            mode=2, compress_ratio=0, eps=rms_norm_eps,
        )
        if status != 0:
            WarningOnce.emit(
                _c4_fused_warned_key + "-normrope",
                "dpsk_v4_norm_rope_gptj returned %d in fused c4 path", status,
            )
            return False
    else:
        # Torch fallback for norm + rope
        nope_head_dim = head_dim - rope_head_dim
        variance = (compressed_boundary * compressed_boundary).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(variance + rms_norm_eps)
        compressed_boundary = compressed_boundary * rrms * rms_norm_weight.float().unsqueeze(0)
        boundary_positions = all_positions.index_select(0, boundary_sel).long()
        compressed_pos = (boundary_positions // compress_ratio) * compress_ratio
        cs = cos_sin_cache[compressed_pos]
        cos_vals, sin_vals = cs[:, :half_rope], cs[:, half_rope:]
        rope_part = compressed_boundary[:, nope_head_dim:]
        rope_even, rope_odd = rope_part[:, 0::2], rope_part[:, 1::2]
        compressed_boundary[:, nope_head_dim::2] = rope_even * cos_vals - rope_odd * sin_vals
        compressed_boundary[:, nope_head_dim + 1::2] = rope_even * sin_vals + rope_odd * cos_vals

    # Write to KV cache
    kv_slots = kv_slot_mapping.index_select(0, boundary_sel).long()
    write_ok = kv_slots >= 0
    if capturing:
        write_ok = write_ok & boundary_mask.index_select(0, boundary_sel)
    _masked_paged_write(kv_cache, dest=kv_slots, write_ok=write_ok, data=compressed_boundary)
    return True


def _install_compress_norm_rope_store_triton(fcqc_module: object) -> None:
    fn_name = "compress_norm_rope_store_triton"
    mod_attr = f"{fn_name}{_COMPRESSED_FN_ATTR}"
    if getattr(fcqc_module, mod_attr, False):
        return

    result = _find_xsg_compress_ops(True)
    native_available = result[0]
    gather_op = result[1][0] if result[1] else None
    normrope_op = result[1][1] if len(result[1]) > 1 else None

    def _native_compress(sel, state_cache, token_to_req_indices, all_positions, all_slots,
                         block_table, block_size, state_width, cos_sin_cache, head_dim,
                         rope_head_dim, compress_ratio, overlap, rms_norm_weight,
                         rms_norm_eps):
        if not native_available or _NATIVE_OFF:
            return None
        # Capture gate, only meaningful when the warmup above is disabled: a
        # native kernel whose module has never been loaded returns ret=2 when
        # first launched inside a capture stream, which puts the device into an
        # error state that the try/except below cannot recover from (the next
        # pure-torch index_select and capture_end() both then raise
        # "unrecognized error code") and engine-core init dies.
        _capturing = getattr(torch.cuda, "is_current_stream_capturing", lambda: False)()
        if _capturing and not _WARMUP_ON:
            return None
        try:
            selected_positions = all_positions.index_select(0, sel)
            selected_slots = all_slots.index_select(0, sel)
            selected_reqs = token_to_req_indices.index_select(0, sel)
            num_selected = sel.shape[0]
            compressed = torch.zeros(
                (num_selected, head_dim), dtype=torch.float32, device=state_cache.device
            )
            gather_op(
                state_cache,
                selected_reqs,
                selected_positions,
                selected_slots,
                block_table,
                compressed,
                block_size,
                head_dim,
                state_width,
                compress_ratio,
                int(bool(overlap)),
            )

            compressed_positions = (
                selected_positions.long() // compress_ratio * compress_ratio
            )
            cos_sin = cos_sin_cache.index_select(0, compressed_positions)
            half_rope = rope_head_dim // 2
            interleaved_freqs = (
                torch.stack((cos_sin[:, :half_rope], cos_sin[:, half_rope:]), dim=-1)
                .reshape(num_selected, rope_head_dim)
                .float()
                .contiguous()
            )
            local_pos = torch.arange(
                num_selected, dtype=torch.int64, device=state_cache.device
            )
            status = normrope_op(
                compressed,
                rms_norm_weight.float().contiguous(),
                local_pos,
                interleaved_freqs,
                mode=2,
                compress_ratio=0,
                eps=rms_norm_eps,
            )
            if status != 0:
                raise RuntimeError(f"dpsk_v4_norm_rope_gptj returned {status}")
            return compressed
        except Exception as exc:  # noqa: BLE001
            WarningOnce.emit(
                "dsv4-compressor-native-failed-once",
                "Native V4 compressor failed (%s); falling back to PyTorch reference",
                str(exc),
            )
            return None

    def _warm_native_once(state_cache, token_to_req_indices, all_positions,
                          all_slots, block_table, block_size, cos_sin_cache,
                          head_dim, rope_head_dim, state_width, compress_ratio,
                          overlap, rms_norm_weight, rms_norm_eps):
        """Launch both native kernels once per geometry, outside any capture.

        A kernel's module is loaded/relocated on its FIRST launch, and that
        load is illegal inside stream capture on this XPU -- the launch then
        returns XPU error 2, which surfaces as
        ``optimized_ops::fused_kv_compress_gather failed ret=2``
        (probe20: cold capture FAILs, warm capture OKs, one eager call being
        the only difference).

        The launch here is a guaranteed no-op on the device: the kernel's own
        prescan drops every token with ``slot_id < 0`` (see
        infer_kv_compress_pipeline.h:310), so with slot_mapping = -1 it reads
        neither state_cache nor block_table. Kernel selection depends only on
        (compress_ratio, overlap, head_size, state_width), and the C++ template
        is instantiated on the index dtypes, so every tensor here is either the
        real one or a 1-element tensor cloned from the real one's dtype -- the
        specialized binary that gets loaded is exactly the one serve needs
        (C4_H512 / C4_H128 / C128_H512).
        """
        if not native_available or _NATIVE_OFF or not _WARMUP_ON:
            return
        key = (int(head_dim), int(state_width), int(compress_ratio),
               int(bool(overlap)))
        if key in _WARMED:
            return
        _WARMED.add(key)
        dev = state_cache.device
        req0 = torch.zeros(1, dtype=token_to_req_indices.dtype, device=dev)
        pos0 = torch.zeros(1, dtype=all_positions.dtype, device=dev)
        slot_skip = torch.full((1,), -1, dtype=all_slots.dtype, device=dev)
        scratch = torch.zeros((1, head_dim), dtype=torch.float32, device=dev)
        gather_op(state_cache, req0, pos0, slot_skip, block_table, scratch,
                  block_size, head_dim, state_width, compress_ratio,
                  int(bool(overlap)))
        half_rope = rope_head_dim // 2
        cs = cos_sin_cache[:1]
        freqs = (torch.stack((cs[:, :half_rope], cs[:, half_rope:]), dim=-1)
                 .reshape(1, rope_head_dim).float().contiguous())
        normrope_op(scratch, rms_norm_weight.float().contiguous(),
                    torch.zeros(1, dtype=torch.int64, device=dev), freqs,
                    mode=2, compress_ratio=0, eps=rms_norm_eps)
        LOGGER.info("[compress-warm] loaded native modules for head=%d "
                    "state_width=%d ratio=%d overlap=%d", *key)

    def _compress_core(
        sel, wv, state_cache, token_to_req_indices, all_positions, all_slots,
        block_table, block_size, state_width, cos_sin_cache, kv_cache,
        kv_slot_mapping, head_dim, rope_head_dim,
        compress_ratio, overlap, rms_norm_weight, rms_norm_eps,
    ):
        normed = _native_compress(
            sel=sel,
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            all_positions=all_positions,
            all_slots=all_slots,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
        )
        if normed is not None:
            kv_slots = kv_slot_mapping.index_select(0, sel).long()
            _masked_paged_write(
                kv_cache,
                dest=kv_slots,
                write_ok=wv & (kv_slots >= 0),
                data=normed,
            )
            return

        coff = 2 if overlap else 1
        window = coff * compress_ratio
        nope_head_dim = head_dim - rope_head_dim
        half_rope = rope_head_dim // 2
        device = state_cache.device
        T = sel.shape[0]

        boundary_positions = all_positions.index_select(0, sel).long()
        boundary_req = token_to_req_indices.index_select(0, sel).long()

        offsets = torch.arange(-(window - 1), 1, device=device).unsqueeze(0)
        gather_pos = boundary_positions.unsqueeze(1) + offsets
        gather_mask = gather_pos >= 0
        gather_pos_safe = gather_pos.clamp(min=0)

        block_i = gather_pos_safe // block_size
        block_off = gather_pos_safe % block_size
        max_blocks_bt = block_table.shape[1]
        flat_bt_idx = (
            boundary_req.unsqueeze(1) * max_blocks_bt
            + block_i.clamp(max=max_blocks_bt - 1)
        ).reshape(-1)
        block_numbers = block_table.reshape(-1)[flat_bt_idx].reshape(T, window)

        tokens_in_window = torch.arange(window, device=device)
        head_offset = (
            (tokens_in_window >= compress_ratio).long() * head_dim
            if overlap
            else torch.zeros(window, dtype=torch.long, device=device)
        )

        # Strided gather straight out of the paged view. state_cache is a
        # non-contiguous view into the KV pool (observed in serve:
        # (17138, 4, 2048) stride (416384, 2048, 1)), so the previous
        # reshape(-1, last) had to materialise the WHOLE cache (~562 MB) just to
        # read T*window <= 2048 rows -- O(cache) work per call, 62 calls/step,
        # and unavoidable under FULL capture because the native path is gated
        # off there. Two-dim advanced indexing is bit-identical to
        # sc_flat[bn * block_size + bo] and capture-safe (docs/probe14_*.py).
        gathered = state_cache[
            block_numbers.reshape(-1).long(), block_off.reshape(-1).long()
        ].reshape(T, window, -1)

        ho = head_offset.unsqueeze(0).expand(T, -1)
        dim_idx = torch.arange(head_dim, device=device)
        kv_start = ho.unsqueeze(-1) + dim_idx
        score_start = (state_width + ho).unsqueeze(-1) + dim_idx

        kv_states = gathered.gather(2, kv_start).float()
        score_states = gathered.gather(2, score_start).float()

        mask_expand = gather_mask.unsqueeze(-1)
        score_states = score_states.masked_fill(~mask_expand, float("-inf"))
        kv_states = kv_states.masked_fill(~mask_expand, 0.0)

        weights = torch.softmax(score_states, dim=1)
        compressed = (kv_states * weights).sum(dim=1)

        variance = (compressed * compressed).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(variance + rms_norm_eps)
        normed_torch = compressed * rrms * rms_norm_weight.float().unsqueeze(0)

        compressed_pos = (boundary_positions // compress_ratio) * compress_ratio
        cs = cos_sin_cache[compressed_pos]
        cos_vals = cs[:, :half_rope]
        sin_vals = cs[:, half_rope:]

        rope_part = normed_torch[:, nope_head_dim:]
        rope_even = rope_part[:, 0::2]
        rope_odd = rope_part[:, 1::2]
        new_even = rope_even * cos_vals - rope_odd * sin_vals
        new_odd = rope_even * sin_vals + rope_odd * cos_vals
        normed_torch[:, nope_head_dim::2] = new_even
        normed_torch[:, nope_head_dim + 1::2] = new_odd

        kv_slots = kv_slot_mapping.index_select(0, sel).long()
        _masked_paged_write(
            kv_cache,
            dest=kv_slots,
            write_ok=wv & (kv_slots >= 0),
            data=normed_torch,
        )

    def compress_norm_rope_store_triton(
        state_cache, num_actual, token_to_req_indices, positions,
        slot_mapping, block_table, block_size, state_width,
        cos_sin_cache, kv_cache, k_cache_metadata, pdl_kwargs,
        head_dim, rope_head_dim, compress_ratio, overlap,
        use_fp4_cache, rms_norm_weight, rms_norm_eps,
        quant_block, token_stride, scale_dim,
    ):
        del pdl_kwargs, quant_block, token_stride, scale_dim, use_fp4_cache

        # --- Fused C4 path: flash_compress_4_decode + norm_rope ---
        global _c4_stashed_kv_score, _c4_stashed_ape
        _kv_score_input_full = _c4_stashed_kv_score
        _ape_param = _c4_stashed_ape
        if (_C4_FUSED_ENABLED and compress_ratio == 4
                and _kv_score_input_full is not None and _ape_param is not None):
            success = _fused_c4_path_full(
                state_cache=state_cache,
                num_actual=num_actual,
                positions=positions,
                slot_mapping=slot_mapping,
                block_size=block_size,
                block_table=block_table,
                token_to_req_indices=token_to_req_indices,
                cos_sin_cache=cos_sin_cache,
                kv_cache=kv_cache,
                kv_slot_mapping=k_cache_metadata.slot_mapping,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                compress_ratio=compress_ratio,
                rms_norm_weight=rms_norm_weight,
                rms_norm_eps=rms_norm_eps,
                kv_score_input_full=_kv_score_input_full,
                ape_param=_ape_param,
            )
            if success:
                _c4_stashed_kv_score = None
                _c4_stashed_ape = None
                return
            _c4_stashed_kv_score = None
            _c4_stashed_ape = None
            # Fall through to legacy path

        all_positions = positions[:num_actual]
        all_slots = slot_mapping[:num_actual]
        valid_mask = (all_slots >= 0) & ((all_positions + 1) % compress_ratio == 0)

        capturing_func = getattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        capturing = capturing_func()
        if capturing:
            sel = torch.arange(num_actual, device=state_cache.device)
            wv = valid_mask
        else:
            _warm_native_once(
                state_cache=state_cache,
                token_to_req_indices=token_to_req_indices,
                all_positions=all_positions, all_slots=all_slots,
                block_table=block_table, block_size=block_size,
                cos_sin_cache=cos_sin_cache, head_dim=head_dim,
                rope_head_dim=rope_head_dim, state_width=state_width,
                compress_ratio=compress_ratio, overlap=overlap,
                rms_norm_weight=rms_norm_weight, rms_norm_eps=rms_norm_eps,
            )
            _warm_paged_store_once(kv_cache)
            if not bool(valid_mask.any()):
                return
            sel = valid_mask.nonzero(as_tuple=True)[0]
            wv = torch.ones_like(sel, dtype=torch.bool)

        _compress_core(
            sel=sel,
            wv=wv,
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            all_positions=all_positions,
            all_slots=all_slots,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            kv_slot_mapping=k_cache_metadata.slot_mapping,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
        )

    setattr(fcqc_module, fn_name, compress_norm_rope_store_triton)
    setattr(fcqc_module, mod_attr, True)
    LOGGER.info(
        "Patched compress_norm_rope_store_triton into %s (native=%s)",
        fcqc_module.__name__,
        native_available,
    )


def apply(master_enabled_check: bool = True) -> List[str]:
    """Register lazy hooks covering compressor metadata + state save + compress pipeline."""
    if not master_enabled_check:
        return []

    from vllm_kunlun.config.deepseek_v4 import FeatureFlags

    flags = FeatureFlags()
    if not (flags.compressor_save_native or flags.compressor_vectorized_fallback):
        WarningOnce.emit(
            "dsv4-compressor-all-disabled",
            "Both compressor switches are off; skipping V4 compressed-KV hooks",
        )
        return []

    for utils_target in _SLOT_TARGETS:
        _register_lazy(
            utils_target,
            lambda m: _is_fn_patched(getattr(m, _SLOT_FN_NAME, None)),
            _install_slot_mapping,
        )

    _register_lazy(
        _SPS_TARGET_MODULE,
        lambda m: _is_fn_patched(getattr(m, "save_partial_states", None)),
        _install_save_partial_states,
    )
    _register_lazy(
        _VECT_TARGET_MODULE,
        lambda m: bool(
            getattr(
                m,
                f"compress_norm_rope_store_triton{_COMPRESSED_FN_ATTR}",
                False,
            )
        ),
        _install_compress_norm_rope_store_triton,
    )
    return []
