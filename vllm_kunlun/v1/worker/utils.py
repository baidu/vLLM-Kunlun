# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific monkey-patches for ``vllm.v1.worker.utils``.

This module does NOT replace ``vllm.v1.worker.utils``; it patches the
``KVBlockZeroer`` class in place so the very same class object captured
elsewhere (e.g. ``from vllm.v1.worker.utils import KVBlockZeroer`` at
``gpu_model_runner`` top level) is mutated.

Triggering: imported from ``vllm_kunlun.__init__`` after the import
hook is installed, ensuring upstream is loaded first.
"""

import logging

from vllm.v1.worker.utils import KVBlockZeroer as _upstream_cls

logger = logging.getLogger("vllm_kunlun")


def _init_meta(
    self,
    attn_groups_iter,
    kernel_block_sizes,
    cache_dtype,
    runner_only_attn_layers,
    static_forward_context,
):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    kv_entries = []
    # Dedup by Python object identity rather than ``data_ptr()``: K and V
    # tensors of an interleaved/strided layout can share underlying
    # storage (and therefore data_ptr), which would otherwise cause one
    # of them to be silently skipped — leaving half of the KV cache
    # un-zeroed across requests.
    seen_ids = set()
    for group in attn_groups_iter:
        spec = group.kv_cache_spec
        if type(spec) is not FullAttentionSpec:
            continue
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue
        kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
        if kernel_bs <= 0 or spec.block_size < kernel_bs:
            logger.warning(
                "[KunlunPlugin] KVBlockZeroer: skipping group with "
                "invalid block_size=%s vs kernel_bs=%s",
                spec.block_size,
                kernel_bs,
            )
            continue
        ratio = spec.block_size // kernel_bs
        block_dim = group.backend.get_kv_cache_block_dim(
            kernel_bs,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        # Reference shape from backend includes a leading "2" for K/V
        # split. Actual per-tensor shape may have that dim already
        # removed (kv_cache stored as list[K, V]). Compute the offset
        # to map reported block_dim onto the actual tensor.
        ref_shape = group.backend.get_kv_cache_shape(
            1234567,
            kernel_bs,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        ref_ndim = len(ref_shape)
        for layer_name in group.layer_names:
            if layer_name in runner_only_attn_layers:
                continue
            kv_list = static_forward_context[layer_name].kv_cache
            if not isinstance(kv_list, (list, tuple)):
                kv_list = [kv_list]
            for kv in kv_list:
                if isinstance(kv, list):
                    continue
                key = id(kv)
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                # Shift block_dim by the leading-dim difference.
                actual_dim = block_dim - (ref_ndim - kv.ndim)
                if actual_dim < 0 or actual_dim >= kv.ndim:
                    logger.warning(
                        "[KunlunPlugin] KVBlockZeroer: layer=%s "
                        "actual_dim=%s out of range (kv.ndim=%s, "
                        "block_dim=%s, ref_ndim=%s); skipping this "
                        "tensor to avoid corrupting wrong axis",
                        layer_name,
                        actual_dim,
                        kv.ndim,
                        block_dim,
                        ref_ndim,
                    )
                    continue
                kv_entries.append((kv, actual_dim, ratio))
    self._kv_entries = kv_entries
    # logger.info(
    #     "[KunlunPlugin] KVBlockZeroer.init_meta: %d kv tensors registered; "
    #     "shapes/dims/ratios=%s",
    #     len(kv_entries),
    #     [(tuple(t.shape), d, r) for (t, d, r) in kv_entries],
    # )


def _zero_block_ids(self, block_ids):
    # //todo because ssm_state dtype is only fp16, so zero_block_ids is not needed
    return
    if not block_ids or not getattr(self, "_kv_entries", None):
        return
    for kv, block_dim, ratio in self._kv_entries:
        dim_size = kv.shape[block_dim]
        if ratio == 1:
            for bid in block_ids:
                bi = int(bid)
                if 0 <= bi < dim_size:
                    kv.select(block_dim, bi).zero_()
        else:
            for bid in block_ids:
                base = int(bid) * ratio
                for j in range(ratio):
                    idx = base + j
                    if 0 <= idx < dim_size:
                        kv.select(block_dim, idx).zero_()


# Idempotent monkey-patch: safe under fork() and re-import.
if not getattr(_upstream_cls, "_kunlun_patched", False):
    _upstream_cls.init_meta = _init_meta
    _upstream_cls.zero_block_ids = _zero_block_ids
    _upstream_cls._kunlun_patched = True
    logger.info(
        "[KunlunPlugin] KVBlockZeroer patched in vllm_kunlun/v1/worker/utils.py"
    )
