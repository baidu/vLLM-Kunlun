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
import sys
from collections import defaultdict
from typing import Any

import torch

import vllm.v1.worker.utils as _upstream_utils
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
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


def _kunlun_allows_multi_attention_layer_index() -> bool:
    is_kunlun = getattr(current_platform, "is_kunlun", None)
    if callable(is_kunlun) and is_kunlun():
        return True
    return (
        current_platform.is_out_of_tree()
        and getattr(current_platform, "device_name", None) == "cuda"
        and getattr(current_platform, "device_type", None) == "cuda"
    )


def _kunlun_bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Any],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """Kunlun-compatible ``bind_kv_cache``.

    Upstream already permits multiple attention cache names sharing one layer
    index on CUDA/ROCm/XPU/CPU. Kunlun intentionally stays PlatformEnum.OOT
    while exposing tensors as ``cuda`` devices, so MiniMax-M3's main cache plus
    ``.index_cache`` side cache hit upstream's fallback NotImplementedError.
    This keeps the upstream behavior and adds the narrow Kunlun OOT case only.
    """

    assert len(runner_kv_caches) == 0

    index2name: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            if (
                current_platform.is_cuda_alike()
                or current_platform.is_xpu()
                or current_platform.is_cpu()
                or _kunlun_allows_multi_attention_layer_index()
            ):
                pass
            else:
                raise NotImplementedError
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


def _patch_bind_kv_cache_aliases() -> None:
    if getattr(_upstream_utils.bind_kv_cache, "_kunlun_patched", False):
        return

    _kunlun_bind_kv_cache._kunlun_patched = True
    _upstream_utils.bind_kv_cache = _kunlun_bind_kv_cache

    for module_name in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.attn_utils",
    ):
        mod = sys.modules.get(module_name)
        if mod is not None and hasattr(mod, "bind_kv_cache"):
            mod.bind_kv_cache = _kunlun_bind_kv_cache

    logger.info("[KunlunPlugin] bind_kv_cache patched for Kunlun OOT side caches")


_patch_bind_kv_cache_aliases()
