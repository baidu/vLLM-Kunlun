# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize legacy Kunlun cache pages without copying storage.

Block transport treats the page contents as opaque rows. The original
head/token ordering stays in those bytes; only addressing views change.
"""

from dataclasses import replace
from math import prod

import torch
from lmcache.integration.vllm.experimental import dispatch
from lmcache.integration.vllm.kv_cache_group_edits import (
    apply_kv_cache_group_edits,
    validate_kv_cache_groups,
)
from lmcache.integration.vllm.kv_cache_groups import create_engine_group_infos_from_vllm
from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPConnector as UpstreamLMCacheMPConnector,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)


def _attention_page_view(cache: torch.Tensor, logical_block_size: int) -> torch.Tensor:
    """Group dense legacy kernel pages into scheduler-addressed HND pages."""
    if cache.ndim != 5 or cache.shape[0] != 2:
        raise ValueError(
            "Kunlun legacy attention expects [2, blocks, heads, tokens, dim], "
            f"got {tuple(cache.shape)}"
        )
    _, kernel_blocks, heads, kernel_size, width = cache.shape
    if logical_block_size <= 0 or kernel_size <= 0:
        raise ValueError("attention block sizes must be positive")
    if logical_block_size % kernel_size:
        raise ValueError("logical block size must be a multiple of kernel block size")
    ratio = logical_block_size // kernel_size
    if kernel_blocks % ratio:
        raise ValueError("kernel page count must contain complete logical blocks")
    inner_shape = (heads, kernel_size, width)
    inner_strides = (kernel_size * width, width, 1)
    if any(
        size > 1 and actual != expected
        for size, actual, expected in zip(
            inner_shape, cache.stride()[2:], inner_strides
        )
    ):
        raise ValueError("legacy attention requires dense data inside each K/V page")
    half_kernel = heads * kernel_size * width
    half_logical = ratio * half_kernel
    num_blocks = kernel_blocks // ratio
    row_width = heads * width
    kv_stride, block_stride = cache.stride()[:2]
    if kv_stride == half_kernel and block_stride >= 2 * half_kernel:
        # Hybrid allocation puts K and V together in each physical kernel page.
        # Combining subpages preserves their bytes, including their K/V ordering.
        if ratio > 1 and block_stride != 2 * half_kernel:
            raise ValueError("padded kernel pages cannot be joined without copying")
        return cache.as_strided(
            (num_blocks, 1, logical_block_size, 2 * row_width),
            (ratio * block_stride, 2 * half_logical, 2 * row_width, 1),
        )
    if block_stride == half_kernel and kv_stride >= kernel_blocks * half_kernel:
        # Layer-local allocation keeps the whole K pool before the whole V pool.
        return cache.as_strided(
            (2, num_blocks, 1, logical_block_size, row_width),
            (kv_stride, half_logical, half_logical, row_width, 1),
        )
    raise ValueError("unsupported legacy attention K/V and block strides")


def _mamba_state_row(cache, spec) -> torch.Tensor:
    """Expose packed conv/SSM bytes while retaining the physical page stride."""
    if isinstance(cache, torch.Tensor):
        return cache.view(torch.int8)
    if not cache or len(cache) != len(spec.shapes):
        raise ValueError("Mamba states must match the declared conv/SSM shapes")
    first = cache[0]
    offset = 0
    for state, shape, dtype in zip(cache, spec.shapes, spec.dtypes):
        if (
            tuple(state.shape) != (first.shape[0], *shape)
            or state.dtype != dtype
            or state.data_ptr() != first.data_ptr() + offset
            or state.stride(0) * state.element_size() != spec.page_size_bytes
        ):
            raise ValueError("Mamba states must share one packed page per block")
        offset += prod(shape) * dtype.itemsize
    return first.view(torch.int8).as_strided(
        (first.shape[0], 1, 1, offset),
        (spec.page_size_bytes, offset, offset, 1),
    )


def prepare_kv_cache_views(kv_cache_config: KVCacheConfig | None, kv_caches):
    """Register logical Attention/Mamba pages with a consistent HND hint."""
    validate_kv_cache_groups(kv_cache_config)
    if kv_cache_config is None:
        return dict(kv_caches)
    result = dict(kv_caches)
    mamba_groups = []
    for group in kv_cache_config.kv_cache_groups:
        per_layer_specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        if get_kv_cache_spec_kind(group.kv_cache_spec) == KVCacheSpecKind.MAMBA:
            mamba_groups.append(group)
        for name in group.layer_names:
            spec = per_layer_specs[name] if per_layer_specs else group.kv_cache_spec
            if get_kv_cache_spec_kind(spec) == KVCacheSpecKind.MAMBA:
                result[name] = _mamba_state_row(result[name], spec)
            elif isinstance(spec, AttentionSpec):
                cache = result[name]
                if isinstance(cache, torch.Tensor) and cache.ndim == 5:
                    result[name] = _attention_page_view(cache, spec.block_size)
    if mamba_groups:
        # Reuse upstream's unified-row tiling and alignment inside this page.
        result = apply_kv_cache_group_edits(
            replace(kv_cache_config, kv_cache_groups=mamba_groups),
            result,
            layout_hints={"kv_layout": "HND"},
        )
    return result


class LMCacheMPConnector(UpstreamLMCacheMPConnector):
    """LMCache MP connector for Kunlun HND attention and Mamba caches."""

    def register_kv_caches(self, kv_caches):
        layout_hints = {"kv_layout": "HND"}
        kv_caches = prepare_kv_cache_views(self._kv_cache_config, kv_caches)
        engine_group_infos = create_engine_group_infos_from_vllm(
            self._kv_cache_config,
            kv_caches,
            layout_hints=layout_hints,
            dcp_size=self._dcp_size,
        )
        self.worker_adapter.register_kv_caches(
            kv_caches,
            engine_group_infos=engine_group_infos,
            layout_hints=layout_hints,
        )
        if self.dispatcher is not None:
            dispatch(
                self.dispatcher,
                "register",
                kv_caches=kv_caches,
                kv_cache_config=self._kv_cache_config,
                vllm_config=self._vllm_config,
            )
