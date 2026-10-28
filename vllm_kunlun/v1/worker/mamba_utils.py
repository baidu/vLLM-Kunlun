# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun replacement for ``vllm.v1.worker.mamba_utils``.

Installed as a whole-module redirect from ``registration/module_redirects.py``.
Two upstream mechanisms do not run on Kunlun XPU:

* ``batch_memcpy`` must go through ``torch.ops.xspeedgate_ops.batch_memcpy``
  instead of launching the Triton ``batch_memcpy_kernel``, and
  ``MambaCopyBuffers.create`` has to allocate the int64 pointer/size buffers
  that op is specified for.
* The mamba "align" cache mode drives its state copies from fused Triton
  kernels (``postprocess_mamba_fused_kernel`` and friends).  Those fail with
  ``RuntimeError: Triton Error [CUDA]: CUDA_ERROR_NOT_SUPPORTED``, and align
  mode is reached on every decode step as soon as prefix caching is enabled
  for a hybrid model, so ``postprocess_mamba_align_gpu`` is reimplemented here
  as a device-side plan (``_q38_align_plan_gpu``), with a blocking-D2H
  fallback behind ``Q38_ALIGN_GPU_PLAN=0``.

Whatever this module does not reimplement is served from the real upstream
module by the ``__getattr__`` at the bottom, so the export surface stays
complete -- including the V2 ``MambaSpecDecodeGPUContext`` that
``vllm/v1/worker/gpu/model_states/mamba_hybrid.py`` imports at module scope.
"""

import dataclasses
import itertools
import os
from collections.abc import Callable
from math import prod
from typing import Any

import numpy as np
import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaCopySpec,
    MambaStateCopyFunc,
    MambaStateCopyFuncsByType,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec as _upstream_conv_copy_spec,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))

        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    torch.ops.xspeedgate_ops.batch_memcpy(src_ptrs, dst_ptrs, sizes)


#     grid = (batch,)
#     BLOCK_SIZE = 1024
#     batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)

# def _make_uint8_view_from_ptr(ptr: int, size: int, device: torch.device) -> torch.Tensor:
#    storage = torch._C._construct_storage_from_data_pointer(ptr, device, size)
#    tensor = torch.empty(0, dtype=torch.uint8, device=device)
#    return tensor.set_(storage, 0, (size,), (1,))


# def batch_memcpy(src_ptrs, dst_ptrs, sizes):
#    batch = src_ptrs.shape[0]
#    assert dst_ptrs.shape[0] == batch
#    assert sizes.shape[0] == batch
#    if batch == 0:
#        return
#
#    device = src_ptrs.device
#    src_ptrs_cpu = src_ptrs.detach().cpu().tolist()
#    dst_ptrs_cpu = dst_ptrs.detach().cpu().tolist()
#    sizes_cpu = sizes.detach().cpu().tolist()
#
#    for src_ptr, dst_ptr, size in zip(src_ptrs_cpu, dst_ptrs_cpu, sizes_cpu):
#        if size <= 0 or src_ptr == dst_ptr:
#            continue
#        src = _make_uint8_view_from_ptr(src_ptr, size, device)
#        dst = _make_uint8_view_from_ptr(dst_ptr, size, device)
#        dst.copy_(src, non_blocking=True)


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i in range(len(kv_cache_config.kv_cache_groups)):
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
        if isinstance(kv_cache_spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(kv_cache_spec)
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"
    # __Q38_ALIGN_SPEC__: 上游断言所有 MambaSpec 完全相等，但 Qwen4Exp 同时有
    # GDN（conv + ssm 两个状态）和 PLE short-conv（只有 conv 一个状态），两个
    # spec 本来就不相等。调用方只用返回 spec 的 block_size /
    # num_speculative_blocks，所以只断言这两项一致即可。
    ref = mamba_specs[0]
    for spec in mamba_specs[1:]:
        assert (
            spec.block_size == ref.block_size
        ), f"mamba block_size mismatch: {spec.block_size} vs {ref.block_size}"
        assert spec.num_speculative_blocks == ref.num_speculative_blocks, (
            "mamba num_speculative_blocks mismatch: "
            f"{spec.num_speculative_blocks} vs {ref.num_speculative_blocks}"
        )
    return mamba_group_ids, ref


def validate_mamba_state_copy_funcs(kv_cache_config, copy_funcs):
    """Validate backend-specific copy functions against physical states."""
    if not isinstance(copy_funcs, dict):
        return
    found = set()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if not isinstance(spec, MambaSpec):
            continue
        found.add(spec.mamba_type)
        funcs = copy_funcs.get(spec.mamba_type)
        assert funcs is not None, f"missing copy functions for {spec.mamba_type}"
        assert len(funcs) == len(spec.shapes), (
            f"copy/state count mismatch for {spec.mamba_type}: "
            f"{len(funcs)} vs {len(spec.shapes)}"
        )
    assert found.issubset(copy_funcs), "copy mapping contains incomplete backend set"


@dataclasses.dataclass
class MambaCopyBuffers:
    src_ptrs: CpuGpuBuffer
    dst_ptrs: CpuGpuBuffer
    sizes: CpuGpuBuffer
    mamba_group_ids: list[int]
    mamba_spec: MambaSpec
    offset: int = 0

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: MambaStateCopyFuncsByType | tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaCopyBuffers":
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
        if not isinstance(copy_funcs, dict):
            copy_funcs = {mamba_spec.mamba_type: copy_funcs}
        for gid in mamba_group_ids:
            spec = kv_cache_config.kv_cache_groups[gid].kv_cache_spec
            funcs = copy_funcs.get(spec.mamba_type)
            assert (
                funcs is not None
            ), f"missing Mamba copy functions for {spec.mamba_type}"
            assert len(funcs) == len(spec.shapes), (
                f"Mamba state/copy mismatch for {spec.mamba_type}: "
                f"{len(spec.shapes)} states vs {len(funcs)} copy functions"
            )
        entries_per_req = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            * len(
                copy_funcs[
                    kv_cache_config.kv_cache_groups[gid].kv_cache_spec.mamba_type
                ]
            )
            for gid in mamba_group_ids
        )
        n = max_num_reqs * entries_per_req
        return cls(
            src_ptrs=make_buffer(n, dtype=torch.int64),
            dst_ptrs=make_buffer(n, dtype=torch.int64),
            sizes=make_buffer(n, dtype=torch.int64),
            mamba_group_ids=mamba_group_ids,
            mamba_spec=mamba_spec,
        )


# __Q38_OVLFIX__
_Q38_CV_LVL = os.environ.get("Q38_COPY_VERIFY", "")
_Q38_CV = _Q38_CV_LVL not in ("", "0")
_Q38_CT = os.environ.get("Q38_COPY_TORCH", "") not in ("", "0")
_Q38_FIX = os.environ.get("Q38_OVL_FIX", "") not in ("", "0") and not _Q38_CT
_Q38_ENTRIES: list = []
_Q38_SAFE: list = []
_Q38_STAT = {"n": 0, "bad": 0, "ovl": 0, "fixed": 0}


def _q38_row(state, blk):
    row = state[blk]
    try:
        return row.reshape(-1)
    except Exception:  # noqa: BLE001
        return row.contiguous().reshape(-1)


def _q38_locate(state, copy_spec):
    """从地址反推 (来源 block, block 内元素偏移, 元素个数)。"""
    tot = (int(copy_spec.start_addr) - int(state.data_ptr())) // state.element_size()
    bs = state.stride(0)
    return tot // bs, tot % bs, int(copy_spec.num_elements)


def _q38_is_overlap(state, copy_spec, dest_block_id):
    sblk, off, n = _q38_locate(state, copy_spec)
    return sblk == int(dest_block_id) and 0 < off < n


def _q38_note(state, copy_spec, dest_block_id, req_state, path):
    sblk, off, n = _q38_locate(state, copy_spec)
    exp = None
    if _Q38_CV:
        exp = _q38_row(state, sblk)[off : off + n].clone()
    _Q38_ENTRIES.append(
        (
            state,
            sblk,
            int(dest_block_id),
            off,
            n,
            getattr(req_state, "req_id", "?"),
            exp,
            path,
        )
    )


def _q38_run_copies(entries):
    for state, sblk, dblk, off, n, rid, exp, path in entries:
        s = _q38_row(state, sblk)
        d = _q38_row(state, dblk)
        d[:n].copy_(s[off : off + n].clone())


def _q38_verify():
    st = _Q38_STAT
    bad = []
    for state, sblk, dblk, off, n, rid, exp, path in _Q38_ENTRIES:
        st["n"] += 1
        if sblk == dblk and 0 < off < n:
            st["ovl"] += 1
        d = _q38_row(state, dblk)
        if not bool((d[:n] == exp).all().item()):
            st["bad"] += 1
            bad.append(
                (rid, sblk, dblk, off, n, int((d[:n] != exp).sum().item()), path)
            )
    for rid, si, di, off, n, nd, path in bad[:6]:
        pass


def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: MambaStateCopyFuncsByType | tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    # __Q38_ALIGN_SPEC__: 上游 get_conv_copy_spec 按逻辑形状切片，Kunlun 的 PLE
    # conv 状态逻辑是 (dim, state_len)、物理是 state-major，必须按内存布局平移。
    mamba_state_copy_funcs = (
        {k: kunlun_state_copy_funcs(v) for k, v in mamba_state_copy_funcs.items()}
        if isinstance(mamba_state_copy_funcs, dict)
        else kunlun_state_copy_funcs(mamba_state_copy_funcs)
    )

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        group_spec = kv_cache_config.kv_cache_groups[mamba_group_id].kv_cache_spec
        funcs = (
            mamba_state_copy_funcs[group_spec.mamba_type]
            if isinstance(mamba_state_copy_funcs, dict)
            else mamba_state_copy_funcs
        )
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )

                _q38_ovl = _Q38_FIX and _q38_is_overlap(state, copy_spec, dest_block_id)
                if _Q38_CV or _Q38_CT or _q38_ovl:
                    _q38_note(
                        state,
                        copy_spec,
                        dest_block_id,
                        req_state,
                        "safe" if _q38_ovl else "batch",
                    )
                if _q38_ovl:
                    _Q38_SAFE.append(_Q38_ENTRIES[-1])
                    _Q38_STAT["fixed"] += 1
                    continue
                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):
    n = copy_bufs.offset
    if n:
        if _Q38_CT:
            _q38_run_copies(_Q38_ENTRIES)
        else:
            batch_memcpy(
                copy_bufs.src_ptrs.copy_to_gpu(n),
                copy_bufs.dst_ptrs.copy_to_gpu(n),
                copy_bufs.sizes.copy_to_gpu(n),
            )
    if _Q38_SAFE:
        _q38_run_copies(_Q38_SAFE)
        del _Q38_SAFE[:]
    if _Q38_CV and _Q38_ENTRIES:
        _q38_verify()
    del _Q38_ENTRIES[:]


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    # We need to clear mamba_state_idx for resumed requests. When requests are
    # force-preempted (e.g., during reset_prefix_cache / KV cache flush),
    # they appear in resumed_req_ids without a corresponding entry in
    # preempted_req_ids, leaving stale mamba_state_idx entries that can
    # point to block indices beyond the new (smaller) block allocation.
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)

    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


@dataclasses.dataclass
class MambaBuffers:
    """Single owner for all mamba-specific runner buffers."""

    preprocess: MambaCopyBuffers
    # __Q38_ALIGN_SPEC__: 投机解码（MTP）+ hybrid 模型时 runner 需要 align 的
    # postprocess。上游是 triton fused kernel 的 MambaSpecDecodeGPUContext，
    # XPU 上没有对应内核，这里换成 CPU 版上下文（见 MambaSpecDecodeCPUContext）。
    postprocess_align: "MambaSpecDecodeCPUContext | None" = None

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: MambaStateCopyFuncsByType | tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
        device=None,
        with_postprocess_align: bool = False,
    ) -> "MambaBuffers":
        postprocess_align = None
        if with_postprocess_align:
            postprocess_align = MambaSpecDecodeCPUContext.create(
                max_num_reqs, kv_cache_config, copy_funcs, make_buffer
            )
        return cls(
            preprocess=MambaCopyBuffers.create(
                max_num_reqs, kv_cache_config, copy_funcs, make_buffer
            ),
            postprocess_align=postprocess_align,
        )


def get_hybrid_attention_mamba_layout(
    kv_cache_shape: tuple[int, ...],
    kv_cache_stride: tuple[int, ...],
    kv_cache_spec: AttentionSpec,
    block_dim: int,
    layer_idx: int,
    kernel_block_size: int,
) -> tuple[tuple[int, ...], int]:
    """
    Compute the stride and storage offset for the hybrid attention+mamba layout.

    Args:
        kv_cache_shape: The shape of the KV cache tensor.
        kv_cache_stride: The stride of the KV cache tensor.
        kv_cache_spec: The specification of the KV cache.
        layer_idx: The index of the layer.
        kernel_num_blocks: The number of kernel blocks.
        kernel_block_size: The size of the kernel block.
    Returns:
        A tuple containing the target stride and storage offset.
    """
    target_stride_list = list(kv_cache_stride)
    storage_offset = 0

    attn_pack_size = kv_cache_spec.pack_size
    # block_dim: 0 means (num_blocks, 2, ...); 1 means (2, num_blocks, ...).
    if block_dim != 0:
        # Hybrid attention+mamba uses (2, num_blocks, ...) logical shape but
        # (num_blocks, 2, ...) physical layout.
        assert kv_cache_shape[0] == 2, (
            "Fail to determine whether the layout is "
            "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
            f"a tensor of shape {kv_cache_shape}"
        )
        assert block_dim == 1
        hidden_size = prod(kv_cache_shape[2:])
        target_stride_list[0] = hidden_size
        target_stride_list[1] = 2 * hidden_size
    # When multiple attention layers share one physical KV cache block
    # (attn_pack_size > 1), scale the block-dim stride by attn_pack_size
    # and compute this layer's element offset within the shared block.
    if attn_pack_size > 1:
        target_stride_list[block_dim] *= attn_pack_size
        dtype_size = get_dtype_size(kv_cache_spec.dtype)
        num_element_per_page = kv_cache_spec.page_size_bytes // dtype_size
        num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
        num_element_per_attn_pack = (
            num_element_per_page // num_blocks_per_kv_block // attn_pack_size
        )
        attn_pack_idx = layer_idx % attn_pack_size
        storage_offset = attn_pack_idx * num_element_per_attn_pack
    return tuple(target_stride_list), storage_offset


def postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    If a blocks is converted from partial block to full block in this step, copy the
    state from the block for running state to the new full block.
    """
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = (
            num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        )
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        aligned_new_computed_tokens = (
            new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        )
        # TODO: how to ensure all blocks that cache_blocks called are cached here?
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


# ---------------------------------------------------------------------------
# __Q38_ALIGN_SPEC__ align（prefix caching）+ 投机解码在 Kunlun XPU 上的实现
# ---------------------------------------------------------------------------
# 上游 0.25.1 的 postprocess_mamba_align_gpu 依赖 triton fused kernel
# (MambaSpecDecodeGPUContext + postprocess_mamba_fused_kernel)，好处是不用把
# num_accepted_tokens 同步回 CPU。XPU 上没有这些内核（本文件顶部的 batch_memcpy
# 也已换成 xspeedgate 算子），所以这里退一步：
#   1) 前向之前把 postprocess 需要的 per-request 元数据暂存到 CPU
#      (num_scheduled / num_computed / num_draft / state_idx / req_state)，
#      因为这些值在前向之后会被 runner 更新掉；
#   2) 前向之后同步一次 num_accepted_tokens（D2H），在 CPU 上算搬运计划；
#   3) 复用已有的 collect_mamba_copy_meta + batch_memcpy 做实际拷贝。
# 拷贝只在跨 mamba block（block_size，默认 128 token）边界那一步才发生；而且部署
# 本来就是 --no-async-scheduling，多这一次 D2H 不引入新的停顿类型。


def _row_state_axis(row: torch.Tensor) -> int:
    # conv 状态里状态长度所在的轴：内存上最外层（stride 最大）的那根。
    strides = row.stride()
    return max(range(row.dim()), key=lambda d: strides[d])


def kunlun_conv_copy_spec(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    # 按物理布局平移 conv 滑窗，替代上游 get_conv_copy_spec。
    #
    # 上游按逻辑形状分 SD / DS 两条路，且 DS + offset > 0 直接断言必须交给 fused
    # kernel。Kunlun 上 GDN 的 conv 状态逻辑物理都是 (state_len, dim)，而 PLE 的
    # 逻辑是 (dim, state_len)、物理是 state-major（stride (1, dim)，见
    # ple_xpu_state.row_plan）。两者的共同点是：状态长度轴在内存上是最外层、行内
    # 紧凑。所以统一按 stride 最大的那根轴平移，与逻辑朝向无关。
    src_block_id = int(block_ids[cur_block_idx])
    row = state[src_block_id]
    # __Q38_ALIGN_FIX1__: num_accepted_tokens 可能是 numpy int32（来自
    # input_batch.num_accepted_tokens_cpu），下面要和 64 位指针相加，必须先转
    # Python int，否则 numpy 按 int32 算，指针一加就 OverflowError。
    offset = int(num_accepted_tokens) - 1
    if offset == 0:
        return MambaCopySpec(start_addr=row.data_ptr(), num_elements=row.numel())
    axis = _row_state_axis(row)
    state_len = int(row.shape[axis])
    inner = int(row.stride()[axis])
    assert (
        0 <= offset < state_len
    ), f"conv state shift out of range: offset={offset} state_len={state_len}"
    assert inner * state_len == row.numel(), (
        "conv state row is not compact along the state axis: "
        f"shape={tuple(row.shape)} stride={tuple(row.stride())}"
    )
    return MambaCopySpec(
        start_addr=int(row.data_ptr()) + offset * inner * state.element_size(),
        num_elements=(state_len - offset) * inner,
    )


_Q38_COPY_FUNCS_CACHE: dict[tuple, tuple] = {}


def kunlun_state_copy_funcs(
    funcs: tuple[MambaStateCopyFunc, ...],
) -> tuple[MambaStateCopyFunc, ...]:
    # 把上游 get_conv_copy_spec 替换成 kunlun_conv_copy_spec，其余（temporal）
    # 保持不变：temporal 是整行指针拷贝，padding 在行之间，逻辑朝向无关。
    cached = _Q38_COPY_FUNCS_CACHE.get(funcs)
    if cached is None:
        cached = tuple(
            (
                kunlun_conv_copy_spec
                if (
                    f is _upstream_conv_copy_spec
                    or getattr(f, "__name__", "") == "get_conv_copy_spec"
                )
                else f
            )
            for f in funcs
        )
        _Q38_COPY_FUNCS_CACHE[funcs] = cached
    return cached


@dataclasses.dataclass
class MambaSpecDecodeCPUContext:
    # align + 投机解码时 postprocess 需要的上下文（对应上游
    # MambaSpecDecodeGPUContext，但元数据留在 CPU 的 numpy 数组里）。
    copy_bufs: MambaCopyBuffers
    num_scheduled_tokens: Any
    num_computed_tokens: Any
    num_draft_tokens: Any
    state_idx: Any
    req_states: list
    num_reqs: int = 0
    staged: bool = False
    # __Q38_ALIGN_GPUPLAN__: 设备侧算搬运计划要用的 per-request 输入。这些值前向
    # 之前就已知，跟 stage 一起 H2D，不引入任何回读。
    running: Any = None
    state_idx_gpu: Any = None
    valid: Any = None
    accepted_out: Any = None
    plan_desc: Any = None

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaSpecDecodeCPUContext":
        return cls(
            copy_bufs=MambaCopyBuffers.create(
                max_num_reqs, kv_cache_config, copy_funcs, make_buffer
            ),
            num_scheduled_tokens=np.zeros(max_num_reqs, dtype=np.int64),
            num_computed_tokens=np.zeros(max_num_reqs, dtype=np.int64),
            num_draft_tokens=np.zeros(max_num_reqs, dtype=np.int64),
            state_idx=np.zeros(max_num_reqs, dtype=np.int64),
            req_states=[None] * max_num_reqs,
            # __Q38_ALIGN_GPUPLAN__
            running=make_buffer(max_num_reqs, dtype=torch.int64),
            state_idx_gpu=make_buffer(max_num_reqs, dtype=torch.int64),
            valid=make_buffer(max_num_reqs, dtype=torch.int64),
            accepted_out=make_buffer(max_num_reqs, dtype=torch.int32),
        )


def stage_postprocess_inputs_to_gpu(
    ctx: MambaSpecDecodeCPUContext,
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    num_reqs: int,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
) -> None:
    # 名字沿用上游（runner 直接按这个名字调用）。这里其实是暂存到 CPU：这些值
    # 前向之后会被 runner 更新，postprocess 需要的是本步开始时的快照。
    # 必须在 preprocess_mamba 之后调用，这样 mamba_state_idx 已是本步的
    # running-state block。
    num_scheduled = scheduler_output.num_scheduled_tokens
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = requests[req_id]
        ctx.req_states[i] = req_state
        ctx.num_scheduled_tokens[i] = num_scheduled[req_id]
        ctx.num_computed_tokens[i] = req_state.num_computed_tokens
        ctx.num_draft_tokens[i] = len(spec_tokens.get(req_id, ()))
        ctx.state_idx[i] = mamba_state_idx[req_id]
        # __Q38_ALIGN_GPUPLAN__: 同一份信息喂给设备侧计划。running = 本步结束时
        # running state 覆盖的 token 数（不含 draft），与原 CPU 循环同一个公式。
        ctx.running.np[i] = (
            ctx.num_computed_tokens[i]
            + ctx.num_scheduled_tokens[i]
            - ctx.num_draft_tokens[i]
        )
        ctx.state_idx_gpu.np[i] = ctx.state_idx[i]
        ctx.valid.np[i] = 1
    if num_reqs and _Q38_GPUPLAN and ctx.running is not None:
        ctx.running.copy_to_gpu(num_reqs)
        ctx.state_idx_gpu.copy_to_gpu(num_reqs)
        ctx.valid.copy_to_gpu(num_reqs)
    ctx.num_reqs = num_reqs
    ctx.staged = True


_Q38_LAYOUT_LOGGED = False


def _q38_log_state_layout_once(kv_cache_config, forward_context, mamba_group_ids):
    # 只打一次：确认每个 mamba 状态的物理布局与 kunlun_conv_copy_spec 的假设一致。
    global _Q38_LAYOUT_LOGGED
    if _Q38_LAYOUT_LOGGED:
        return
    _Q38_LAYOUT_LOGGED = True
    for gid in mamba_group_ids:
        layer_names = kv_cache_config.kv_cache_groups[gid].layer_names
        name = layer_names[0]
        attention = forward_context[name]
        for si, state in enumerate(attention.kv_cache):
            # __Q38_STORAGE__: 真实 storage 有多大？高 block id 会不会越界？
            try:
                _nb = state.untyped_storage().nbytes()
            except Exception as _e:  # noqa: BLE001
                _nb = -1


# ---------------------------------------------------------------------------
# __Q38_ALIGN_GPUPLAN__ 方案 A：align postprocess 的搬运计划全部在设备侧算
# ---------------------------------------------------------------------------
# 原路径（下面 postprocess_mamba_align_gpu 里 Q38_ALIGN_GPU_PLAN=0 的那条）每步都
# 要 num_accepted_tokens.cpu()，这是一次阻塞 D2H，正好卡在主模型和 MTP draft 之
# 间，实测 +4.9 ms/step（ITL 33.3 -> 38.2 ms）。这里把「谁要搬、从哪搬到哪、搬多
# 少字节」全部改成张量算术，host 一次都不读设备上的值：
#   * per-request 的 running / state_idx 在前向之前就已知，跟 stage 一起 H2D；
#   * num_accepted_tokens 留在设备上参与计算；
#   * block id 从 input_batch 的 block table GPU 张量 gather（本步 _prepare_inputs
#     里已经 commit_block_table，和 req_state.block_ids 同源）；
#   * 不活跃的条目 size 置 0：xspeedgate_ops.batch_memcpy 会安全跳过（实测 size==0
#     不碰目标，条目数 512->4096、活跃 0->512 都是 0.006 ms/call）；
#   * src_block == dest_block 时把 accepted 改写成 1 也在设备上做（torch.where），
#     然后 non_blocking D2H。语义与非 align 分支一致：下一步 _prepare_inputs 里
#     num_accepted_tokens_event.synchronize() 之后才有人读这个 CPU 副本。
# 条目顺序（group -> layer -> state）与 collect_mamba_copy_meta 完全一致，所以
# Q38_ALIGN_PLAN_VERIFY=1 可以逐条目对比两条路算出来的 (src, dst, size)。
# 同块 conv 平移（src_block == dest_block 且 bias > 0，align 下是常态）dst 与 src
# 在同一行内重叠，batch_memcpy 内部是否保证读写顺序未知，所以这类条目走 scratch
# 两跳：src -> scratch -> dst，多两次 0.006 ms 的发射，换掉一个未定义行为。
# 与原路径的一个语义差异：bias == state_len 时 conv 该搬的字节数正好是 0，原
# CPU 路径的 assert 0 <= offset < state_len 会炸，这里自然退化成「不搬 conv，只
# 搬 temporal」，和上游 SD 布局切片 state[blk, offset:] 的语义一致。
_Q38_GPUPLAN = os.environ.get("Q38_ALIGN_GPU_PLAN", "1") not in ("", "0")
_Q38_PLAN_VERIFY = os.environ.get("Q38_ALIGN_PLAN_VERIFY", "") not in ("", "0")
_Q38_PLAN_OVL_SCRATCH = os.environ.get("Q38_PLAN_OVL_SCRATCH", "1") not in ("", "0")
_Q38_PLAN_STAT = {"steps": 0, "cmp": 0, "bad": 0}


@dataclasses.dataclass
class _Q38PlanDesc:
    """每个 (group, layer, state) 条目的静态描述符，常驻 GPU。"""

    base: torch.Tensor  # int64[E] 状态张量首地址
    blk_bytes: torch.Tensor  # int64[E] block 之间的字节步长
    row_bytes: torch.Tensor  # int64[E] 一个 block 内有效字节数
    inner_bytes: torch.Tensor  # int64[E] conv: 每个状态位置的字节数；temporal: 0
    col_shift: torch.Tensor  # int64[E] conv: 0；temporal: 1（源列随 bias 前移）
    is_conv: torch.Tensor  # bool[E]
    scratch_off: torch.Tensor  # int64[E] 条目在每请求 scratch 段内的字节偏移
    scratch: torch.Tensor  # uint8[max_num_reqs * scratch_stride] 或 None
    scratch_stride: int
    group_slices: list  # [(group_id, start, stop)]
    entries: int
    # __Q38_PLAN_V2__: 各条目在 cat 后的大 block table 里的列偏移（懒算，首次拿到
    # block table 的列数才知道），ncol 变了就重算。
    col_off: Any = None
    ncol: int = 0


def _q38_is_conv_func(func) -> bool:
    return (
        func is kunlun_conv_copy_spec
        or func is _upstream_conv_copy_spec
        or getattr(func, "__name__", "")
        in ("get_conv_copy_spec", "kunlun_conv_copy_spec")
    )


def _q38_build_plan_desc(
    max_num_reqs: int,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_group_ids: list[int],
    mamba_state_copy_funcs: Any,
    device,
) -> _Q38PlanDesc:
    # 条目顺序必须与 collect_mamba_copy_meta 的三重循环一致。
    funcs_by_type = (
        {k: kunlun_state_copy_funcs(v) for k, v in mamba_state_copy_funcs.items()}
        if isinstance(mamba_state_copy_funcs, dict)
        else kunlun_state_copy_funcs(mamba_state_copy_funcs)
    )
    base, blk_bytes, row_bytes, inner_bytes, col_shift, is_conv = [], [], [], [], [], []
    scratch_off, slices = [], []
    stride = 0
    for gid in mamba_group_ids:
        start = len(base)
        group = kv_cache_config.kv_cache_groups[gid]
        funcs = (
            funcs_by_type[group.kv_cache_spec.mamba_type]
            if isinstance(funcs_by_type, dict)
            else funcs_by_type
        )
        for layer_name in group.layer_names:
            kv_caches: list[torch.Tensor] = forward_context[layer_name].kv_cache
            assert len(kv_caches) == len(funcs), (
                f"state/copy-func mismatch on {layer_name}: "
                f"{len(kv_caches)} vs {len(funcs)}"
            )
            for state, func in zip(kv_caches, funcs):
                row = state[0]
                esz = state.element_size()
                conv = _q38_is_conv_func(func)
                if conv:
                    axis = _row_state_axis(row)
                    inner = int(row.stride()[axis])
                    assert inner * int(row.shape[axis]) == row.numel(), (
                        "conv state row is not compact along the state axis: "
                        f"shape={tuple(row.shape)} stride={tuple(row.stride())}"
                    )
                else:
                    assert getattr(func, "__name__", "") == "get_temporal_copy_spec", (
                        f"unsupported mamba copy func {func!r} on {layer_name}; "
                        "device-side plan only models conv/temporal"
                    )
                    inner = 0
                nbytes = int(row.numel()) * esz
                base.append(int(state.data_ptr()))
                blk_bytes.append(int(state.stride(0)) * esz)
                row_bytes.append(nbytes)
                inner_bytes.append(inner * esz)
                col_shift.append(0 if conv else 1)
                is_conv.append(conv)
                scratch_off.append(stride if conv else 0)
                if conv:
                    stride += nbytes
        slices.append((gid, start, len(base)))

    def _t(v, dtype=torch.int64):
        return torch.tensor(v, dtype=dtype, device=device)

    scratch = None
    if _Q38_PLAN_OVL_SCRATCH and stride:
        scratch = torch.empty(max_num_reqs * stride, dtype=torch.uint8, device=device)
    print(
        "[Q38-PLAN] entries=%d conv=%d scratch=%.1fMiB stride=%dB groups=%s"
        % (
            len(base),
            sum(is_conv),
            0.0 if scratch is None else scratch.numel() / 1048576.0,
            stride,
            slices,
        ),
        flush=True,
    )
    return _Q38PlanDesc(
        base=_t(base),
        blk_bytes=_t(blk_bytes),
        row_bytes=_t(row_bytes),
        inner_bytes=_t(inner_bytes),
        col_shift=_t(col_shift),
        is_conv=_t(is_conv, torch.bool),
        scratch_off=_t(scratch_off),
        scratch=scratch,
        scratch_stride=stride,
        group_slices=slices,
        entries=len(base),
    )


def _q38_align_plan_gpu(
    *,
    ctx: "MambaSpecDecodeCPUContext",
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: GPUInputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: Any,
) -> None:
    accepted = num_accepted_tokens_gpu[:num_reqs]
    if not ctx.staged or num_reqs == 0:
        # dummy / warmup 前向没 stage 过元数据：只要保证 CPU 副本最终拿到本步的
        # 接受数即可（下一步 event.synchronize() 之后才有人读）。
        if num_reqs:
            num_accepted_tokens_cpu_tensor[:num_reqs].copy_(accepted, non_blocking=True)
        return
    copy_bufs = ctx.copy_bufs
    block_size = int(copy_bufs.mamba_spec.block_size)
    desc = ctx.plan_desc
    if desc is None:
        desc = ctx.plan_desc = _q38_build_plan_desc(
            int(ctx.running.gpu.shape[0]),
            kv_cache_config,
            forward_context,
            copy_bufs.mamba_group_ids,
            mamba_state_copy_funcs,
            accepted.device,
        )
    r = min(num_reqs, ctx.num_reqs)
    e = desc.entries
    acc = accepted[:r].to(torch.int64)
    running = ctx.running.gpu[:r]
    src_idx = ctx.state_idx_gpu.gpu[:r]
    valid = ctx.valid.gpu[:r] != 0

    # 与原 CPU 循环逐行对应的判定：本步是否有 block 从 partial 变 full。
    aligned = (
        torch.div(running + acc - 1, block_size, rounding_mode="floor") * block_size
    )
    bias = (aligned - running).clamp_(min=0)
    dest_idx = torch.div(aligned, block_size, rounding_mode="floor") - 1
    cross = valid & (aligned >= running) & (dest_idx >= 0)
    same = src_idx == dest_idx
    # collect_mamba_copy_meta 开头的 early-return：源和目标同一个 block 且不需要
    # 平移时什么都不用搬（但 accepted 仍要改写成 1，见下面的 fix）。
    need = cross & ~(same & (bias == 0))

    src_col = src_idx.unsqueeze(1) + bias.unsqueeze(1) * desc.col_shift.unsqueeze(0)
    dst_col = dest_idx.unsqueeze(1).expand(r, e)
    sizes = torch.where(
        need.unsqueeze(1),
        desc.row_bytes.unsqueeze(0) - bias.unsqueeze(1) * desc.inner_bytes.unsqueeze(0),
        torch.zeros((), dtype=torch.int64, device=acc.device),
    ).clamp_(min=0)
    # __Q38_PLAN_V2__: 每 group 一段循环会带来 O(group 数) 次微 op 发射（本模型 37
    # 个 mamba group，实测 ~11 ms/step，比省掉的那次阻塞 D2H 还贵）。这里把所有
    # group 的 block table cat 成一张 (r, G*ncol)，列号加 group 列偏移，一次 gather
    # 出全部条目的 src/dst block id。cat 是 1 次发射，整步发射降到 ~20 次。
    tables = [
        input_batch.block_table[gid].block_table.gpu for gid, _, _ in desc.group_slices
    ]
    ncol = int(tables[0].shape[1])
    for t in tables[1:]:
        assert int(t.shape[1]) == ncol, (
            "mamba groups have different max_num_blocks_per_req: "
            f"{[int(x.shape[1]) for x in tables]}"
        )
    if desc.col_off is None or desc.ncol != ncol:
        off = [0] * e
        for pos, (_, a, b) in enumerate(desc.group_slices):
            for i in range(a, b):
                off[i] = pos * ncol
        desc.col_off = torch.tensor(off, dtype=torch.int64, device=acc.device)
        desc.ncol = ncol
    bt = torch.cat([t[:r] for t in tables], dim=1)
    sc = src_col.clamp_(0, ncol - 1) + desc.col_off.unsqueeze(0)
    dc = dst_col.clamp(0, ncol - 1) + desc.col_off.unsqueeze(0)
    src_blk = bt.gather(1, sc).to(torch.int64)
    dst_blk = bt.gather(1, dc).to(torch.int64)
    src_ptrs = (
        desc.base.unsqueeze(0)
        + src_blk * desc.blk_bytes.unsqueeze(0)
        + bias.unsqueeze(1) * desc.inner_bytes.unsqueeze(0)
    )
    dst_ptrs = desc.base.unsqueeze(0) + dst_blk * desc.blk_bytes.unsqueeze(0)
    same_blk = src_blk == dst_blk

    if desc.scratch is not None:
        # 同块 conv 平移：src 与 dst 在同一行内重叠，拆成 src -> scratch -> dst。
        ovl = same_blk & desc.is_conv.unsqueeze(0) & (bias.unsqueeze(1) > 0)
        scratch_ptrs = (
            int(desc.scratch.data_ptr())
            + torch.arange(r, dtype=torch.int64, device=acc.device).unsqueeze(1)
            * desc.scratch_stride
            + desc.scratch_off.unsqueeze(0)
        )
        sz_ovl = torch.where(
            ovl, sizes, torch.zeros((), dtype=torch.int64, device=acc.device)
        )
        sz_direct = torch.where(
            ovl, torch.zeros((), dtype=torch.int64, device=acc.device), sizes
        )
        batch_memcpy(src_ptrs.reshape(-1), scratch_ptrs.reshape(-1), sz_ovl.reshape(-1))
        batch_memcpy(src_ptrs.reshape(-1), dst_ptrs.reshape(-1), sz_direct.reshape(-1))
        batch_memcpy(scratch_ptrs.reshape(-1), dst_ptrs.reshape(-1), sz_ovl.reshape(-1))
    else:
        batch_memcpy(src_ptrs.reshape(-1), dst_ptrs.reshape(-1), sizes.reshape(-1))

    # 源块就是目标块时，下一步的 preprocess_mamba 必须按 init_token_idx = 0 读，
    # 所以把 accepted 改写成 1 —— 原路径写 num_accepted_tokens_cpu[i]，这里在设备
    # 上写，再一次 non_blocking D2H。
    out = ctx.accepted_out.gpu
    out[:num_reqs].copy_(accepted)
    out[:r] = torch.where(cross & same, torch.ones_like(out[:r]), out[:r])
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(out[:num_reqs], non_blocking=True)
    _Q38_PLAN_STAT["steps"] += 1
    if _Q38_PLAN_VERIFY:
        _q38_plan_verify(
            ctx,
            num_reqs,
            r,
            accepted,
            kv_cache_config,
            forward_context,
            mamba_state_copy_funcs,
            src_ptrs,
            dst_ptrs,
            sizes,
            cross & same,
        )
    ctx.staged = False


def _q38_plan_verify(
    ctx,
    num_reqs,
    r,
    accepted,
    kv_cache_config,
    forward_context,
    mamba_state_copy_funcs,
    src_ptrs,
    dst_ptrs,
    sizes,
    fix_mask,
) -> None:
    """把设备侧计划与原 CPU 逻辑算出的描述符逐条对比。会同步，只在调试时开。"""
    copy_bufs = ctx.copy_bufs
    block_size = int(copy_bufs.mamba_spec.block_size)
    raw = accepted[:r].tolist()
    flat_sz = sizes.reshape(-1)
    keep = flat_sz > 0
    got = sorted(
        zip(
            src_ptrs.reshape(-1)[keep].tolist(),
            dst_ptrs.reshape(-1)[keep].tolist(),
            flat_sz[keep].tolist(),
        )
    )
    got_fix = fix_mask.nonzero().flatten().tolist()

    copy_bufs.offset = 0
    exp_fix, err = [], None
    try:
        for i in range(r):
            req_state = ctx.req_states[i]
            if req_state is None:
                continue
            running = int(
                ctx.num_computed_tokens[i]
                + ctx.num_scheduled_tokens[i]
                - ctx.num_draft_tokens[i]
            )
            aligned = (running + int(raw[i]) - 1) // block_size * block_size
            if aligned < running:
                continue
            src_i = int(ctx.state_idx[i])
            dst_i = aligned // block_size - 1
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                copy_bufs.mamba_group_ids,
                src_i,
                dst_i,
                aligned - running,
                req_state,
                forward_context,
            )
            if src_i == dst_i:
                exp_fix.append(i)
    except Exception as ex:  # noqa: BLE001
        err = repr(ex)
    n = copy_bufs.offset
    exp = sorted(
        zip(
            copy_bufs.src_ptrs.np[:n].tolist(),
            copy_bufs.dst_ptrs.np[:n].tolist(),
            copy_bufs.sizes.np[:n].tolist(),
        )
    )
    copy_bufs.offset = 0
    del _Q38_ENTRIES[:]
    del _Q38_SAFE[:]

    _Q38_PLAN_STAT["cmp"] += 1
    ok = err is None and exp == got and exp_fix == got_fix
    if not ok:
        _Q38_PLAN_STAT["bad"] += 1
        print(
            "[Q38-PLAN] MISMATCH step=%d reqs=%d/%d err=%s\n"
            "  host n=%d gpu n=%d\n  host_only=%s\n  gpu_only=%s\n"
            "  fix host=%s gpu=%s"
            % (
                _Q38_PLAN_STAT["steps"],
                r,
                num_reqs,
                err,
                len(exp),
                len(got),
                [x for x in exp if x not in got][:4],
                [x for x in got if x not in exp][:4],
                exp_fix,
                got_fix,
            ),
            flush=True,
        )
    elif n or got:
        print(
            "[Q38-PLAN] match step=%d entries=%d fix=%s stat=%s"
            % (_Q38_PLAN_STAT["steps"], n, exp_fix, _Q38_PLAN_STAT),
            flush=True,
        )


def postprocess_mamba_align_gpu(
    *,
    bufs: MambaBuffers,
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: GPUInputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
) -> None:
    # 本步如果有 block 从 partial 变成 full，就把 running state 搬到那个 full
    # block（顺带按接受的 token 数平移 conv 滑窗）。上游是 fused kernel，这里是
    # 同步一次 + CPU 循环，见文件末尾 __Q38_ALIGN_SPEC__ 说明。
    ctx = bufs.postprocess_align
    assert ctx is not None, "postprocess_align context missing"
    if _Q38_GPUPLAN and ctx.running is not None:
        # __Q38_ALIGN_GPUPLAN__: 计划在设备上算，host 不回读。Q38_ALIGN_GPU_PLAN=0
        # 切回下面的原路径（阻塞 D2H + CPU 循环）。
        _q38_align_plan_gpu(
            ctx=ctx,
            num_reqs=num_reqs,
            num_accepted_tokens_gpu=num_accepted_tokens_gpu,
            num_accepted_tokens_cpu_tensor=num_accepted_tokens_cpu_tensor,
            input_batch=input_batch,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context,
            mamba_state_copy_funcs=mamba_state_copy_funcs,
        )
        return
    # .cpu() 是阻塞 D2H，之后 CPU 侧就能直接读到本步真实的接受 token 数。
    accepted = num_accepted_tokens_gpu[:num_reqs].cpu()
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(accepted)
    if not ctx.staged:
        # dummy / warmup 前向没有 stage 过元数据，只需保证上面的同步语义。
        return
    num_accepted = accepted.tolist()
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    copy_bufs = ctx.copy_bufs
    mamba_group_ids = copy_bufs.mamba_group_ids
    block_size = copy_bufs.mamba_spec.block_size
    _q38_log_state_layout_once(kv_cache_config, forward_context, mamba_group_ids)
    copy_bufs.offset = 0
    for i in range(min(num_reqs, ctx.num_reqs)):
        req_state = ctx.req_states[i]
        if req_state is None:
            continue
        num_tokens_running_state = int(
            ctx.num_computed_tokens[i]
            + ctx.num_scheduled_tokens[i]
            - ctx.num_draft_tokens[i]
        )
        new_num_computed_tokens = num_tokens_running_state + int(num_accepted[i]) - 1
        aligned_new_computed_tokens = new_num_computed_tokens // block_size * block_size
        if aligned_new_computed_tokens < num_tokens_running_state:
            continue
        accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
        src_block_idx = int(ctx.state_idx[i])
        dest_block_idx = aligned_new_computed_tokens // block_size - 1
        collect_mamba_copy_meta(
            copy_bufs,
            kv_cache_config,
            mamba_state_copy_funcs,
            mamba_group_ids,
            src_block_idx,
            dest_block_idx,
            accept_token_bias,
            req_state,
            forward_context,
        )
        if src_block_idx == dest_block_idx:
            num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)
    ctx.staged = False


# --- pass-through for the upstream names not reimplemented above -------------
#
# This module is aliased into ``sys.modules`` under the upstream name, so a
# plain ``import vllm.v1.worker.mamba_utils`` here would hand back this module.
# Load the real one from its file instead, under a private name, and resolve
# anything missing against it. PEP 562 means ``from vllm.v1.worker.mamba_utils
# import MambaSpecDecodeGPUContext`` goes through this hook too, which is what
# keeps ``gpu/model_states/mamba_hybrid.py`` importable.
#
# Loading it lazily matters: the module defines Triton kernels at import time
# and nothing on the Kunlun V1 path needs them, so the cost is only paid if
# something actually asks for an upstream-only name.
_UPSTREAM_MODULE: Any = None


def _upstream_module():
    """Import the genuine ``vllm.v1.worker.mamba_utils`` from its file."""
    global _UPSTREAM_MODULE
    if _UPSTREAM_MODULE is None:
        import importlib.util
        import os.path

        import vllm.v1.worker as _worker_pkg

        path = os.path.join(os.path.dirname(_worker_pkg.__file__), "mamba_utils.py")
        spec = importlib.util.spec_from_file_location(
            "vllm_kunlun._upstream_v1_worker_mamba_utils", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _UPSTREAM_MODULE = module
    return _UPSTREAM_MODULE


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_upstream_module(), name)
