# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun Mamba state-copy, align processing and initialization helpers.

The upstream state-management classes and scheduling remain in use. Their
copy and align launch sites call XSpeedGate kernels through this module.
MambaStateZeroer reuses a bounded zero buffer to clear selected state rows
without allocating a tensor the size of the cache pool.
"""

import logging

import numpy as np
import torch
import vllm.v1.worker.mamba_utils as _up

logger = logging.getLogger("vllm_kunlun")


class _TorchKernel:
    """Adapt a Python function to Triton's ``kernel[grid](...)`` call shape."""

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", type(fn).__name__)

    def __getitem__(self, _grid):
        return self.fn


def _preprocess_mamba_align(
    idx_mapping,
    state_idx,
    num_computed_tokens,
    query_start_loc,
    num_accepted_tokens,
    src_col,
    src_off,
    num_reqs,
    **kwargs,
):
    if num_reqs == 0:
        return
    torch.ops.xspeedgate_ops.preprocess_mamba_align_fused_kernel(
        idx_mapping,
        state_idx,
        num_computed_tokens,
        query_start_loc,
        num_accepted_tokens,
        src_col,
        src_off,
        int(num_reqs),
        int(kwargs["MAMBA_BLOCK_SIZE"]),
    )


def _initialize_from_forward_context(original):
    def initialize(self, *args, **kwargs):
        initialized = self.is_initialized
        original(self, *args, **kwargs)
        if initialized and hasattr(self, "_kunlun_align_buffers"):
            return
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        elem_sizes = self.state_elem_sizes.long()
        rows = self.state_dim_row_count
        row_strides = self.state_dim_row_stride
        if not is_conv_state_dim_first():
            # SD 描述为一行，每个逻辑元素包含一个 token 的全部通道。
            is_conv = self.state_conv_widths > 0
            elem_sizes = torch.where(
                is_conv, self.state_inner_sizes * elem_sizes, elem_sizes
            )
            rows = torch.where(is_conv, 1, rows)
            row_strides = torch.where(
                is_conv, self.state_conv_widths.long() * elem_sizes, row_strides
            )
        self._kunlun_precopy_metadata = elem_sizes, rows, row_strides
        self._kunlun_postprocess_metadata = elem_sizes.int(), rows, row_strides
        capacity = self.num_accepted_tokens_out.numel()
        device = self.num_accepted_tokens_out.device
        self._kunlun_align_rows = torch.arange(
            capacity, dtype=torch.int32, device=device
        )
        self._kunlun_align_buffers = torch.empty(
            (4, capacity), dtype=torch.int32, device=device
        )
        # padding 请求写入额外槽位，避免覆盖有效 slot 0。
        self._kunlun_align_accepted_staging = torch.empty(
            capacity + 1, dtype=torch.int32, device=device
        )

    return initialize


def _run_fused_precopy(
    self, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping
):
    if num_reqs == 0 or not self.is_initialized:
        return
    elem_sizes, rows, row_strides = self._kunlun_precopy_metadata
    torch.ops.xspeedgate_ops.precopy_mamba_align_fused_kernel(
        state_idx_gpu,
        src_col_gpu,
        token_bias_gpu,
        self.block_table_ptrs,
        self.block_table_stride_req,
        self.state_base_addrs,
        self.state_block_strides,
        elem_sizes,
        self.state_inner_sizes,
        self.state_conv_widths,
        self.state_group_indices,
        rows,
        row_strides,
        idx_mapping[:num_reqs],
        int(num_reqs),
        1024,
        True,
        1,
    )


def _run_fused_postprocess_align(
    self,
    num_reqs,
    num_accepted_tokens_gpu,
    state_idx_gpu,
    new_num_computed_tokens_gpu,
    idx_mapping,
):
    if num_reqs == 0 or not self.is_initialized or num_accepted_tokens_gpu.numel() == 0:
        return
    mapping = idx_mapping[:num_reqs]
    valid = mapping >= 0
    slots = mapping.clamp_min(0).long()
    accepted, state, computed, accepted_out = self._kunlun_align_buffers[:, :num_reqs]
    # 旧接口按 batch 顺序读取决策数组。
    torch.index_select(num_accepted_tokens_gpu, 0, slots, out=accepted)
    torch.index_select(state_idx_gpu, 0, slots, out=state)
    torch.index_select(new_num_computed_tokens_gpu, 0, slots, out=computed)
    accepted_out.copy_(accepted)
    batch_mapping = torch.where(valid, self._kunlun_align_rows[:num_reqs], -1)
    elem_sizes, rows, row_strides = self._kunlun_postprocess_metadata
    torch.ops.xspeedgate_ops.postprocess_mamba(
        accepted,
        state,
        accepted,
        computed,
        accepted,
        self.block_table_ptrs,
        self.block_table_stride_req,
        self.state_base_addrs,
        self.state_block_strides,
        elem_sizes,
        self.state_inner_sizes,
        self.state_conv_widths,
        self.state_group_indices,
        rows,
        row_strides,
        accepted_out,
        batch_mapping,
        int(num_reqs),
        int(self.block_size),
        True,
        True,
    )
    capacity = num_accepted_tokens_gpu.numel()
    staging = self._kunlun_align_accepted_staging
    staging[:capacity].copy_(num_accepted_tokens_gpu)
    write_slots = torch.where(valid, mapping, capacity).long()
    staging.index_copy_(0, write_slots, accepted_out)
    num_accepted_tokens_gpu.copy_(staging[:capacity])


def _patch_mamba_align_xspeedgate() -> None:
    _up.preprocess_mamba_align_fused_kernel = _TorchKernel(_preprocess_mamba_align)
    context = _up.MambaSpecDecodeGPUContext
    if not getattr(context, "_kunlun_align_xspeedgate_patched", False):
        context.initialize_from_forward_context = _initialize_from_forward_context(
            context.initialize_from_forward_context
        )
        context.run_fused_precopy = _run_fused_precopy
        context.run_fused_postprocess_align = _run_fused_postprocess_align
        context._kunlun_align_xspeedgate_patched = True


_patch_mamba_align_xspeedgate()


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    """xspeedgate stand-in for upstream's Triton ``batch_memcpy_kernel``.

    ``xspeedgate_ops.batch_memcpy`` is specified for int64 pointer and size
    tensors, and every buffer that reaches it comes from the
    ``MambaCopyBuffers.create`` override below, which allocates exactly that:
    the op has a single call path (upstream ``preprocess_mamba`` ->
    ``do_mamba_copy_block``), and ``MambaCopyBuffers`` has a single construction
    site (upstream ``MambaBuffers.create``), which goes through the override.

    The dtypes are therefore asserted rather than coerced. An earlier version
    reinterpreted mismatches with ``Tensor.view``, which is only lossless
    between same-itemsize dtypes -- an int32 buffer would have been silently
    re-read as half as many int64 values. Nothing produces such a buffer today,
    so a mismatch means the call path changed and should fail loudly.
    """
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    if batch == 0:
        return
    for name, tensor in (
        ("src_ptrs", src_ptrs),
        ("dst_ptrs", dst_ptrs),
        ("sizes", sizes),
    ):
        assert tensor.dtype is torch.int64, (
            f"xspeedgate_ops.batch_memcpy expects int64 {name}, got "
            f"{tensor.dtype}; buffers should come from the Kunlun "
            f"MambaCopyBuffers.create override in {__name__}"
        )
    torch.ops.xspeedgate_ops.batch_memcpy(src_ptrs, dst_ptrs, sizes)


def _mamba_copy_buffers_create(
    cls,
    max_num_reqs,
    kv_cache_config,
    copy_funcs,
    make_buffer,
):
    """Same as upstream ``MambaCopyBuffers.create`` but with int64 buffers.

    Upstream allocates ``uint64`` pointers and ``int32`` sizes
    (mamba_utils.py:449-451); the xspeedgate op is specified for int64.

    Note the ``v0.15.0-dev`` branch keeps ``int32`` sizes here (#351), so the two
    release branches disagree on what ``xspeedgate_ops.batch_memcpy`` wants. The
    int64 choice predates this change -- it is what the v0.25.1 branch has
    shipped since #392 -- and is kept as-is; the assertion in ``batch_memcpy``
    above turns any future mismatch into a loud failure rather than a silent
    reinterpret.
    """
    mamba_group_ids, mamba_spec = _up.get_mamba_groups(kv_cache_config)
    entries_per_req = sum(
        len(kv_cache_config.kv_cache_groups[gid].layer_names) for gid in mamba_group_ids
    ) * len(copy_funcs)
    n = max_num_reqs * entries_per_req
    return cls(
        src_ptrs=make_buffer(n, dtype=torch.int64),
        dst_ptrs=make_buffer(n, dtype=torch.int64),
        sizes=make_buffer(n, dtype=torch.int64),
        mamba_group_ids=mamba_group_ids,
        mamba_spec=mamba_spec,
    )


# ``do_mamba_copy_block`` and friends resolve ``batch_memcpy`` from the upstream
# module globals, and gpu_model_runner.py:202 imports the module object rather
# than the name, so both call sites pick these up at call time.
_up.batch_memcpy = batch_memcpy
_up.MambaCopyBuffers.create = classmethod(_mamba_copy_buffers_create)
logger.info(
    "[KunlunPlugin] mamba_utils patched (xspeedgate batch_memcpy, int64 buffers)"
)
logger.info(
    "[KunlunPlugin] mamba align patched (complete XSpeedGate wheel)",
)


class MambaStateZeroer:
    """Reuse one zero row as the source of bounded, strided state writes.

    Called outside graph capture on the runner's current stream. Keep the zero
    source alive for the runner lifetime: batch_memcpy consumes raw pointers.
    """

    def __init__(self):
        self._zero = None

    def clear(self, state_blocks):
        plans = {}
        device = None
        max_bytes = 0
        # Validate the complete plan before modifying any state.
        for state, block_ids in state_blocks:
            blocks = sorted(set(block_ids) - {0})
            if not blocks:
                continue
            if state.ndim < 2:
                raise ValueError("Mamba state must have a block dimension")
            inner = 1
            for dim in range(state.ndim - 1, 0, -1):
                if state.stride(dim) != inner:
                    raise ValueError("Mamba state rows must be contiguous")
                inner *= state.shape[dim]
            if state.stride(0) < inner:
                raise ValueError("Mamba state rows must not overlap")
            if blocks[0] < 0 or blocks[-1] >= state.shape[0]:
                raise ValueError("Mamba state block index out of bounds")
            if device is not None and device != state.device:
                raise ValueError("Mamba states must be on the same device")
            device = state.device
            size = inner * state.element_size()
            if not size:
                continue
            max_bytes = max(max_bytes, size)
            stride = state.stride(0) * state.element_size()
            key = (state.data_ptr(), stride, size)
            plans.setdefault(key, set()).update(blocks)
        if not plans:
            return
        if device.type != "cuda":
            raise ValueError("MambaStateZeroer requires the Kunlun CUDA shim")

        # Conv and SSM commonly occupy adjacent intervals in the same page.
        # Combine only exact adjacency with identical selected rows; preserve
        # all padding and any state belonging to another request.
        merged = []
        for (address, stride, size), blocks in sorted(
            plans.items(), key=lambda item: (item[0][1], item[0][0])
        ):
            if merged:
                prev_address, prev_stride, prev_size, prev_blocks = merged[-1]
                if (
                    stride == prev_stride
                    and address == prev_address + prev_size
                    and blocks == prev_blocks
                    and prev_size + size <= stride
                ):
                    merged[-1] = (prev_address, stride, prev_size + size, blocks)
                    max_bytes = max(max_bytes, prev_size + size)
                    continue
            merged.append((address, stride, size, blocks))
        if (
            self._zero is None
            or self._zero.device != device
            or self._zero.numel() < max_bytes
        ):
            self._zero = torch.zeros(max_bytes, dtype=torch.uint8, device=device)
        count = sum(len(blocks) for _, _, _, blocks in merged)
        host = np.empty((3, count), dtype=np.int64)
        host[0].fill(self._zero.data_ptr())
        offset = 0
        for address, stride, size, blocks in merged:
            end = offset + len(blocks)
            host[1, offset:end] = (
                address
                + np.fromiter(sorted(blocks), dtype=np.int64, count=len(blocks))
                * stride
            )
            host[2, offset:end] = size
            offset = end
        metadata = torch.from_numpy(host).to(device=device)
        torch.ops.xspeedgate_ops.batch_memcpy(metadata[0], metadata[1], metadata[2])
