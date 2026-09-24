# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun overrides for ``vllm.v1.worker.mamba_utils``.

Exactly two things differ from upstream on Kunlun XPU:

* ``batch_memcpy`` must go through ``torch.ops.xspeedgate_ops.batch_memcpy``
  instead of launching the Triton ``batch_memcpy_kernel``.
* ``MambaCopyBuffers.create`` allocates ``int64`` pointer/size buffers, which is
  what the xspeedgate op expects (upstream uses ``uint64``/``int32``).

Everything else -- the 5 Triton kernels, ``MambaSpecDecodeGPUContext``,
``MambaBuffers``, and the V1 pre/postprocess helpers -- is left untouched.

That is safe because ``@triton.jit`` is lazy: decorating a kernel compiles
nothing, only a ``kernel[grid](...)`` launch does. The kernels we do not
replace are reachable only from the mamba "align" cache mode, which requires
prefix caching to be enabled.

This replaces a 396-line fork of an *older* upstream revision that was missing
12 symbols the current upstream imports. Two of them broke Qwen3.5 outright::

    vllm/v1/worker/gpu/model_states/mamba_hybrid.py:27
    ImportError: cannot import name 'MambaSpecDecodeGPUContext'

and four more were latent ``AttributeError``s on the V1 path
(``gpu_model_runner.py`` lines 1547, 1570, 2098, 4258). Patching the two real
deltas in place, instead of hand-maintaining a whole export surface, removes
that class of failure entirely.

Also dropped here: ``get_hybrid_attention_mamba_layout`` and
``postprocess_mamba``, two symbols the old fork carried that exist neither
upstream nor in any caller.
"""

import logging
from types import SimpleNamespace

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
    torch.ops.xspeedgate_ops.mamba_align_preprocess(
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
    def initialize(self, kv_cache_config, forward_context, copy_funcs, block_tables):
        original(
            self,
            kv_cache_config,
            forward_context,
            copy_funcs,
            block_tables,
        )
        try:
            from vllm.model_executor.layers.mamba.mamba_utils import (
                is_conv_state_dim_first,
            )

            dim_first = is_conv_state_dim_first()
        except Exception:
            dim_first = False
        metas = []
        for group_local_idx, group_id in enumerate(self.mamba_group_ids):
            group = kv_cache_config.kv_cache_groups[group_id]
            for layer_name in group.layer_names:
                for state_type_idx, state in enumerate(
                    forward_context[layer_name].kv_cache
                ):
                    copy_func = copy_funcs[state_type_idx]
                    metas.append(
                        SimpleNamespace(
                            state=state,
                            group_idx=group_local_idx,
                            is_conv="conv" in getattr(copy_func, "__name__", ""),
                            dim_first=dim_first,
                        )
                    )
        self._kunlun_mamba_metas = metas
        # MRV2 passes a batch slice of stable storage. Keep its full capacity,
        # since later batches can be larger than the first one.
        capacity = self.num_accepted_tokens_out.numel()
        self._kunlun_mamba_block_tables = [
            table.as_strided((capacity, table.shape[1]), table.stride())
            for table in block_tables
        ]
        self._kunlun_mamba_states = [meta.state for meta in metas]
        layouts = []
        for meta in metas:
            state = meta.state
            table = self._kunlun_mamba_block_tables[meta.group_idx]
            if state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError("Unsupported Mamba state dtype")
            inner = 1
            for dim in range(state.ndim - 1, 0, -1):
                if state.stride(dim) != inner:
                    raise ValueError("Mamba state inner dimensions must be contiguous")
                inner *= state.shape[dim]
            if state.stride(0) < inner or (meta.is_conv and state.ndim != 3):
                raise ValueError("Invalid Mamba state layout")
            channels = state.shape[1 if dim_first else 2] if meta.is_conv else 0
            width = state.shape[2 if dim_first else 1] if meta.is_conv else 0
            layouts.append([
                state.data_ptr(), state.stride(0) * state.element_size(),
                inner * state.element_size(), state.element_size(),
                channels, width, int(dim_first), table.data_ptr(),
                table.shape[1], state.shape[0], capacity,
            ])
        self._kunlun_mamba_layouts = torch.tensor(
            layouts, dtype=torch.int64, device=self.num_accepted_tokens_out.device
        )
        self._kunlun_align_dst_col = torch.empty_like(self.num_accepted_tokens_out)
        self._kunlun_align_token_bias = torch.empty_like(self.num_accepted_tokens_out)

    return initialize


def _run_fused_precopy(
    self, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping
):
    if num_reqs == 0 or not self.is_initialized:
        return
    torch.ops.xspeedgate_ops.mamba_align_state_copy_batched(
        self._kunlun_mamba_states,
        self._kunlun_mamba_layouts,
        idx_mapping,
        src_col_gpu,
        state_idx_gpu,
        token_bias_gpu,
        int(num_reqs),
        True,  # The forward reads the in-block accepted offset itself.
    )


def _run_fused_postprocess_align(
    self,
    num_reqs,
    num_accepted_tokens_gpu,
    state_idx_gpu,
    new_num_computed_tokens_gpu,
    idx_mapping,
):
    if num_reqs == 0 or not self.is_initialized:
        return
    dst_col = self._kunlun_align_dst_col
    token_bias = self._kunlun_align_token_bias
    torch.ops.xspeedgate_ops.mamba_align_postprocess(
        num_accepted_tokens_gpu,
        state_idx_gpu,
        new_num_computed_tokens_gpu,
        idx_mapping,
        dst_col,
        token_bias,
        int(num_reqs),
        int(self.block_size),
    )
    torch.ops.xspeedgate_ops.mamba_align_state_copy_batched(
        self._kunlun_mamba_states,
        self._kunlun_mamba_layouts,
        idx_mapping,
        state_idx_gpu,
        dst_col,
        token_bias,
        int(num_reqs),
    )


def _patch_mamba_align_xspeedgate() -> None:
    _up.preprocess_mamba_align_fused_kernel = _TorchKernel(
        _preprocess_mamba_align
    )
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
