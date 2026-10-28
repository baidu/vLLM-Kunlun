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

import torch
import vllm.v1.worker.mamba_utils as _up

logger = logging.getLogger("vllm_kunlun")


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
