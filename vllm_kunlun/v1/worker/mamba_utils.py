# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun replacement for ``vllm.v1.worker.mamba_utils``.

Exactly two things differ from upstream on Kunlun XPU:

* ``batch_memcpy`` must go through ``torch.ops.xspeedgate_ops.batch_memcpy``
  instead of launching the Triton ``batch_memcpy_kernel``.
* ``MambaCopyBuffers.create`` allocates ``int64`` pointer/size buffers, which is
  what the xspeedgate op expects (upstream uses ``uint64``/``int32``).

Everything else -- the 5 Triton kernels, ``MambaSpecDecodeGPUContext``,
``MambaBuffers``, and the V1 pre/postprocess helpers -- is reused verbatim from
the genuine upstream module via ``load_upstream`` + ``reexport``.

That is safe because ``@triton.jit`` is lazy: decorating a kernel compiles
nothing, only a ``kernel[grid](...)`` launch does. The kernels we do not
replace are reachable only from the mamba "align" cache mode, which requires
prefix caching to be enabled.

This replaces a 396-line fork of an *older* upstream revision that was missing
12 symbols the current upstream imports. Two of them broke Qwen3.5 outright::

    vllm/v1/worker/gpu/model_states/mamba_hybrid.py:27
    ImportError: cannot import name 'MambaSpecDecodeGPUContext'

and four more were latent ``AttributeError``s on the V1 path
(``gpu_model_runner.py`` lines 1547, 1570, 2098, 4258). Deriving the export
surface from upstream instead of hand-maintaining it removes that whole class
of failure.

Also dropped here: ``get_hybrid_attention_mamba_layout`` and
``postprocess_mamba``, two symbols the old fork carried that exist neither
upstream nor in any caller.
"""

import logging

import torch

from vllm_kunlun.v1.worker.gpu._upstream import load_upstream, reexport

logger = logging.getLogger("vllm_kunlun")

_up = load_upstream("vllm.v1.worker.mamba_utils")

# Must come before our own definitions below so they win in this namespace.
reexport(_up, globals())


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    """xspeedgate stand-in for upstream's Triton ``batch_memcpy_kernel``.

    ``xspeedgate_ops.batch_memcpy`` wants int64 pointer tensors. Buffers built
    by our ``MambaCopyBuffers.create`` below are already int64; the ``view``
    calls only matter if some other caller hands us upstream-dtype buffers.
    ``view`` is a metadata-only reinterpret, so it is free and bit-exact.
    """
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    if batch == 0:
        return
    if src_ptrs.dtype is not torch.int64:
        src_ptrs = src_ptrs.view(torch.int64)
    if dst_ptrs.dtype is not torch.int64:
        dst_ptrs = dst_ptrs.view(torch.int64)
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


# ``do_mamba_copy_block`` and friends resolve ``batch_memcpy`` from the
# *upstream* module globals, so the override has to land on ``_up`` too -- not
# just in this module's namespace.
if not getattr(_up, "_kunlun_v2_patched", False):
    _up.batch_memcpy = batch_memcpy
    _up.MambaCopyBuffers.create = classmethod(_mamba_copy_buffers_create)
    MambaCopyBuffers = _up.MambaCopyBuffers
    _up._kunlun_v2_patched = True
    logger.info(
        "[KunlunPlugin] mamba_utils patched (xspeedgate batch_memcpy, int64 "
        "copy buffers; %d upstream symbols re-exported)",
        len([k for k in vars(_up) if not k.startswith("__")]),
    )
