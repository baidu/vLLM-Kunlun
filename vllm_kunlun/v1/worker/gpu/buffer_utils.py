# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun overrides for ``vllm.v1.worker.gpu.buffer_utils``.

All of the module's pure-Python machinery (``UvaBufferPool``,
``UvaBackedTensor``, dataclasses, constants) is left alone. Only the two Triton
kernel launch sites are overridden:

* ``StagedWriteTensor.apply_write`` — applies staged row/segment writes to a
  device tensor. Replaces ``_apply_write_kernel`` with the Kunlun native op
  ``torch.ops.xspeedgate_ops.apply_write``. The native op takes the same
  ``(indices, starts, contents, cu_lens)`` staging layout the upstream kernel
  does; the only adaptation is that it requires a 2-D ``output``, so 1-D buffers
  (``total_len``, ``num_computed_tokens``) are passed through an ``[N, 1]`` view
  (``stride(0) == 1``), which reproduces the kernel's flat
  ``row * stride(0) + start`` addressing exactly.
* ``FusedStagedWriter.apply`` — upstream fuses writes across several tensors
  through raw pointers. Kunlun never calls it because the Kunlun
  ``BlockTables.apply_staged_writes`` override loops ``apply_write`` per group
  instead (the native ``apply_write`` is single-group only). It is overridden
  here to raise, so an accidental caller fails loudly rather than launching an
  uncompilable Triton kernel.

UVA handling: upstream ``UvaBuffer`` hard-raises when ``is_uva_available()`` is
False. Kunlun XPU presents as CUDA (``torch_xmlir``); when UVA is available the
upstream classes are used unchanged. When it is not, a device-tensor fallback
is installed so the ``.uva`` views become real device tensors kept in sync via
explicit H2D copies. That fallback pins the pool's per-step staging buffers but
deliberately not the multi-GB ``uva_instead_of_gpu`` ones -- see
``_uvabuffer_init`` for why the two roles differ.
"""

import logging

import torch
import vllm.v1.worker.gpu.buffer_utils as _up
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    get_accelerator_view_from_cpu_tensor,
)

logger = logging.getLogger("vllm_kunlun")


def _apply_write(self) -> None:
    """Native-op replacement of ``StagedWriteTensor.apply_write``.

    For each staged write ``p`` the upstream Triton kernel writes
    ``contents[cu_start:cu_end]`` into the device buffer at row
    ``indices[p]``, column ``starts[p]``. The native op takes exactly the same
    staging arrays -- built here the same way upstream ``apply_write`` builds
    them (``copy_to_uva`` for the int32 metadata, ``async_tensor_h2d`` for the
    contents) -- and requires a 2-D ``output``. 1-D buffers are viewed as
    ``[N, 1]`` (contiguous, so ``stride(0) == 1``), which makes the op's
    ``row * stride(0) + start`` arithmetic match the kernel's flat addressing.
    """
    n = len(self._staged_write_indices)
    if n == 0:
        return

    indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
    starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
    cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)
    write_contents = async_tensor_h2d(
        self._staged_write_contents, device=self.device, dtype=self.dtype
    )

    out = self.gpu if self.gpu.dim() == 2 else self.gpu.view(self.gpu.shape[0], -1)
    torch.ops.xspeedgate_ops.apply_write(
        out, indices_uva, starts_uva, write_contents, cu_lens_uva
    )

    self.clear_staged_writes()


def _fused_apply(self, tensors, output_ptrs, output_strides) -> None:
    raise NotImplementedError(
        "FusedStagedWriter.apply is not supported on Kunlun XPU; "
        "BlockTables.apply_staged_writes loops per-group apply_write instead."
    )


_up.StagedWriteTensor.apply_write = _apply_write
_up.FusedStagedWriter.apply = _fused_apply


def _pinned_alloc_supported() -> bool:
    """Probe whether a pinned host allocation actually succeeds here.

    ``is_uva_available()`` answers this via ``is_pin_memory_available()``, which
    is a capability query rather than an allocation, so probe the real thing.
    Kept separate from the view probe below so a failure tells us *which* of the
    two is missing.
    """
    try:
        torch.zeros(1, dtype=torch.int32, device="cpu", pin_memory=True)
    except Exception as e:  # noqa: BLE001 - any failure means "unsupported"
        logger.info("[KunlunPlugin] pinned host allocation unavailable (%s)", e)
        return False
    return True


_PINNED_OK = _pinned_alloc_supported()


def _uva_view_supported() -> bool:
    """Probe whether a real UVA accelerator view can be built on this platform.

    ``is_uva_available()`` only checks for pinned memory, which is True on
    Kunlun XPU. The actual view is created by
    ``get_accelerator_view_from_cpu_tensor``, which dispatches on
    ``current_platform.is_xpu() / is_cuda_alike()`` and then calls a
    ``vllm._C`` custom op. KunlunPlatform matches neither branch (and
    ``vllm._C`` is not built), so the call raises. Probe it once instead of
    trusting ``is_uva_available()``.
    """
    if not is_uva_available() or not _PINNED_OK:
        return False
    try:
        probe = torch.zeros(1, dtype=torch.int32, device="cpu", pin_memory=True)
        get_accelerator_view_from_cpu_tensor(probe)
    except Exception as e:  # noqa: BLE001 - any failure means "unsupported"
        logger.info("[KunlunPlugin] UVA view probe failed (%s)", e)
        return False
    return True


# Set while ``UvaBufferPool`` is constructing its buffers; see the comment on
# ``_uvabuffer_init`` for why the role has to be carried out of band.
_BUILDING_POOL = False


if not _uva_view_supported():
    # Device-tensor fallback: keep a plain CPU source of truth and a real
    # device tensor as the ``.uva`` view, synced with explicit H2D copies.
    logger.warning(
        "[KunlunPlugin] UVA unavailable; using device-tensor fallback for "
        "Model Runner V2 buffers (extra H2D copies per step; staging buffers "
        "%s).",
        "pinned" if _PINNED_OK else "pageable, so H2D copies stay synchronous",
    )

    def _uvabuffer_init(self, size, dtype):
        """Device-tensor stand-in for the upstream UVA-backed buffer.

        Upstream pins unconditionally because its ``.uva`` *is* the pinned host
        memory. Here ``.uva`` is a separate device tensor, so pinning is only
        worth it for one of ``UvaBuffer``'s two upstream roles:

        * ``UvaBufferPool.__init__`` (upstream buffer_utils.py:67) builds the
          per-step H2D staging buffers, sized by ``max_num_reqs`` -- kilobytes.
          Pinning these is what makes the ``non_blocking=True`` copy in
          ``_pool_copy_to_uva`` genuinely asynchronous: a copy out of pageable
          memory has to be staged through a driver buffer before the call
          returns, which blocks the host and makes the pool's round-robin
          pointless.
        * ``StagedWriteTensor.__init__`` with ``uva_instead_of_gpu=True``
          (upstream buffer_utils.py:141) allocates bulk storage that upstream
          documents as "extremely large (e.g., several GBs)" -- ``all_token_ids``
          is ``max_num_reqs x max_model_len``. Pinning that would lock GBs of
          unswappable host memory, and nothing ever reads its ``.cpu`` side
          anyway; only ``.uva`` is used.

        ``UvaBuffer``'s own arguments do not say which role it is being built
        for, so ``_BUILDING_POOL`` carries it. Pools are only ever constructed
        during single-threaded worker init, so a module-level flag is enough.
        """
        pin = _BUILDING_POOL and _PINNED_OK
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=pin)
        self.np = self.cpu.numpy()
        dev = torch.device("cuda", torch.cuda.current_device())
        self.uva = torch.zeros(size, dtype=dtype, device=dev)

    _orig_pool_init = _up.UvaBufferPool.__init__

    def _pool_init(self, *args, **kwargs):
        """Run the upstream pool constructor with the staging role marked.

        Wrapping rather than reimplementing keeps this immune to upstream
        changing what else the pool sets up.
        """
        global _BUILDING_POOL
        _BUILDING_POOL = True
        try:
            _orig_pool_init(self, *args, **kwargs)
        finally:
            _BUILDING_POOL = False

    def _pool_copy_to_uva(self, x):
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        # Safe to issue asynchronously: the round-robin means this buffer is
        # not rewritten for another ``max_concurrency`` steps, and the copy is
        # queued ahead of the forward that consumes it.
        buf.uva[:n].copy_(buf.cpu[:n], non_blocking=True)
        return buf.uva[:n]

    _up.UvaBuffer.__init__ = _uvabuffer_init
    _up.UvaBufferPool.__init__ = _pool_init
    _up.UvaBufferPool.copy_to_uva = _pool_copy_to_uva
