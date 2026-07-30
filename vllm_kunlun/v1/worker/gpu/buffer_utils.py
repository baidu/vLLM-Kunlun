# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun torch-native replacement for ``vllm.v1.worker.gpu.buffer_utils``.

The genuine upstream module is executed via ``load_upstream`` so that all its
pure-Python machinery (``UvaBufferPool``, ``UvaBackedTensor``, dataclasses,
constants) is reused verbatim. Only the two Triton kernel launch sites are
overridden with torch-native equivalents:

* ``StagedWriteTensor.apply_write`` — applies staged row/segment writes to a
  device tensor. Replaces ``_apply_write_kernel``.
* ``FusedStagedWriter.apply`` — upstream fuses writes across several tensors
  through raw pointers. Kunlun never calls it because the Kunlun
  ``BlockTables.apply_staged_writes`` override loops ``apply_write`` per group
  instead (raw-pointer fan-out is not expressible in torch-native). It is
  overridden here to raise, so an accidental caller fails loudly rather than
  launching an uncompilable Triton kernel.

UVA handling: upstream ``UvaBuffer`` hard-raises when ``is_uva_available()`` is
False. Kunlun XPU presents as CUDA (``torch_xmlir``); when UVA is available the
upstream classes are used unchanged. When it is not, a device-tensor fallback
is installed so the ``.uva`` views become real device tensors kept in sync via
explicit H2D copies.
"""

import logging

import torch
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from vllm_kunlun.v1.worker.gpu._upstream import load_upstream

logger = logging.getLogger("vllm_kunlun")

_up = load_upstream("vllm.v1.worker.gpu.buffer_utils")

# Re-export the upstream public surface so consumers importing from this
# (swapped) module name resolve the same symbols.
async_copy_to_gpu = _up.async_copy_to_gpu
set_default_max_concurrency = _up.set_default_max_concurrency
UvaBuffer = _up.UvaBuffer
UvaBufferPool = _up.UvaBufferPool
UvaBackedTensor = _up.UvaBackedTensor
StagedWriteTensor = _up.StagedWriteTensor
FusedStagedWriter = _up.FusedStagedWriter
# Imported (but never called) by the genuine block_table module; keep the name
# so its ``from ...buffer_utils import _load_ptr`` succeeds.
_load_ptr = getattr(_up, "_load_ptr", None)


def _apply_write(self) -> None:
    """torch-native replacement of ``StagedWriteTensor.apply_write``.

    For each staged write ``i`` the upstream Triton kernel writes
    ``contents[cu_start:cu_end]`` into the flat device buffer at offset
    ``indices[i] * gpu.stride(0) + starts[i]``. This reproduces it with slice
    assignments on ``self.gpu`` (contiguous, so a flattened view + linear
    offset matches the kernel's pointer arithmetic exactly).
    """
    n = len(self._staged_write_indices)
    if n == 0:
        return

    flat = self.gpu.view(-1)
    stride0 = self.gpu.stride(0)
    contents = torch.tensor(
        self._staged_write_contents, dtype=self.dtype, device=self.device
    )

    cu_start = 0
    for i in range(n):
        cu_end = self._staged_write_cu_lens[i]
        length = cu_end - cu_start
        if length > 0:
            base = (
                self._staged_write_indices[i] * stride0 + self._staged_write_starts[i]
            )
            flat[base : base + length] = contents[cu_start:cu_end]
        cu_start = cu_end

    self.clear_staged_writes()


def _fused_apply(self, tensors, output_ptrs, output_strides) -> None:
    raise NotImplementedError(
        "FusedStagedWriter.apply is not supported on Kunlun XPU; "
        "BlockTables.apply_staged_writes loops per-group apply_write instead."
    )


StagedWriteTensor.apply_write = _apply_write
FusedStagedWriter.apply = _fused_apply


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
    if not is_uva_available():
        return False
    try:
        probe = torch.zeros(1, dtype=torch.int32, device="cpu", pin_memory=True)
        get_accelerator_view_from_cpu_tensor(probe)
    except Exception as e:  # noqa: BLE001 - any failure means "unsupported"
        logger.info("[KunlunPlugin] UVA view probe failed (%s)", e)
        return False
    return True


if not _uva_view_supported():
    # Device-tensor fallback: keep a plain CPU source of truth and a real
    # device tensor as the ``.uva`` view, synced with explicit H2D copies.
    logger.warning(
        "[KunlunPlugin] UVA unavailable; using device-tensor fallback for "
        "Model Runner V2 buffers (extra H2D copies per step)."
    )

    def _uvabuffer_init(self, size, dtype):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        dev = torch.device("cuda", torch.cuda.current_device())
        self.uva = torch.zeros(size, dtype=dtype, device=dev)

    def _pool_copy_to_uva(self, x):
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        buf.uva[:n].copy_(buf.cpu[:n], non_blocking=True)
        return buf.uva[:n]

    UvaBuffer.__init__ = _uvabuffer_init
    UvaBufferPool.copy_to_uva = _pool_copy_to_uva
