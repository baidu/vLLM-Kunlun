# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun overrides for ``vllm.v1.worker.gpu.buffer_utils``.

All of the module's pure-Python machinery (``UvaBufferPool``,
``UvaBackedTensor``, dataclasses, constants) is left alone. Only the two Triton
kernel launch sites are overridden with Kunlun equivalents:

* ``StagedWriteTensor.apply_write`` — applies staged row/segment writes to a
  device tensor. Replaces ``_apply_write_kernel`` with
  ``torch.ops.xspeedgate_ops.apply_write``. The native op consumes the same
  ``(indices, starts, contents, cu_lens)`` layout, but requires a 2-D output;
  scalar-per-request buffers are therefore exposed as ``[N, 1]`` views.
* ``FusedStagedWriter.apply`` — upstream fuses writes across several tensors
  through raw pointers. Kunlun never calls it because the Kunlun
  ``BlockTables.apply_staged_writes`` override loops ``apply_write`` per group
  instead (the native apply_write operator handles one tensor per launch). It is
  overridden here to raise, so an accidental caller fails loudly rather than
  launching an uncompilable Triton kernel.

For long staged value lists, NumPy builds the typed CPU payload faster than
per-scalar torch conversion. Native staged-write metadata uses mapped host
allocations. Other buffers keep the upstream view or explicit-copy fallback
because torch operators do not accept mapped host device pointers.

UVA handling: upstream ``UvaBuffer`` hard-raises when ``is_uva_available()`` is
False. Kunlun XPU presents as CUDA (``torch_xmlir``); when UVA is available the
upstream classes are used unchanged. When it is not, a device-tensor fallback
is installed so the ``.uva`` views become real device tensors kept in sync via
explicit H2D copies. That fallback pins the pool's per-step staging buffers but
deliberately not the multi-GB ``uva_instead_of_gpu`` ones -- see
``_uvabuffer_init`` for why the two roles differ.
"""

import ctypes
import logging
import math
from types import SimpleNamespace

import numpy as np
import torch
import vllm.v1.worker.gpu.buffer_utils as _up
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = logging.getLogger("vllm_kunlun")

_NUMPY_DTYPES = {
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float32: np.float32,
}


def _apply_write(self) -> None:
    """Native-op replacement of ``StagedWriteTensor.apply_write``.

    For each staged write ``p``, copy
    ``contents[cu_start:cu_end]`` into output row ``indices[p]`` starting at
    column ``starts[p]``. Metadata is copied through the configured UVA pool;
    long value lists retain the local NumPy fast path. One-dimensional output
    buffers are viewed as ``[N, 1]`` to match the native ABI without copying.
    """
    n = len(self._staged_write_indices)
    if n == 0:
        return
    indices = self.write_indices.copy_to_uva(self._staged_write_indices)
    starts = self.write_starts.copy_to_uva(self._staged_write_starts)
    cu_lens = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)
    values = self._staged_write_contents
    if len(values) >= 1024 and self.dtype in _NUMPY_DTYPES:
        # Long Python lists are expensive for torch.tensor to unpack. NumPy
        # creates the same typed CPU payload without scalar Torch conversion.
        cpu = torch.from_numpy(np.asarray(values, dtype=_NUMPY_DTYPES[self.dtype]))
        if _PINNED_OK:
            cpu = cpu.pin_memory()
        contents = cpu.to(self.device, non_blocking=_PINNED_OK)
    else:
        contents = _up.async_tensor_h2d(values, device=self.device, dtype=self.dtype)
    # XSpeedGate requires 2D, including scalar-per-request state buffers.
    output = self.gpu.unsqueeze(-1) if self.gpu.ndim == 1 else self.gpu
    torch.ops.xspeedgate_ops.apply_write(output, indices, starts, contents, cu_lens)
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


class _Device(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _Dtype(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _Device),
        ("ndim", ctypes.c_int),
        ("dtype", _Dtype),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("offset", ctypes.c_uint64),
    ]


class _Allocation:
    def __init__(self, nbytes):
        self.lib = ctypes.CDLL("libcudart.so.12")
        self.free = self.lib.cudaFreeHost
        self.free.argtypes = [ctypes.c_void_p]
        self.free.restype = ctypes.c_int
        alloc = self.lib.cudaHostAlloc
        alloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        alloc.restype = ctypes.c_int
        self.ptr = ctypes.c_void_p()
        rc = alloc(ctypes.byref(self.ptr), max(nbytes, 1), 2)
        if rc:
            raise RuntimeError(f"cudaHostAllocMapped failed: {rc}")

    def __del__(self):
        if getattr(self, "ptr", None) and self.ptr.value:
            self.free(self.ptr)


def _mapped_host_buffer(size, dtype):
    shape = (size,) if isinstance(size, int) else tuple(size)
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    allocation = _Allocation(nbytes)
    host = (ctypes.c_byte * max(nbytes, 1)).from_address(allocation.ptr.value)
    # frombuffer retains host, which owns the allocation.
    host._allocation = allocation
    cpu = torch.frombuffer(host, dtype=dtype, count=math.prod(shape)).reshape(shape)
    cpu.zero_()
    get = allocation.lib.cudaHostGetDevicePointer
    get.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    get.restype = ctypes.c_int
    ptr = ctypes.c_void_p()
    rc = get(ctypes.byref(ptr), allocation.ptr, 0)
    if rc:
        raise RuntimeError(f"cudaHostGetDevicePointer failed: {rc}")
    # Keep the original CPU tensor's DLPack deleter and owner. Only the exported
    # view changes address/device; the allocation is freed after both views die.
    capsule = torch.utils.dlpack.to_dlpack(cpu)
    address = ctypes.pythonapi.PyCapsule_GetPointer
    address.argtypes = [ctypes.py_object, ctypes.c_char_p]
    address.restype = ctypes.c_void_p
    tensor = _Tensor.from_address(address(capsule, b"dltensor"))
    tensor.data = ptr.value
    tensor.device = _Device(2, torch.cuda.current_device())
    return cpu, torch.utils.dlpack.from_dlpack(capsule)


class _NativeUvaBufferPool(_up.UvaBufferPool):
    """Mapped metadata for native kernels; torch indexing still needs HBM."""

    def __init__(self, size, dtype, max_concurrency):
        self.size = size
        self.dtype = dtype
        self.max_concurrency = max_concurrency
        self._curr = 0
        self._uva_bufs = []
        for _ in range(max_concurrency):
            cpu, uva = _mapped_host_buffer(size, dtype)
            self._uva_bufs.append(SimpleNamespace(cpu=cpu, np=cpu.numpy(), uva=uva))

    def copy_to_uva(self, x):
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        n = len(x)
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        dst[:n] = x
        return buf.uva[:n]


_mapped_host_available = None


def _native_pool(pool):
    global _mapped_host_available
    if _mapped_host_available is False:
        return pool
    try:
        mapped = _NativeUvaBufferPool(pool.size, pool.dtype, pool.max_concurrency)
        if _mapped_host_available is None:
            logger.info("Mapped host metadata enabled for native MRV2 kernels")
        _mapped_host_available = True
        return mapped
    except (OSError, RuntimeError) as e:
        _mapped_host_available = False
        logger.warning(
            "Mapped host allocation unavailable; retaining H2D buffers: %s", e
        )
        return pool


_original_staged_init = _up.StagedWriteTensor.__init__


def _staged_init(self, *args, **kwargs):
    _original_staged_init(self, *args, **kwargs)
    self.write_indices = _native_pool(self.write_indices)
    self.write_starts = _native_pool(self.write_starts)
    self.write_cu_lens = _native_pool(self.write_cu_lens)


_up.StagedWriteTensor.__init__ = _staged_init
