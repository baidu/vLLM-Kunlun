# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific monkey-patches enabling vLLM sleep mode on Kunlun XPU.

Two independent fixes, both required for ``--enable-sleep-mode``:

1. **Offload copies must go through the CUDA driver API.**
   ``CuMemAllocator.sleep()`` / ``wake_up()`` move the pool contents with
   ``libcudart.cudaMemcpy``, i.e. the CUDA *runtime* copy. The device
   addresses involved are VMM virtual addresses (allocated by
   ``cuMemCreate`` + ``cuMemMap`` in the cumem allocator), and the runtime
   copy segfaults on them; the *driver* copies ``cuMemcpyDtoH_v2`` /
   ``cuMemcpyHtoD_v2`` understand them. Instead of reimplementing the two
   methods, we swap the module-global ``libcudart`` for a passthrough shim
   whose ``cudaMemcpy`` is pinned to one direction, for the duration of the
   call. ``sleep()`` only ever copies device->host and ``wake_up()`` only
   host->device, so the direction is known statically and no pointer
   introspection is needed -- which matters because Kunlun runs with
   ``CUDA_FAKE_UVA_ENABLE=1``, where deciding the direction from a pointer's
   address space is not reliable.

2. **The weights pool must be re-mapped before a standalone
   ``reload_weights()``.** After a level-2 sleep the weights VMM pool is
   unmapped; loading into the weight tensors without a preceding
   ``wake_up(tags=["weights"])`` writes to released device memory and fails
   with XPU error -707, taking down the engine. The ``sleep()`` /
   ``wake_up()`` wrappers maintain ``_kunlun_unmapped_tags`` on the
   allocator so the runner patch can detect that state.

Triggering: post-import hooks registered in ``vllm_kunlun/__init__.py``.
Both patches are idempotent under fork() and re-import via the
``_kunlun_sleep_patched`` / ``_kunlun_reload_weights_patched`` flags.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import inspect
import logging
from typing import Any

logger = logging.getLogger("vllm_kunlun")

_DTOH = "DeviceToHost"
_HTOD = "HostToDevice"

# ``libcuda.so.1`` handle with the two driver copy entry points configured.
# Populated by ``_init_driver_copies()`` at patch time.
_libcuda: Any = None


def _init_driver_copies() -> None:
    """Load libcuda and declare the driver copy signatures.

    ``CUdeviceptr`` is a 64-bit unsigned integer, host pointers are plain
    ``void *``.
    """
    global _libcuda
    if _libcuda is not None:
        return

    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuMemcpyDtoH_v2.restype = ctypes.c_int
    lib.cuMemcpyDtoH_v2.argtypes = [
        ctypes.c_void_p,  # dstHost
        ctypes.c_ulonglong,  # srcDevice
        ctypes.c_size_t,  # ByteCount
    ]
    lib.cuMemcpyHtoD_v2.restype = ctypes.c_int
    lib.cuMemcpyHtoD_v2.argtypes = [
        ctypes.c_ulonglong,  # dstDevice
        ctypes.c_void_p,  # srcHost
        ctypes.c_size_t,  # ByteCount
    ]
    _libcuda = lib


class _DirectionPinnedLibcudart:
    """Passthrough proxy over ``CudaRTLibrary`` with a driver-based copy.

    Every attribute except ``cudaMemcpy`` resolves on the real runtime
    library, so installing this in place of ``cumem.libcudart`` cannot
    change the behaviour of anything else.
    """

    def __init__(self, wrapped: Any, direction: str) -> None:
        self._wrapped = wrapped
        self._direction = direction

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def cudaMemcpy(self, dst: int, src: int, count: int) -> None:  # noqa: N802
        """Copy ``count`` bytes in the pinned direction via the driver API."""
        if self._direction == _DTOH:
            ret = _libcuda.cuMemcpyDtoH_v2(dst, src, count)
            assert ret == 0, f"cuMemcpyDtoH_v2 failed with CUresult={ret}"
        else:
            ret = _libcuda.cuMemcpyHtoD_v2(dst, src, count)
            assert ret == 0, f"cuMemcpyHtoD_v2 failed with CUresult={ret}"


@contextlib.contextmanager
def _driver_memcpy(module: Any, direction: str):
    """Route ``module``'s runtime copies through the driver API.

    Not thread-safe by design: ``sleep()`` / ``wake_up()`` are serialized
    engine-level operations executed on the worker's own thread.
    """
    original = module.libcudart
    module.libcudart = _DirectionPinnedLibcudart(original, direction)
    try:
        yield
    finally:
        module.libcudart = original


def _check_memcpy_assumption(func: Any, direction: str) -> None:
    """Warn if the wrapped method no longer does exactly one runtime copy.

    The direction-pinned shim is only correct while ``sleep()`` copies
    solely device->host and ``wake_up()`` solely host->device. This is a
    cheap tripwire so an upstream change surfaces as a log line instead of
    silently copying in the wrong direction.
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return

    count = source.count("libcudart.cudaMemcpy")
    if count != 1:
        logger.warning(
            "[KunlunPlugin] %s contains %d libcudart.cudaMemcpy call(s), "
            "expected exactly 1 (%s). The driver-copy shim in "
            "vllm_kunlun/device_allocator/cumem.py pins the copy direction "
            "and may now be wrong -- please re-check it against upstream.",
            getattr(func, "__qualname__", func),
            count,
            direction,
        )


def patch_cumem(module: Any) -> None:
    """Wrap ``CuMemAllocator.sleep`` / ``wake_up`` on ``module``.

    ``module`` is the upstream ``vllm.device_allocator.cumem`` module.
    """
    cls = getattr(module, "CuMemAllocator", None)
    if cls is None or getattr(cls, "_kunlun_sleep_patched", False):
        return

    if not getattr(module, "cumem_available", False):
        # No cumem extension (sleep mode is unavailable anyway). Mark as
        # handled so the post-import hook stops retrying.
        cls._kunlun_sleep_patched = True
        logger.debug(
            "[KunlunPlugin] cumem allocator unavailable, sleep-mode patch skipped"
        )
        return

    _init_driver_copies()

    original_sleep = cls.sleep
    original_wake_up = cls.wake_up
    _check_memcpy_assumption(original_sleep, _DTOH)
    _check_memcpy_assumption(original_wake_up, _HTOD)

    @functools.wraps(original_sleep)
    def sleep(self, offload_tags=None):
        with _driver_memcpy(module, _DTOH):
            original_sleep(self, offload_tags)
        # sleep() unmaps every allocation it walked, regardless of whether it
        # was offloaded to CPU first (offload_tags only decides backup vs
        # discard), so every tag still on record is now unmapped.
        self._kunlun_unmapped_tags = {
            data.tag for data in self.pointer_to_data.values()
        }

    @functools.wraps(original_wake_up)
    def wake_up(self, tags=None):
        with _driver_memcpy(module, _HTOD):
            original_wake_up(self, tags)
        if tags is None:
            self._kunlun_unmapped_tags = set()
        else:
            self._kunlun_unmapped_tags = set(
                getattr(self, "_kunlun_unmapped_tags", ())
            ) - set(tags)

    cls.sleep = sleep
    cls.wake_up = wake_up
    # Class-level default so readers work before the first sleep().
    cls._kunlun_unmapped_tags = frozenset()
    cls._kunlun_sleep_patched = True
    logger.info(
        "[KunlunPlugin] CuMemAllocator.sleep/wake_up patched to use driver "
        "memcpy in vllm_kunlun/device_allocator/cumem.py"
    )


def _wake_up_weights_if_unmapped(runner: Any) -> None:
    """Re-map the weights pool if a level-2 sleep left it unmapped.

    ``sleep(offload_tags=())`` (level 2) discards the weights pool. A
    standalone ``reload_weights()`` -- no ``wake_up(tags=["weights"])``
    first -- would load into weight tensors backed by released device
    memory and fail with XPU error -707.
    """
    if not runner.vllm_config.model_config.enable_sleep_mode:
        return

    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator.get_instance()
    if "weights" in getattr(allocator, "_kunlun_unmapped_tags", ()):
        logger.info(
            "[KunlunPlugin] reload_weights after a level-2 sleep: re-mapping "
            "the weights pool before loading"
        )
        allocator.wake_up(tags=["weights"])


def patch_reload_weights(module: Any) -> None:
    """Prepend the weights re-map to ``GPUModelRunner.reload_weights``.

    ``module`` is the upstream ``vllm.v1.worker.gpu_model_runner`` module.
    """
    cls = getattr(module, "GPUModelRunner", None)
    if cls is None:
        return

    original = cls.reload_weights
    if getattr(original, "_kunlun_reload_weights_patched", False):
        return

    @functools.wraps(original)
    def reload_weights(self, *args, **kwargs):
        _wake_up_weights_if_unmapped(self)
        return original(self, *args, **kwargs)

    reload_weights._kunlun_reload_weights_patched = True
    cls.reload_weights = reload_weights
    logger.info(
        "[KunlunPlugin] GPUModelRunner.reload_weights patched for level-2 "
        "sleep in vllm_kunlun/device_allocator/cumem.py"
    )
