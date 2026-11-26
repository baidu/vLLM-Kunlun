# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging
import os
from typing import TYPE_CHECKING

import torch
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

if TYPE_CHECKING:
    from lmcache.v1.platform.base.device_ops import DeviceOps


class KunlunDeviceSpec(DeviceSpec):
    """Kunlun uses torch.cuda APIs with native byte-page transfer operations."""

    @property
    def backend_name(self) -> str:
        return "kunlun"

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def torch_module_name(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        if importlib.util.find_spec("torch_xmlir") is None:
            return False
        importlib.import_module("torch_xmlir")
        return torch.cuda.is_available()

    @property
    def ops_cls(self) -> type["DeviceOps"]:
        enabled = os.environ.get("LMCACHE_KUNLUN_TRANSFER_ENABLED", "1").strip()
        if enabled == "1":
            from .device_ops import KunlunDeviceOps

            ops_cls = KunlunDeviceOps
        elif enabled == "0":
            from lmcache.v1.platform.base.device_ops import DeviceOps

            ops_cls = DeviceOps
        else:
            raise ValueError(
                "LMCACHE_KUNLUN_TRANSFER_ENABLED must be '1' or '0', "
                f"got {enabled!r}"
            )
        logging.getLogger(__name__).info(
            "LMCache Kunlun transfer enabled=%s (%s.%s)",
            enabled,
            ops_cls.__module__,
            ops_cls.__name__,
        )
        return ops_cls

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend]:
        """Register shared host buffers through XMLIR's CUDA runtime bindings.

        LMCache maps the same shared-memory pool in each worker. Registering
        that mapping enables nonblocking D2H copies on the store stream;
        registration failures retain upstream's unpinned fallback.
        """
        from lmcache.v1.platform.devices.cuda.pin_memory import CudaPinMemoryBackend

        return CudaPinMemoryBackend
