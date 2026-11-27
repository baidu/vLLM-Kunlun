# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native transfers using LMCache's upstream layout and transport preparation."""

import logging

import torch
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc

logger = logging.getLogger(__name__)


class KunlunDeviceOps(DeviceOps):
    """Pass upstream transfer inputs to the native operation without view edits."""

    _native_logged = False

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor: list[torch.Tensor],
        lmcache_objects_ptrs: list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        device: torch.device | str,
        direction: TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        """Submit on the current stream; the upstream caller owns completion."""
        if direction not in (TransferDirection.D2H, TransferDirection.H2D):
            raise ValueError(f"Unsupported transfer direction: {direction}")
        ids = (
            block_ids.cpu().tolist()
            if isinstance(block_ids, torch.Tensor)
            else block_ids
        )
        shape = [
            int(getattr(shape_desc, field))
            for field in (
                "kv_size",
                "nl",
                "nb",
                "bs",
                "nh",
                "hs",
                "element_size",
                "block_stride_elems",
            )
        ]
        if not self._native_logged:
            logger.info(
                "Kunlun LMCache native multi_layer_block_kv_transfer; "
                "upstream format=%s",
                engine_kv_format,
            )
            self._native_logged = True
        torch.ops.xspeedgate_ops.multi_layer_block_kv_transfer(
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            ids,
            direction == TransferDirection.H2D,
            shape,
            lmcache_chunk_size,
            int(engine_kv_format),
            skip_prefix_n_blocks,
        )
