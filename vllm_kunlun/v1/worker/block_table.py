# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific monkey-patch for ``vllm.v1.worker.block_table``.

Replaces ``BlockTable.compute_slot_mapping`` (which dispatches a Triton
kernel ``_compute_slot_mapping_kernel`` upstream) with the native Kunlun
XPU op ``kunlun_ops.compute_slot_mappings``. Kunlun XPU cannot JIT-compile
Triton kernels via the CUDA driver path; the previous torch-native fallback
relied on ``torch.searchsorted``, which has no Kunlun XPU implementation and
silently falls back to CPU (host sync + D2H/H2D round trip every step). The
native op computes the whole mapping on-device in a single launch.

Triggering: imported from ``vllm_kunlun.__init__`` post-import hook
once ``vllm.v1.worker.block_table`` is loaded. Idempotent under fork()
and re-import via the ``_kunlun_slot_patched`` flag on the class.
"""

import logging

import kunlun_ops
import torch
from vllm.v1.worker.block_table import PAD_SLOT_ID
from vllm.v1.worker.block_table import BlockTable as _upstream_cls

logger = logging.getLogger("vllm_kunlun")


def _compute_slot_mapping(self, num_reqs, query_start_loc, positions):
    num_tokens = positions.shape[0]
    total_cp_world_size = self.pcp_world_size * self.dcp_world_size
    total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
    block_sizes = torch.tensor(
        [self.block_size], dtype=torch.int32, device=self.block_table.gpu.device
    )
    kunlun_ops.compute_slot_mappings(
        [self.slot_mapping.gpu],  # list
        [self.block_table.gpu],  # list
        positions,
        query_start_loc,
        block_sizes,  # int32 tensor
        num_reqs,
        num_tokens,
        PAD_SLOT_ID,
        total_cp_world_size,
        total_cp_rank,
        self.cp_kv_cache_interleave_size,
    )


# Idempotent monkey-patch: safe under fork() and re-import.
if not getattr(_upstream_cls, "_kunlun_slot_patched", False):
    _upstream_cls.compute_slot_mapping = _compute_slot_mapping
    _upstream_cls._kunlun_slot_patched = True
    logger.info(
        "[KunlunPlugin] BlockTable.compute_slot_mapping patched "
        "in vllm_kunlun/v1/worker/block_table.py"
    )
