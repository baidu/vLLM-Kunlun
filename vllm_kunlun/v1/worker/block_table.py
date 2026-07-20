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


# def _compute_slot_mapping(self, num_reqs, query_start_loc, positions):
#     num_tokens = positions.shape[0]
#     max_num_tokens = self.max_num_batched_tokens
#     block_size = self.block_size
#     # CPU mirrors maintained by CpuGpuBuffer. block_table.np is kept in
#     # sync via commit_block_table (CPU->GPU H2D each step), so reading
#     # from it here observes the same data the upstream Triton kernel
#     # would have read from block_table.gpu.
#     block_table_np = self.block_table.np
#     slot_mapping_np = self.slot_mapping.np
#     total_cp = self.pcp_world_size * self.dcp_world_size

#     # Common case: no context parallelism. Pure NumPy path.
#     if total_cp == 1:
#         if num_tokens > 0:
#             # One D2H sync for two small tensors:
#             #   positions: int64[num_tokens] (<= 64 KB at max_num_batched_tokens=8192)
#             #   qsl:       int32[num_reqs+1]
#             pos_np = positions[:num_tokens].cpu().numpy()
#             qsl_np = query_start_loc[: num_reqs + 1].cpu().numpy()
#             # Per-token req index via searchsorted (matches the Triton
#             # kernel's behavior).
#             token_arange = np.arange(num_tokens, dtype=qsl_np.dtype)
#             req_idx = np.searchsorted(qsl_np, token_arange, side="right") - 1
#             np.clip(req_idx, 0, num_reqs - 1, out=req_idx)
#             block_idx = pos_np // block_size
#             offset = pos_np - block_idx * block_size
#             block_num = block_table_np[req_idx, block_idx].astype(np.int64)
#             np.add(
#                 block_num * block_size,
#                 offset,
#                 out=slot_mapping_np[:num_tokens],
#             )
#         if max_num_tokens > num_tokens:
#             slot_mapping_np[num_tokens:max_num_tokens] = PAD_SLOT_ID
#         # H2D the prepared slot_mapping in one shot.
#         self.slot_mapping.copy_to_gpu()
#         return

#     # CP path: per-req loop on host, also in NumPy.
#     total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
#     cp_int = self.cp_kv_cache_interleave_size
#     virtual_block_size = block_size * total_cp
#     qsl_np = query_start_loc[: num_reqs + 1].cpu().numpy()
#     pos_np = positions[:num_tokens].cpu().numpy() if num_tokens > 0 else None
#     for r in range(num_reqs):
#         s, e = int(qsl_np[r]), int(qsl_np[r + 1])
#         if e <= s:
#             continue
#         pos = pos_np[s:e]
#         block_indices = pos // virtual_block_size
#         block_numbers = block_table_np[r, block_indices].astype(np.int64)
#         virtual_off = pos - block_indices * virtual_block_size
#         is_local = (virtual_off // cp_int) % total_cp == total_cp_rank
#         local_off = (virtual_off // (total_cp * cp_int)) * cp_int + (
#             virtual_off % cp_int
#         )
#         slot = block_numbers * block_size + local_off
#         slot_mapping_np[s:e] = np.where(is_local, slot, PAD_SLOT_ID)
#     if max_num_tokens > num_tokens:
#         slot_mapping_np[num_tokens:max_num_tokens] = PAD_SLOT_ID
#     self.slot_mapping.copy_to_gpu()

# Idempotent monkey-patch: safe under fork() and re-import.
if not getattr(_upstream_cls, "_kunlun_slot_patched", False):
    _upstream_cls.compute_slot_mapping = _compute_slot_mapping
    _upstream_cls._kunlun_slot_patched = True
    logger.info(
        "[KunlunPlugin] BlockTable.compute_slot_mapping patched "
        "in vllm_kunlun/v1/worker/block_table.py"
    )
