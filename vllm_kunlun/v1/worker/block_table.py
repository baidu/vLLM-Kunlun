# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific monkey-patch for ``vllm.v1.worker.block_table``.

Replaces ``BlockTable.compute_slot_mapping`` (which dispatches a Triton
kernel ``_compute_slot_mapping_kernel`` upstream) with a torch-native
equivalent. Kunlun XPU cannot JIT-compile Triton kernels via the CUDA
driver path.

Triggering: imported from ``vllm_kunlun.__init__`` post-import hook
once ``vllm.v1.worker.block_table`` is loaded. Idempotent under fork()
and re-import via the ``_kunlun_slot_patched`` flag on the class.
"""

import logging

import torch
from vllm.v1.worker.block_table import PAD_SLOT_ID
from vllm.v1.worker.block_table import BlockTable as _upstream_cls

logger = logging.getLogger("vllm_kunlun")


def _compute_slot_mapping(self, num_reqs, query_start_loc, positions):
    num_tokens = positions.shape[0]
    max_num_tokens = self.max_num_batched_tokens
    block_size = self.block_size
    slot_mapping = self.slot_mapping.gpu
    block_table = self.block_table.gpu
    total_cp = self.pcp_world_size * self.dcp_world_size

    # Common case: no context parallelism. Pure torch index path.
    if total_cp == 1:
        if num_tokens > 0:
            pos = positions[:num_tokens].to(torch.int64)
            # Per-token req index: search query_start_loc.
            qsl = query_start_loc[: num_reqs + 1].to(torch.int64)
            token_arange = torch.arange(
                num_tokens, device=pos.device, dtype=torch.int64
            )
            # req_idx[t] = number of starts <= t, minus 1.
            req_idx = (torch.searchsorted(qsl, token_arange, right=True) - 1).clamp_(
                min=0, max=num_reqs - 1
            )
            block_idx = pos // block_size
            offset = pos - block_idx * block_size
            block_num = block_table[req_idx, block_idx].to(torch.int64)
            slot_mapping[:num_tokens] = block_num * block_size + offset
        if max_num_tokens > num_tokens:
            slot_mapping[num_tokens:max_num_tokens] = PAD_SLOT_ID
        return

    # CP path: fall back to per-req loop on host.
    total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
    cp_int = self.cp_kv_cache_interleave_size
    virtual_block_size = block_size * total_cp
    qsl_cpu = query_start_loc[: num_reqs + 1].cpu().tolist()
    for r in range(num_reqs):
        s, e = qsl_cpu[r], qsl_cpu[r + 1]
        if e <= s:
            continue
        pos = positions[s:e].to(torch.int64)
        block_indices = pos // virtual_block_size
        block_numbers = block_table[r, block_indices].to(torch.int64)
        virtual_off = pos - block_indices * virtual_block_size
        is_local = (virtual_off // cp_int) % total_cp == total_cp_rank
        local_off = (virtual_off // (total_cp * cp_int)) * cp_int + (
            virtual_off % cp_int
        )
        slot = block_numbers * block_size + local_off
        slot_mapping[s:e] = torch.where(
            is_local, slot, torch.full_like(slot, PAD_SLOT_ID)
        )
    if max_num_tokens > num_tokens:
        slot_mapping[num_tokens:max_num_tokens] = PAD_SLOT_ID


# Idempotent monkey-patch: safe under fork() and re-import.
if not getattr(_upstream_cls, "_kunlun_slot_patched", False):
    _upstream_cls.compute_slot_mapping = _compute_slot_mapping
    _upstream_cls._kunlun_slot_patched = True
    logger.info(
        "[KunlunPlugin] BlockTable.compute_slot_mapping patched "
        "in vllm_kunlun/v1/worker/block_table.py"
    )
