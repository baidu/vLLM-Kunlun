# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun overrides for ``vllm.v1.worker.gpu.block_table``.

Importing this module rebinds three Triton-backed ``BlockTables`` methods on the
upstream class with torch-native / ``kunlun_ops`` equivalents; everything else
about the class is left untouched. The import is driven by the post-import hook
registered in ``vllm_kunlun/registration/compat_patches.py``.

* ``apply_staged_writes`` — loop ``StagedWriteTensor.apply_write`` per group
  (the Kunlun ``buffer_utils`` provides a torch-native ``apply_write``), which
  also removes the need for the fused multi-group Triton writer.
* ``gather_block_tables`` — gather source rows by ``idx_mapping`` with
  ``index_select`` and zero the padded rows.
* ``compute_slot_mappings`` — reuse the native ``kunlun_ops.compute_slot_mappings``
  (same op the V1 Kunlun path uses), feeding it block tables pre-gathered by
  ``idx_mapping`` so the op's per-token request lookup matches V2 semantics.
"""

import logging

import kunlun_ops
import torch
import vllm.v1.worker.gpu.block_table as _up

logger = logging.getLogger("vllm_kunlun")

PAD_SLOT_ID = _up.PAD_SLOT_ID


def _apply_staged_writes(self) -> None:
    # Single- and multi-group both handled by per-group torch-native writes.
    for block_table in self.block_tables:
        block_table.apply_write()
    self.num_blocks.copy_to_uva()


def _gather_block_tables(self, idx_mapping: torch.Tensor, num_reqs_padded: int):
    num_reqs = idx_mapping.shape[0]
    idx_long = idx_mapping.to(torch.long)
    for i in range(self.num_kv_cache_groups):
        src = self.block_tables[i].gpu  # [max_num_reqs, max_num_blocks]
        dst = self.input_block_tables[i]
        if num_reqs_padded > num_reqs:
            dst[num_reqs:num_reqs_padded].zero_()
        if num_reqs > 0:
            dst[:num_reqs] = src.index_select(0, idx_long)
    return tuple(bt[:num_reqs_padded] for bt in self.input_block_tables)


def _compute_slot_mappings(
    self,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_tokens_padded: int,
) -> torch.Tensor:
    num_reqs = idx_mapping.shape[0]
    num_groups = self.num_kv_cache_groups
    # Total number of real tokens this step (one small D2H sync). Everything
    # past this is padding and must map to PAD_SLOT_ID.
    num_tokens = int(query_start_loc[num_reqs].item())

    # Pad the whole buffer first; valid slots are overwritten below.
    self.slot_mappings.fill_(PAD_SLOT_ID)

    if num_tokens > 0:
        idx_long = idx_mapping.to(torch.long)
        # Pre-gather block tables into batch order so the native op's
        # per-token request index (derived from query_start_loc) addresses
        # the correct rows -- V2 keeps block tables in request-state order and
        # indirects through idx_mapping, which the V1-style op does not do.
        bt_batch = [
            self.block_tables[g].gpu.index_select(0, idx_long)
            for g in range(num_groups)
        ]
        slot_list = [self.slot_mappings[g] for g in range(num_groups)]
        kunlun_ops.compute_slot_mappings(
            slot_list,
            bt_batch,
            positions,
            query_start_loc,
            self.block_sizes_tensor,
            num_reqs,
            num_tokens,
            PAD_SLOT_ID,
            self.cp_size,
            self.cp_rank,
            self.cp_interleave,
        )

    return self.slot_mappings[:, :num_tokens_padded]


_up.BlockTables.apply_staged_writes = _apply_staged_writes
_up.BlockTables.gather_block_tables = _gather_block_tables
_up.BlockTables.compute_slot_mappings = _compute_slot_mappings
logger.info("[KunlunPlugin] V2 BlockTables patched (torch-native + kunlun_ops)")
