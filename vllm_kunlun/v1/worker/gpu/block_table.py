# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op overrides for ``vllm.v1.worker.gpu.block_table``.

Importing this module rebinds three Triton-backed ``BlockTables`` methods on the
upstream class with Kunlun native ops; everything else about the class is left
untouched. The import is driven by the post-import hook registered in
``vllm_kunlun/registration/compat_patches.py``.

* ``apply_staged_writes`` — loop ``StagedWriteTensor.apply_write`` per group
  (the Kunlun ``buffer_utils`` provides an ``xspeedgate_ops.apply_write``-backed
  ``apply_write``), which also removes the need for the fused multi-group
  writer.
* ``gather_block_tables`` — ``torch.ops.xspeedgate_ops.gather_block_tables``.
  The native op does the ``idx_mapping`` gather, the per-state
  ``num_blocks``-bounded row copy, and the padded-row zeroing on-device in one
  launch, matching upstream ``_gather_block_tables_kernel`` (which only copies
  ``[0, num_blocks)`` of each valid row and zeros padded rows).
* ``compute_slot_mappings`` — ``torch.ops.xspeedgate_ops.compute_slot_mappings``.
  The native op takes the freely available host int ``num_tokens_padded`` and
  derives the real token count from ``query_start_loc`` on-device, so it removes
  the ``.item()`` host sync the previous ``kunlun_ops`` wrapper had to pay every
  step. It also does the ``idx_mapping`` indirection internally (so the state
  block tables are passed as-is, not pre-gathered) and clears the padded tail of
  the buffer to ``PAD_SLOT_ID`` (-1) itself.

Both gather methods index with ``idx_mapping`` unguarded, which is safe: the
``-1`` sentinel never reaches them. Their callers pass either
``InputBatch.idx_mapping`` (model_runner.py:1026/1032), built from
``req_id_to_index.get`` and so always non-negative, or the draft speculator's
``self.idx_mapping[:num_reqs]`` sliced to the unpadded request count
(spec_decode/autoregressive/speculator.py:359) -- the ``-1`` padding it writes
lives at ``[num_reqs:]``. See the sentinel-invariant note in
``vllm_kunlun/v1/worker/gpu/input_batch.py`` for where ``-1`` does occur.

The native ``compute_slot_mappings`` produces ``PAD_SLOT_ID == -1`` for padding
directly, so this override no longer depends on the ``kunlun_ops`` package.
"""

import logging

import torch
import vllm.v1.worker.gpu.block_table as _up

logger = logging.getLogger("vllm_kunlun")

PAD_SLOT_ID = _up.PAD_SLOT_ID


def _apply_staged_writes(self) -> None:
    # Single- and multi-group both handled by per-group native writes.
    for block_table in self.block_tables:
        block_table.apply_write()
    self.num_blocks.copy_to_uva()


def _gather_block_tables(self, idx_mapping: torch.Tensor, num_reqs_padded: int):
    # The native op gathers by idx_mapping, copies each valid row's first
    # ``num_blocks[state]`` entries, and zeros padded rows -- all on-device.
    torch.ops.xspeedgate_ops.gather_block_tables(
        [bt.gpu for bt in self.block_tables],
        self.input_block_tables,
        self.num_blocks.gpu,
        idx_mapping,
        num_reqs_padded,
    )
    return tuple(bt[:num_reqs_padded] for bt in self.input_block_tables)


def _compute_slot_mappings(
    self,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_tokens_padded: int,
) -> torch.Tensor:
    # ``num_tokens_padded`` is a host int already known to the caller; the
    # native op reads the real token count from ``query_start_loc[num_reqs]``
    # on-device and pads the remainder of ``slot_mappings`` with PAD_SLOT_ID
    # itself, so there is no host sync here (unlike the old kunlun_ops path,
    # which had to ``.item()`` the real token count to size its write). It also
    # does the idx_mapping indirection internally, so the state block tables are
    # passed as-is rather than pre-gathered into batch order.
    torch.ops.xspeedgate_ops.compute_slot_mappings(
        [bt.gpu for bt in self.block_tables],
        idx_mapping,
        query_start_loc,
        positions,
        self.slot_mappings,
        self.block_sizes_tensor,
        num_tokens_padded,
        self.cp_rank,
        self.cp_size,
        self.cp_interleave,
    )
    return self.slot_mappings[:, :num_tokens_padded]


_up.BlockTables.apply_staged_writes = _apply_staged_writes
_up.BlockTables.gather_block_tables = _gather_block_tables
_up.BlockTables.compute_slot_mappings = _compute_slot_mappings
logger.info("[KunlunPlugin] V2 BlockTables patched (xspeedgate_ops native)")
