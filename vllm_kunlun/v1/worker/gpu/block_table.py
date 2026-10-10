# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacements for ``vllm.v1.worker.gpu.block_table``.

``BlockTables.gather_block_tables`` and ``compute_slot_mappings`` launch Triton
kernels upstream. Kunlun XPU cannot JIT-compile Triton kernels, so each is
replaced by its xspeedgate_ops equivalent.

``BlockTables.apply_staged_writes`` is **not** replaced: with the fused
multi-group ``xspeedgate_ops.apply_write`` available, the upstream body works
verbatim -- its single-group branch lands in ``StagedWriteTensor.apply_write``
and its multi-group branch in ``FusedStagedWriter.apply``, both of which the
``buffer_utils`` overlay redirects to the op.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)

logger = logging.getLogger("vllm_kunlun")


def gather_block_tables(
    self,
    idx_mapping: torch.Tensor,
    num_reqs_padded: int,
    out: tuple[torch.Tensor, ...] | None = None,
    out_ptrs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if self.num_kv_cache_groups == 0:
        return ()
    if out is None:
        out = tuple(self.input_block_tables)
        out_ptrs = self.input_block_table_ptrs
    else:
        assert out_ptrs is not None
        assert len(out) == self.num_kv_cache_groups
    torch.ops.xspeedgate_ops.gather_block_tables(
        [b.gpu for b in self.block_tables],
        list(out),
        self.num_blocks.gpu,
        idx_mapping,
        num_reqs_padded,
    )
    return tuple(bt[:num_reqs_padded] for bt in out)


def compute_slot_mappings(
    self,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_tokens_padded: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if self.num_kv_cache_groups == 0:
        return (self.slot_mappings if out is None else out)[:, :num_tokens_padded]
    slot_mappings = self.slot_mappings if out is None else out
    torch.ops.xspeedgate_ops.compute_slot_mappings(
        [b.gpu for b in self.block_tables],
        idx_mapping,
        query_start_loc,
        positions,
        slot_mappings,
        self.block_sizes_tensor,
        num_tokens_padded,
        self.cp_rank,
        self.cp_size,
        self.cp_interleave,
    )
    return slot_mappings[:, :num_tokens_padded]


gather_block_tables._kunlun_patched = True
compute_slot_mappings._kunlun_patched = True
