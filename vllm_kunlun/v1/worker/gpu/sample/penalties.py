# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacement for ``vllm.v1.worker.gpu.sample.penalties``.

``apply_penalties`` and ``bincount`` launch the Triton ``_penalties_kernel`` /
``_bincount_kernel`` upstream. Kunlun XPU cannot JIT-compile Triton kernels, so
both are replaced by their xspeedgate_ops equivalents.

``bincount`` zeroes the rows it is given itself, exactly like the kernel, so the
caller must not clear them a second time. It derives its own grid from
``prefill_len``, hence ``max_prefill_len`` is unused here.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)

logger = logging.getLogger("vllm_kunlun")


def apply_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    torch.ops.xspeedgate_ops.penalties(
        logits,
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
    )


def bincount(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int,
) -> None:
    torch.ops.xspeedgate_ops.bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
    )


apply_penalties._kunlun_patched = True
bincount._kunlun_patched = True
