# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op overrides for ``vllm.v1.worker.gpu.sample.penalties``.

Reimplements prompt/output token statistics and per-step repetition,
frequency and presence penalties on ``torch.ops.xspeedgate_ops.bincount`` /
``.penalties``. The native bincount derives its own prefill bound, so the
wrapper accepts the upstream ``max_prefill_len`` argument only to preserve the
call signature. Both wrappers return before dispatch for an empty batch.
"""

import logging

import torch
import vllm.v1.worker.gpu.sample.penalties as _up

logger = logging.getLogger("vllm_kunlun")


def bincount(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int | None = None,
) -> None:
    # Native op computes its own bounds; the upstream argument is intentionally
    # ignored.
    if expanded_idx_mapping.numel() == 0:
        return
    torch.ops.xspeedgate_ops.bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
    )


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
    if logits.shape[0] == 0:
        return
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


# ``PenaltiesState`` resolves these functions from the upstream module's
# globals; install the native-op versions there.
_up.apply_penalties = apply_penalties
_up.bincount = bincount
logger.info("[KunlunPlugin] V2 penalties patched (xspeedgate_ops native)")
