# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op overrides for ``vllm.v1.worker.gpu.sample.penalties``.

Reimplements the two Triton functions ``bincount`` (prompt/output token
statistics, built once per new penalty request) and ``apply_penalties``
(per-step repetition / frequency / presence penalties) on the Kunlun native ops
``torch.ops.xspeedgate_ops.bincount`` / ``.penalties``.

The native ``bincount`` op derives its own per-request ``max_prefill_len``
internally, so it drops the trailing ``max_prefill_len`` argument the upstream
``PenaltiesState.apply_staged_writes`` (penalties.py:65-74) passes; the wrapper
accepts and ignores it to keep the call site unchanged. Both native ops handle
the spec-decode draft-token accumulation (the ``expanded_local_pos`` path) that
the previous torch-native stand-in omitted.
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
    # Native op computes its own bounds; ``max_prefill_len`` (passed positionally
    # by upstream) is intentionally ignored.
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


# ``PenaltiesState`` resolves ``apply_penalties`` / ``bincount`` from the
# upstream module's globals, so the native-op versions are installed there.
_up.apply_penalties = apply_penalties
_up.bincount = bincount
logger.info("[KunlunPlugin] V2 penalties patched (xspeedgate_ops native)")
