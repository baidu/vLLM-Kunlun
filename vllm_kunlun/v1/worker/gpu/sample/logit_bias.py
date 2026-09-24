# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op override for ``vllm.v1.worker.gpu.sample.logit_bias``.

Reimplements ``apply_logit_bias`` (allowed-token masking, logit-bias addition,
and min-token stop-token masking) on
``torch.ops.xspeedgate_ops.logit_bias``. The native op requires int32 position
metadata, whereas the live path gathers ``pos`` from an int64 position buffer;
the wrapper casts only when needed and returns early for an empty batch.
"""

import logging

import torch
import vllm.v1.worker.gpu.sample.logit_bias as _up

logger = logging.getLogger("vllm_kunlun")


def apply_logit_bias(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    pos: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    logit_bias_token_ids: torch.Tensor,
    logit_bias: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
) -> None:
    if logits.shape[0] == 0:
        return
    torch.ops.xspeedgate_ops.logit_bias(
        logits,
        expanded_idx_mapping,
        pos.to(torch.int32),
        num_allowed_token_ids,
        allowed_token_ids,
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias,
        min_lens,
        num_stop_token_ids,
        stop_token_ids,
    )


# ``LogitBiasState`` resolves ``apply_logit_bias`` from the upstream module's
# globals; install the native-op version there.
_up.apply_logit_bias = apply_logit_bias
logger.info("[KunlunPlugin] V2 logit_bias patched (xspeedgate_ops native)")
