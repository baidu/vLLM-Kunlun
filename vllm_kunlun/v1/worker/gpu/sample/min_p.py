# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op override for ``vllm.v1.worker.gpu.sample.min_p``."""

import logging

import torch
import vllm.v1.worker.gpu.sample.min_p as _up

logger = logging.getLogger("vllm_kunlun")


def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    """In-place min-p filtering via the Kunlun native op.

    Tokens whose logit is below ``max_logit + log(min_p)`` are set to -inf.
    Rows with ``min_p == 0`` are left unchanged.
    """
    torch.ops.xspeedgate_ops.min_p_inplace(logits, expanded_idx_mapping, min_p)


# ``SamplingStates`` binds this name at sample/states.py:10.
_up.apply_min_p = apply_min_p
logger.info("[KunlunPlugin] V2 min_p patched (xspeedgate_ops native)")
