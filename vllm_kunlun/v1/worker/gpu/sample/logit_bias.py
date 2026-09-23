# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacement for ``vllm.v1.worker.gpu.sample.logit_bias``.

``apply_logit_bias`` launches the Triton ``_bias_kernel`` upstream. Kunlun XPU
cannot JIT-compile Triton kernels, so it is replaced by the xspeedgate_ops
equivalent, which applies allowed-token masking, per-token logit bias and stop
tokens in one launch. The op needs int32 index/count tensors, including ``pos``.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)

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


apply_logit_bias._kunlun_patched = True
