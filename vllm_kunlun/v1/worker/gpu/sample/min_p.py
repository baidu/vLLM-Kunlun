# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun torch-native replacement for ``vllm.v1.worker.gpu.sample.min_p``."""

import torch


def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    """In-place min-p filtering.

    Tokens whose logit is below ``max_logit + log(min_p)`` are set to -inf.
    Rows with ``min_p == 0`` are left unchanged (threshold -> -inf).
    """
    idx = expanded_idx_mapping.to(torch.long)
    mp = min_p[idx].to(torch.float32)  # [num_tokens]
    max_val = logits.max(dim=-1, keepdim=True).values.to(torch.float32)
    log_mp = torch.where(mp > 0.0, torch.log(mp), torch.full_like(mp, float("-inf")))
    threshold = (max_val + log_mp.unsqueeze(1)).to(logits.dtype)
    logits.masked_fill_(logits < threshold, float("-inf"))
