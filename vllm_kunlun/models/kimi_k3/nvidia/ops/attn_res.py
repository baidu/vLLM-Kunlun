# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# [KUNLUN] P800 port of vllm/models/kimi_k3/nvidia/ops/attn_res.py.
# The upstream `_attn_res_kernel` is a @triton.jit CUDA kernel that cannot run
# on Kunlun XPU (Triton CUDA binary load -> CUDA_ERROR_NOT_SUPPORTED). Here we
# keep the exact `attn_res` semantics but implement them with plain torch ops.

import torch


# Consumed by kimi_k3_triton_warmup.py during kernel_warmup().
def get_attn_res_triton_warmup_profiles(
    max_blocks: int,
) -> tuple[tuple[int, bool, int, bool], ...]:
    """Return the small-batch profiles that bypass the native kernel."""
    profiles = [
        (num_blocks, False, -1, True) for num_blocks in range(2, max_blocks + 1)
    ]
    profiles.extend(
        (block_write_idx, True, block_write_idx, True)
        for block_write_idx in range(2, max_blocks)
    )
    profiles.append((max_blocks, True, -1, False))
    return tuple(profiles)


def attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    """Torch equivalent of the upstream ``_attn_res_kernel`` (per-token).

    For each token row:
      1. ``updated_prefix = prefix + delta`` (if delta is given), rounded through
         the prefix dtype to match the bf16 in-place add, then written back into
         ``prefix``.
      2. If ``block_write_idx >= 0``: store ``updated_prefix`` into
         ``blocks[:, block_write_idx, :]``.
      3. If ``num_blocks == 0``: ``mixed = updated_prefix``. Otherwise run an
         "attention residual" softmax over the sources
         ``{blocks[:, :num_blocks], updated_prefix}``:
             logit_i = (source_i . (norm_weight * qk_weight)) * rsqrt(mean(source_i^2) + eps)
             mixed   = softmax_i(logit) @ source_i
      4. If ``output_norm_weight`` is given: RMSNorm(mixed) * output_norm_weight.
    """
    return torch.ops.xspeedgate_ops.attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        eps,
        output_norm_eps,
    )