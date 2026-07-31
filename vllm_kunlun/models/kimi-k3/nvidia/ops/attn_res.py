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
    num_tokens, hidden_size = prefix.shape
    orig_dtype = prefix.dtype

    updated_prefix = prefix.float()
    if delta is not None:
        updated_prefix = updated_prefix + delta.float()
        # Match the BF16 prefix-add result before using it as a residual source.
        updated_prefix = updated_prefix.to(orig_dtype).float()
        prefix.copy_(updated_prefix.to(orig_dtype))

    if block_write_idx >= 0:
        blocks[:, block_write_idx, :] = updated_prefix.to(blocks.dtype)

    if num_blocks == 0:
        mixed = updated_prefix
    else:
        # Fold the residual-norm weight into the qk direction (matches kernel).
        input_qk_weight = norm_weight.float() * qk_weight.float()  # [H]

        # Sources: the first num_blocks blocks plus the (updated) prefix.
        block_vals = blocks[:, :num_blocks, :].float()  # [T, num_blocks, H]
        sources = torch.cat(
            [block_vals, updated_prefix.unsqueeze(1)], dim=1
        )  # [T, num_blocks + 1, H]

        reciprocal_std = torch.rsqrt(
            sources.pow(2).mean(dim=-1) + eps
        )  # [T, S]
        logits = (sources * input_qk_weight).sum(dim=-1) * reciprocal_std  # [T, S]
        weights = torch.softmax(logits, dim=1)  # [T, S]
        mixed = (weights.unsqueeze(-1) * sources).sum(dim=1)  # [T, H]

    output = mixed
    if output_norm_weight is not None:
        output_reciprocal_std = torch.rsqrt(
            mixed.pow(2).mean(dim=-1, keepdim=True) + output_norm_eps
        )
        output = mixed * output_reciprocal_std * output_norm_weight.float()

    return output.to(orig_dtype)