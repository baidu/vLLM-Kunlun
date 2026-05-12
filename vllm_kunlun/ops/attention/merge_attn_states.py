# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import kunlun_ops
import torch


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    # vLLM convention: lse shape is [num_heads, num_tokens]
    # kunlun_ops.attention_merge_stage expects: [num_tokens, num_heads]
    # Must transpose before passing to the kernel and transpose back afterwards.
    num_tokens, num_heads = output.shape[0], output.shape[1]
    if prefix_lse.shape != (num_heads, num_tokens):
        raise RuntimeError(
            f"prefix_lse must be [num_heads, num_tokens]=({num_heads}, {num_tokens}), "
            f"got {tuple(prefix_lse.shape)}"
        )
    if suffix_lse.shape != (num_heads, num_tokens):
        raise RuntimeError(
            f"suffix_lse must be [num_heads, num_tokens]=({num_heads}, {num_tokens}), "
            f"got {tuple(suffix_lse.shape)}"
        )
    out_lse_kernel = torch.empty(
        num_tokens, num_heads, dtype=torch.float32, device=output.device
    )

    kunlun_ops.attention_merge_stage(
        prefix_output,
        prefix_lse.T.contiguous(),  # [num_heads, num_tokens] → [num_tokens, num_heads]
        suffix_output,
        suffix_lse.T.contiguous(),  # same
        output,
        out_lse_kernel,
    )

    if output_lse is not None:
        # Convert back from [num_tokens, num_heads] to [num_heads, num_tokens]
        output_lse.copy_(out_lse_kernel.T)
