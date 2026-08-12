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
    prefill_tokens_with_context: Optional[int] = None,
    output_scale: Optional[torch.Tensor] = None,
) -> None:
    # P800 kunlun_ops.attention_merge_stage does a full log-sum-exp merge over
    # all tokens and has no FP8 output path.
    if output_scale is not None:
        raise NotImplementedError(
            "[KUNLUN] merge_attn_states does not support FP8 output_scale"
        )
    # prefill_tokens_with_context is an upstream optimization: skip the merge for
    # tail tokens that have no prefix context and copy suffix directly. It is safe
    # to ignore here because those tokens carry prefix_lse == -inf, so the LSE
    # merge already degenerates to suffix_output (d_a = exp2(-inf) = 0). This
    # matches the vllm 0.11.0 Kunlun adaptation, which ran chunked prefill
    # correctly without this argument.
    if output_lse is None:
        # attention_merge_stage requires a valid s_merged buffer even when the
        # caller does not need the merged LSE (e.g. the outer prefill merge).
        output_lse = torch.empty_like(suffix_lse)

    return kunlun_ops.attention_merge_stage(
        prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
    )
