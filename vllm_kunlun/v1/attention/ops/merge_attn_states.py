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
    # merge already degenerates to suffix_output (exp(-inf - s_max) == 0; the op
    # uses natural log/exp despite its docstring claiming exp2/log2, measured on
    # P800). This matches the vllm 0.11.0 Kunlun adaptation, which ran chunked
    # prefill correctly without this argument.
    # LSE LAYOUT: vLLM passes log-sum-exp as [num_heads, num_tokens] and
    # kunlun_ops.attention produces that same layout, but
    # attention_merge_stage documents s_a/s_b/s_merged as
    # [num_tokens, num_heads] (kunlun_ops/_attention.py:1278) and derives
    # num_tokens/head_num/head_dim from v_a.shape alone. Handing it the vLLM
    # layout is therefore not a shape error, it is a silent reinterpretation of
    # the buffer: merge weights get shuffled across heads and tokens. The
    # official chunked-prefill test transposes before merging
    # (kunlun_ops test/kunlun_ops/attention/test_chunk_prefill_attention.py:119).
    # The kernel also takes raw data pointers, so non-contiguous inputs (the
    # `x[..., :v_head_dim]` MLA slices) must be materialized first.
    prefix_v = prefix_output.contiguous()
    suffix_v = suffix_output.contiguous()
    prefix_s = prefix_lse.transpose(0, 1).contiguous()
    suffix_s = suffix_lse.transpose(0, 1).contiguous()

    # attention_merge_stage always writes s_merged, even when the caller does
    # not need the merged LSE (e.g. the outer prefill merge).
    merged_s = torch.empty_like(prefix_s)
    merged_v = output if output.is_contiguous() else torch.empty_like(prefix_v)

    ret = kunlun_ops.attention_merge_stage(
        prefix_v, prefix_s, suffix_v, suffix_s, merged_v, merged_s
    )

    if merged_v is not output:
        output.copy_(merged_v)
    if output_lse is not None:
        output_lse.copy_(merged_s.transpose(0, 1))
    return ret
