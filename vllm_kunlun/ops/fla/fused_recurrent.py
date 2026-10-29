# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
from typing import Optional

import torch
import xspeedgate_ops  # noqa: F401 - registers the XSG torch operator


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        inplace_final_state: bool = True,
        cu_seqlens: Optional[torch.LongTensor] = None,
        ssm_state_indices: Optional[torch.Tensor] = None,
        num_accepted_tokens: Optional[torch.Tensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
        o = torch.zeros_like(v)
        if inplace_final_state:
            ht_output = initial_state
        else:
            batch, tokens, _, key_dim = q.shape
            value_heads, value_dim = v.shape[2:]
            state_count = batch * tokens if cu_seqlens is None else tokens
            ht_output = q.new_empty(
                state_count, value_heads, key_dim, value_dim, dtype=torch.float32
            )
        torch.ops.xspeedgate_ops.fused_recurrent_gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            o,
            ht_output,
            beta.ndim == v.ndim,
            use_qk_l2norm_in_kernel,
            inplace_final_state,
            True,
            scale,
            cu_seqlens,
            num_accepted_tokens,
            initial_state,
            ssm_state_indices,
        )
        return o, (initial_state if inplace_final_state else ht_output)


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state
