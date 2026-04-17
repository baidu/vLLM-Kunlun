"""
Copyright (c) 2026 Baidu, Inc. All Rights Reserved.

This file is a part of the vllm-kunlun project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import patch

import torch

import vllm_kunlun._kunlun  # noqa: F401


def _reference_swigluoai_and_mul(x: torch.Tensor) -> torch.Tensor:
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(max=7.0)
    up = up.clamp(min=-7.0, max=7.0)
    glu = gate * torch.sigmoid(gate * 1.702)
    return (up + 1) * glu


def _reference_moe_bias_fused(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    normed_score: torch.Tensor,
    ep_rank: int,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    global_num_experts = w1.shape[0]
    moe_top_k = topk_ids.shape[1]
    out = torch.zeros(
        num_tokens * moe_top_k,
        hidden_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    repeat_x = hidden_states.repeat_interleave(moe_top_k, dim=0)
    topk_ids_flat = topk_ids.flatten()
    for expert_idx in range(global_num_experts):
        expert_id = ep_rank * global_num_experts + expert_idx
        selected = topk_ids_flat == expert_id
        if not selected.any():
            continue
        cur_token = repeat_x[selected]
        groupgemm1 = cur_token @ w1[expert_idx].T
        if w1_bias is not None:
            groupgemm1 = groupgemm1 + w1_bias[expert_idx]
        up_gate = _reference_swigluoai_and_mul(groupgemm1)
        groupgemm2 = up_gate @ w2[expert_idx].T
        if w2_bias is not None:
            groupgemm2 = groupgemm2 + w2_bias[expert_idx]
        out[selected] = groupgemm2
    return (
        (out.view(num_tokens, moe_top_k, hidden_dim) * normed_score.unsqueeze(2))
        .sum(dim=1)
        .to(hidden_states.dtype)
    )


def test_moe_bias_fused_matches_reference_with_optional_biases():
    hidden_states = torch.randn(3, 4)
    w1 = torch.randn(2, 6, 4)
    w2 = torch.randn(2, 4, 3)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.int32)
    normed_score = torch.tensor(
        [[0.6, 0.4], [0.2, 0.8], [0.5, 0.5]], dtype=hidden_states.dtype
    )
    w1_bias = torch.randn(2, 6)
    w2_bias = torch.randn(2, 4)

    result = torch.ops._C.moe_bias_fused(
        hidden_states,
        w1,
        w2,
        topk_ids,
        normed_score,
        0,
        w1_bias,
        w2_bias,
    )

    expected = _reference_moe_bias_fused(
        hidden_states,
        w1,
        w2,
        topk_ids,
        normed_score,
        0,
        w1_bias,
        w2_bias,
    )
    torch.testing.assert_close(result, expected)


def test_moe_bias_fused_matches_reference_without_biases():
    hidden_states = torch.randn(2, 3)
    w1 = torch.randn(2, 4, 3)
    w2 = torch.randn(2, 3, 2)
    topk_ids = torch.tensor([[0, 1], [1, 1]], dtype=torch.int32)
    normed_score = torch.tensor([[0.7, 0.3], [0.1, 0.9]], dtype=hidden_states.dtype)

    result = torch.ops._C.moe_bias_fused(
        hidden_states,
        w1,
        w2,
        topk_ids,
        normed_score,
        0,
        None,
        None,
    )

    expected = _reference_moe_bias_fused(
        hidden_states,
        w1,
        w2,
        topk_ids,
        normed_score,
        0,
        None,
        None,
    )
    torch.testing.assert_close(result, expected)


def test_moe_bias_fused_respects_ep_rank_global_ids():
    hidden_states = torch.randn(2, 3)
    w1 = torch.randn(2, 4, 3)
    w2 = torch.randn(2, 3, 2)
    topk_ids = torch.tensor([[2, 3], [3, 2]], dtype=torch.int32)
    normed_score = torch.tensor([[0.4, 0.6], [0.5, 0.5]], dtype=hidden_states.dtype)

    result = torch.ops._C.moe_bias_fused(
        hidden_states,
        w1,
        w2,
        topk_ids,
        normed_score,
        1,
        None,
        None,
    )

    expected = _reference_moe_bias_fused(
        hidden_states,
        w1,
        w2,
        topk_ids,
        normed_score,
        1,
        None,
        None,
    )
    torch.testing.assert_close(result, expected)


def test_kunlun_fused_moe_uses_cpp_bias_fast_path():
    module_name = "vllm_kunlun.ops._kunlun_ops"
    fake_ops = {
        "cocopod": types.SimpleNamespace(),
        "xspeedgate_ops": types.SimpleNamespace(),
        "kunlun_ops": types.SimpleNamespace(),
    }
    with patch.dict(sys.modules, fake_ops):
        sys.modules.pop(module_name, None)
        kunlun_ops_module = importlib.import_module(module_name)

    hidden_states = torch.randn(2, 3)
    w1 = torch.randn(2, 4, 3)
    w2 = torch.randn(2, 3, 2)
    router_logits = torch.randn(2, 2)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    normed_score = torch.tensor([[0.3, 0.7], [0.6, 0.4]], dtype=hidden_states.dtype)
    w1_bias = torch.randn(2, 4)
    w2_bias = torch.randn(2, 3)

    def _fill_topk(*args, **kwargs):
        kwargs["normed_score"].copy_(normed_score.float())
        kwargs["topk_index"].copy_(topk_ids)

    with (
        patch.object(
            kunlun_ops_module, "kunlun_ops", types.SimpleNamespace(), create=True
        ),
        patch.object(
            torch.ops._C, "moe_softmax_topk_norm", side_effect=_fill_topk, create=True
        ),
    ):
        result = kunlun_ops_module.KunlunOps.fused_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            router_logits=router_logits,
            ep_rank=0,
            moe_top_k=2,
            renormalize=True,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    expected = _reference_moe_bias_fused(
        hidden_states,
        w1,
        w2,
        topk_ids,
        normed_score,
        0,
        w1_bias,
        w2_bias,
    )
    torch.testing.assert_close(result, expected)
