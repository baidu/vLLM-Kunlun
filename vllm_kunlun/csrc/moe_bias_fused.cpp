/*
 * Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
 *
 * This file is a part of the vllm-kunlun project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

namespace {

torch::Tensor swigluoai_and_mul_native(const torch::Tensor& x) {
    auto gate = x.index({torch::indexing::Slice(), torch::indexing::Slice(
        torch::indexing::None, torch::indexing::None, 2)});
    auto up = x.index({torch::indexing::Slice(), torch::indexing::Slice(
        1, torch::indexing::None, 2)});
    gate = torch::clamp_max(gate, 7.0);
    up = torch::clamp(up, -7.0, 7.0);
    auto glu = gate * torch::sigmoid(gate * 1.702);
    return (up + 1) * glu;
}

}  // namespace

torch::Tensor moe_bias_fused(
    const torch::Tensor& hidden_states,
    const torch::Tensor& w1,
    const torch::Tensor& w2,
    const torch::Tensor& topk_ids,
    const torch::Tensor& normed_score,
    int64_t ep_rank,
    const c10::optional<torch::Tensor>& w1_bias,
    const c10::optional<torch::Tensor>& w2_bias) {
    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D");
    TORCH_CHECK(w1.dim() == 3, "w1 must be 3D");
    TORCH_CHECK(w2.dim() == 3, "w2 must be 3D");
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2D");
    TORCH_CHECK(normed_score.dim() == 2, "normed_score must be 2D");
    TORCH_CHECK(
        w1.device() == hidden_states.device(),
        "w1 must be on the same device as hidden_states");
    TORCH_CHECK(
        w2.device() == hidden_states.device(),
        "w2 must be on the same device as hidden_states");
    TORCH_CHECK(
        topk_ids.device() == hidden_states.device(),
        "topk_ids must be on the same device as hidden_states");
    TORCH_CHECK(
        normed_score.device() == hidden_states.device(),
        "normed_score must be on the same device as hidden_states");

    const auto global_num_experts = w1.size(0);
    const auto moe_top_k = topk_ids.size(1);
    const auto hidden_dim = hidden_states.size(1);
    const auto num_tokens = hidden_states.size(0);
    TORCH_CHECK(
        w2.size(0) == global_num_experts,
        "w2.size(0) must equal w1.size(0), but got ",
        w2.size(0),
        " and ",
        global_num_experts);
    TORCH_CHECK(
        topk_ids.size(0) == num_tokens,
        "topk_ids.size(0) must equal hidden_states.size(0), but got ",
        topk_ids.size(0),
        " and ",
        num_tokens);
    TORCH_CHECK(
        normed_score.size(0) == num_tokens,
        "normed_score.size(0) must equal hidden_states.size(0), but got ",
        normed_score.size(0),
        " and ",
        num_tokens);
    TORCH_CHECK(
        normed_score.size(1) == moe_top_k,
        "normed_score.size(1) must equal topk_ids.size(1), but got ",
        normed_score.size(1),
        " and ",
        moe_top_k);
    TORCH_CHECK(
        w1.size(2) == hidden_dim,
        "w1.size(2) must equal hidden_states.size(1), but got ",
        w1.size(2),
        " and ",
        hidden_dim);
    TORCH_CHECK(
        w2.size(1) == hidden_dim,
        "w2.size(1) must equal hidden_states.size(1), but got ",
        w2.size(1),
        " and ",
        hidden_dim);
    TORCH_CHECK(
        w1.size(1) % 2 == 0,
        "w1.size(1) must be even for SwiGLU split, but got ",
        w1.size(1));
    TORCH_CHECK(
        w1.size(1) / 2 == w2.size(2),
        "w1.size(1) / 2 must equal w2.size(2), but got ",
        w1.size(1) / 2,
        " and ",
        w2.size(2));

    if (w1_bias.has_value() && w1_bias->defined()) {
        TORCH_CHECK(w1_bias->dim() == 2, "w1_bias must be 2D");
        TORCH_CHECK(
            w1_bias->device() == hidden_states.device(),
            "w1_bias must be on the same device as hidden_states");
        TORCH_CHECK(
            w1_bias->size(0) == global_num_experts,
            "w1_bias.size(0) must equal w1.size(0), but got ",
            w1_bias->size(0),
            " and ",
            global_num_experts);
        TORCH_CHECK(
            w1_bias->size(1) == w1.size(1),
            "w1_bias.size(1) must equal w1.size(1), but got ",
            w1_bias->size(1),
            " and ",
            w1.size(1));
    }
    if (w2_bias.has_value() && w2_bias->defined()) {
        TORCH_CHECK(w2_bias->dim() == 2, "w2_bias must be 2D");
        TORCH_CHECK(
            w2_bias->device() == hidden_states.device(),
            "w2_bias must be on the same device as hidden_states");
        TORCH_CHECK(
            w2_bias->size(0) == global_num_experts,
            "w2_bias.size(0) must equal w2.size(0), but got ",
            w2_bias->size(0),
            " and ",
            global_num_experts);
        TORCH_CHECK(
            w2_bias->size(1) == hidden_dim,
            "w2_bias.size(1) must equal hidden_states.size(1), but got ",
            w2_bias->size(1),
            " and ",
            hidden_dim);
    }

    auto repeated_hidden = hidden_states.repeat_interleave(moe_top_k, 0);
    auto topk_ids_flat = topk_ids.reshape({-1});
    auto out = torch::zeros(
        {num_tokens * moe_top_k, hidden_dim},
        hidden_states.options());

    for (int64_t expert_idx = 0; expert_idx < global_num_experts; ++expert_idx) {
        auto expert_id = ep_rank * global_num_experts + expert_idx;
        auto selected_mask = topk_ids_flat.eq(expert_id);
        auto selected_indices = torch::nonzero(selected_mask).reshape({-1});
        if (selected_indices.numel() == 0) {
            continue;
        }

        auto cur_token = repeated_hidden.index_select(0, selected_indices);
        auto groupgemm1 = torch::matmul(cur_token, w1[expert_idx].transpose(0, 1));
        if (w1_bias.has_value() && w1_bias->defined()) {
            groupgemm1 = groupgemm1 + (*w1_bias)[expert_idx];
        }

        auto up_gate = swigluoai_and_mul_native(groupgemm1);
        auto groupgemm2 = torch::matmul(up_gate, w2[expert_idx].transpose(0, 1));
        if (w2_bias.has_value() && w2_bias->defined()) {
            groupgemm2 = groupgemm2 + (*w2_bias)[expert_idx];
        }

        out.index_copy_(0, selected_indices, groupgemm2);
    }

    return (out.view({num_tokens, moe_top_k, hidden_dim}) *
            normed_score.unsqueeze(2))
        .sum(1)
        .to(hidden_states.scalar_type());
}

TORCH_LIBRARY_FRAGMENT(_C, m) {
    m.def("moe_bias_fused", &moe_bias_fused);
}
