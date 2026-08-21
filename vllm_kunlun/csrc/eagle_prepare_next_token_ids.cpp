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

std::tuple<torch::Tensor, torch::Tensor> eagle_prepare_next_token_ids_padded(
    const torch::Tensor& sampled_token_ids,
    const torch::Tensor& discard_request_indices,
    int64_t num_discarded_requests,
    const torch::Tensor& backup_next_token_ids,
    int64_t vocab_size) {
    TORCH_CHECK(sampled_token_ids.dim() == 2, "sampled_token_ids must be 2D");
    TORCH_CHECK(
        backup_next_token_ids.dim() == 1,
        "backup_next_token_ids must be 1D");
    TORCH_CHECK(
        backup_next_token_ids.size(0) >= sampled_token_ids.size(0),
        "backup_next_token_ids must have at least batch_size elements");

    auto valid_sampled_token_ids_gpu = sampled_token_ids.clone();

    if (num_discarded_requests > 0) {
        auto discard_indices =
            discard_request_indices.slice(0, 0, num_discarded_requests);
        discard_indices = discard_indices.to(
            valid_sampled_token_ids_gpu.device(), torch::kLong);
        if (discard_indices.numel() > 0) {
            valid_sampled_token_ids_gpu.index_fill_(0, discard_indices, -1);
        }
    }

    torch::Tensor valid_mask;
    if (sampled_token_ids.size(1) == 1) {
        valid_mask = torch::ones_like(
            valid_sampled_token_ids_gpu,
            valid_sampled_token_ids_gpu.options().dtype(torch::kBool));
    } else {
        valid_mask =
            valid_sampled_token_ids_gpu.ne(-1) &
            valid_sampled_token_ids_gpu.lt(vocab_size);
    }

    auto valid_sampled_tokens_count = valid_mask.sum(1);
    auto last_valid_indices = valid_sampled_tokens_count - 1;
    auto last_valid_indices_safe = torch::clamp_min(last_valid_indices, 0);
    auto selected_tokens =
        valid_sampled_token_ids_gpu.gather(1, last_valid_indices_safe.unsqueeze(1))
            .squeeze(1);
    auto next_token_ids = torch::where(
        last_valid_indices.ne(-1),
        selected_tokens,
        backup_next_token_ids.slice(0, 0, sampled_token_ids.size(0)));

    return std::make_tuple(next_token_ids, valid_sampled_tokens_count);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
    m.def(
        "eagle_prepare_next_token_ids_padded",
        &eagle_prepare_next_token_ids_padded);
}
