#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-kunlun project.
#
"""Utility functions for rotary embeddings."""

from typing import Tuple

import torch


def Split_Norm_Rope(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    max_position_embeddings: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Split + Norm + RoPE operation."""
    num_tokens = qkv.shape[0]
    rotary_dim = head_dim
    q_emb_out = torch.empty(
        (num_tokens, q_head_num * head_dim), dtype=qkv.dtype, device=qkv.device
    )
    k_emb_out = torch.empty(
        (num_tokens, kv_head_num * head_dim), dtype=qkv.dtype, device=qkv.device
    )
    v_out = torch.empty(
        (num_tokens, kv_head_num * head_dim), dtype=qkv.dtype, device=qkv.device
    )
    torch.ops._C.split_norm_rope_neox(
        q_emb_out,
        k_emb_out,
        v_out,
        qkv,
        cos_sin_cache,
        q_norm_weight,
        k_norm_weight,
        positions,
        num_tokens,
        max_position_embeddings,
        q_head_num,
        kv_head_num,
        head_dim,
        rotary_dim,
    )
    return q_emb_out, k_emb_out, v_out
