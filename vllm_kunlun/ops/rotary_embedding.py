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
# This file is a part of the vllm-ascend project.
#

import torch
import xspeedgate_ops
import os
from vllm.model_executor.layers.rotary_embedding import (
    RotaryEmbedding, YaRNScalingRotaryEmbedding, DynamicNTKScalingRotaryEmbedding, MRotaryEmbedding)
from typing import Optional, Tuple

def vllm_kunlun_compute_cos_sin_cache(self) -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = self._compute_inv_freq(self.base)
    if hasattr(self, 'scaling_factor'):
        self.max_position_embeddings = int(self.max_position_embeddings * self.scaling_factor)
    t = torch.arange(self.max_position_embeddings, dtype=torch.float)

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    #对于glm4-9b-chat，rope跑forward_native，所以需要cache保持特定的形状，这里通过环境变量控制
    #对于qwen2.5-vl，rope跑mrope，也需要cache保持特定的形状
    #也就是说跑glm4-9b-chat、qwen2.5-vl，需要设置GLM4_CHAT环境变量为1
    if os.getenv('ROPE_NATIVE_2D') == "1":
        cache = torch.cat((cos, sin), dim=-1)
        return cache
    if os.getenv('USE_ORI_ROPE') == "0":
        cache_cos = torch.cat((cos, cos), dim=-1)
        cache_sin = torch.cat((sin, sin), dim=-1)
        # [2, self.max_position_embeddings, self.rotary_dim * 2]
        cache = torch.stack((cache_cos, cache_sin), dim=0).unsqueeze(1)
    else:
        cache = torch.cat((cos, sin), dim=-1).unsqueeze(0).unsqueeze(1)
    return cache


def vllm_kunlun_forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """forward_cuda"""
        from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops

        if self.cos_sin_cache.device != query.device or \
            self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                       dtype=query.dtype)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                         self.cos_sin_cache,
                                         self.is_neox_style, self.rotary_dim,
                                         offsets)
        else:
            query, key = ops.rotary_embedding(positions, query, key, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key

def apply_interleaved_rope(x: torch.Tensor,
                           mrope_section: list[int]) -> torch.Tensor:
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...TT], preserving frequency continuity.
    """
    x_t = x[0].clone()
    x_t[..., 1:mrope_section[1] * 3:3] = x[1, ..., 1:mrope_section[1] * 3:3]
    x_t[..., 2:mrope_section[2] * 3:3] = x[2, ..., 2:mrope_section[2] * 3:3]
    return x_t

def vllm_kunlun_apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      is_neox_style: bool) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)

def vllm_kunlun_mrope_forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 2
        assert key is not None
        
        query, key = torch.ops.xspeedgate_ops.mrotary_embedding_fwd_v0(
            query,
            key,
            positions.to(dtype=torch.int32),
            self.cos_sin_cache,
            self.mrope_interleaved,
            self.is_neox_style,
            self.head_size,
            self.rotary_dim,
            self.mrope_section[0],
            self.mrope_section[1],
            self.mrope_section[2]
        )

        return query, key

RotaryEmbedding.forward_cuda = vllm_kunlun_forward_cuda
RotaryEmbedding.forward = vllm_kunlun_forward_cuda
MRotaryEmbedding.forward_cuda = vllm_kunlun_mrope_forward_cuda
MRotaryEmbedding.forward = vllm_kunlun_mrope_forward_cuda


def Split_Norm_Rope(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    max_position_embeddings: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim:int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = qkv.shape[0]
    rotary_dim=head_dim
    q_emb_out = torch.empty((num_tokens, q_head_num * head_dim), dtype=qkv.dtype, device=qkv.device)
    k_emb_out = torch.empty((num_tokens, kv_head_num * head_dim), dtype=qkv.dtype, device=qkv.device)
    v_out = torch.empty((num_tokens, kv_head_num * head_dim), dtype=qkv.dtype, device=qkv.device)
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
    return  q_emb_out, k_emb_out, v_out
