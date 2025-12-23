#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
# Author: Wang Hao
# Email: wanghao129@baidu.com
#
# This file is a part of the vllm-kunlun project.
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
"""kunlun_ops for lora"""

import torch
import xspeedgate_ops
import time
from torch._C import dtype
import os


def sgmv_shrink(
    inputs: torch.Tensor,  
    lora_a_weights: torch.Tensor,  
    output_tensor: torch.Tensor, 
    block_statistic: torch.Tensor, 
    sorted_tokens_num_lod: torch.Tensor,
    moe_index: torch.Tensor,
    expert_m: torch.Tensor,
    b_seq_start_loc: torch.Tensor, 
    seq_len_tensor: torch.Tensor,  
    lora_indices_tensor: torch.Tensor,  
    batches: int,  
    max_seq_length: int,  
    token_nums: int,  
    scaling: float,  
):
    """
    sgmv_shrink
    """

    if inputs.shape[0] <= 128:
        return torch.ops.xspeedgate_ops.sgmv_shrink_cluster(inputs, lora_a_weights, seq_len_tensor, lora_indices_tensor, output_tensor, scaling)

    expert_num = 9
    device = inputs.device

    lora_ids = lora_indices_tensor.repeat_interleave(seq_len_tensor, dim=0).to(
        device=device, dtype=torch.int32
    )

    lora_ids.masked_fill_(lora_ids < 0, expert_num - 1).unsqueeze_(1)



    torch.ops._C.gen_block_statistic(lora_ids, block_statistic)


    inputs_sorted = torch.zeros_like(inputs, dtype=inputs.dtype, device=device)
    torch.ops._C.moe_pre_sorted(
        inputs,
        lora_ids,
        block_statistic,
        inputs_sorted,
        moe_index,
        expert_m,
        sorted_tokens_num_lod
    )


    output_tensor.unsqueeze_(1)

    torch.ops._C.moe_fc(
        x=inputs_sorted,
        weight=lora_a_weights,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=moe_index,
        moe_topk=1,
        y=output_tensor,
        act=None,
        x_perchannel_max=None,
        w_perchannel_max=None,
        topk_ids=None,
        topk_w=None,
        bias=None,
        tgemm_type=None,
        tweight_type=None,
        scale_n=0,
        scale_k=0,
        use_pack_int4=False
    )

    output_tensor.squeeze_(1).mul_(scaling)

    return output_tensor


def sgmv_expand(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                block_statistic: torch.Tensor,
                sorted_tokens_num_lod: torch.Tensor,
                moe_index: torch.Tensor,
                b_seq_start_loc: torch.Tensor,
                seq_len_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                batches: int,
                max_seq_length: int,
                token_nums: int,
                add_inputs: bool = False):
    """
    sgmv_expand
    """

    if inputs.shape[0] <= 128:
        return torch.ops.xspeedgate_ops.sgmv_expand_cluster(inputs, lora_b_weights, seq_len_tensor, lora_indices_tensor, output_tensor, 0)


    expert_num = 9
    device = inputs.device


    lora_ids = lora_indices_tensor.repeat_interleave(seq_len_tensor, dim=0).to(
        device=device, dtype=torch.int32
    )

    lora_ids.masked_fill_(lora_ids < 0, expert_num - 1).unsqueeze_(1)
  
    out = torch.zeros((token_nums, 1, slice_size), dtype=inputs.dtype, device=device)


    torch.ops._C.moe_fc(
        x=inputs,
        weight=lora_b_weights,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=moe_index,
        moe_topk=1,
        y=out,
        act=None,
        x_perchannel_max=None,
        w_perchannel_max=None,
        topk_ids=None,
        topk_w=None,
        bias=None,
        tgemm_type=None,
        tweight_type=None,
        scale_n=0,
        scale_k=0,
        use_pack_int4=False
    )

    output_post = out.squeeze(1)  
    torch.ops._C.moe_post(
        output_post,                
        moe_index.unsqueeze(1),     
        normed_scale,              
        normed_scale,             
        output_post                
    )


    common_len = min(output_post.shape[1], output_tensor.shape[1])

    limit = min(output_post.shape[0], output_tensor.shape[0])


    if add_inputs:
        output_tensor[:limit, :common_len] += output_post[:limit, :common_len]
    else:
        output_tensor[:limit, :common_len] = output_post[:limit, :common_len]

    return output_tensor


def sgmv_expand_slice(inputs: torch.Tensor,
                      lora_b_weights: torch.Tensor,
                      output_tensor: torch.Tensor,
                      block_statistic: torch.Tensor,
                      sorted_tokens_num_lod: torch.Tensor, 
                      moe_index: torch.Tensor, 
                      normed_scale: torch.Tensor,
                      b_seq_start_loc: torch.Tensor,
                      seq_len_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor,
                      batches: int,  
                      max_seq_length: int, 
                      token_nums: int,
                      slice_offset: int,
                      slice_size: int,
                      add_inputs: bool = False):
    
    """
    sgmv_expand_slice
    """

    if inputs.shape[0] <= 128:
        return torch.ops.xspeedgate_ops.sgmv_expand_cluster(inputs, lora_b_weights, seq_len_tensor, lora_indices_tensor, output_tensor, slice_offset)

    expert_num = 9
    device = inputs.device

    lora_ids = lora_indices_tensor.repeat_interleave(seq_len_tensor, dim=0).to(
        device=device, dtype=torch.int32
    )

    lora_ids.masked_fill_(lora_ids < 0, expert_num - 1).unsqueeze_(1)


    out = torch.zeros((token_nums, 1, slice_size), dtype=inputs.dtype, device=device)


    torch.ops._C.moe_fc(
        x=inputs,
        weight=lora_b_weights,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=moe_index,
        moe_topk=1,
        y=out,
        act=None,
        x_perchannel_max=None,
        w_perchannel_max=None,
        topk_ids=None,
        topk_w=None,
        bias=None,
        tgemm_type=None,
        tweight_type=None,
        scale_n=0,
        scale_k=0,
        use_pack_int4=False
    )

    output_post = out.squeeze(1)  
    torch.ops._C.moe_post(
        output_post,                
        moe_index.unsqueeze(1),     
        normed_scale,              
        normed_scale,             
        output_post                
    )


    slice_end = slice_offset + slice_size
    actual_slice_size = min(slice_size, output_tensor.shape[1] - slice_offset)

    limit = min(output_post.shape[0], output_tensor.shape[0])


    if add_inputs:
        output_tensor[:limit, slice_offset:slice_end] += output_post[:limit, :actual_slice_size]
    else:
        output_tensor[:limit, slice_offset:slice_end] = output_post[:limit, :actual_slice_size]

    return output_tensor


def bgmv_shrink(
    inputs: torch.Tensor,  # [m, hidden_dim]
    lora_a_weights: torch.Tensor,  # [n, 1, r, hidden_dim]
    output_tensor: torch.Tensor,  # [m, r]
    block_statistic: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    moe_index: torch.Tensor,
    expert_m: torch.Tensor,
    lora_indices_tensor: torch.Tensor,  # [m]
    scaling: float = 1.0
) -> torch.Tensor:
    """
    bgmv_shrink
    """
    return torch.ops.xspeedgate_ops.bgmv_shrink_cluster(inputs, lora_a_weights, lora_indices_tensor, output_tensor, scaling)

def bgmv_expand(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                block_statistic: torch.Tensor,
                sorted_tokens_num_lod: torch.Tensor,
                moe_index: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                add_inputs: bool = True):
    """"
        bgmv_expand
    """
    return torch.ops.xspeedgate_ops.bgmv_expand_cluster(inputs, lora_b_weights, lora_indices_tensor, output_tensor, 0)

def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    block_statistic: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    moe_index: torch.Tensor,
    normed_scale: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True
):
    """
        bgmv_expand_slice
    """
    return torch.ops.xspeedgate_ops.bgmv_expand_cluster(inputs, lora_b_weights, lora_indices_tensor, output_tensor, slice_offset)

