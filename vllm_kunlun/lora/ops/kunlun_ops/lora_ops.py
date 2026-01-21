"""kunlun_ops for lora"""
 
import torch
import xspeedgate_ops
import time
from torch._C import dtype
import os
from torch._dynamo import disable


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
 

    return torch.ops.xspeedgate_ops.sgmv_shrink_cluster(inputs, lora_a_weights, seq_len_tensor, lora_indices_tensor, output_tensor, scaling)
 


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
 
    return torch.ops.xspeedgate_ops.sgmv_expand_cluster(inputs, lora_b_weights, seq_len_tensor, lora_indices_tensor, output_tensor, 0)
 


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
 

    return torch.ops.xspeedgate_ops.sgmv_expand_cluster(inputs, lora_b_weights, seq_len_tensor, lora_indices_tensor, output_tensor, slice_offset)
 
 
 
 


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
# @my_wrapper

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