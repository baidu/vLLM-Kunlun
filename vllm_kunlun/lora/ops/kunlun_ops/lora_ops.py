"""kunlun_ops for lora"""
 
import torch


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
    if seq_len_tensor.size(0) == 1:
                lora_a_2d = lora_a_weights.squeeze(0)
                torch.ops._C.lora_matmul_inplace(
                    inputs,
                    lora_a_2d,
                    output_tensor,
                    x_trans=False,
                    w_trans=True,
                    alpha=1.0,
                    beta=1.0
                )
    else:
        torch.ops._C.sgmv_shrink_lora(
            inputs, lora_a_weights, output_tensor, block_statistic,
            sorted_tokens_num_lod, moe_index, expert_m, b_seq_start_loc,
            seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
            token_nums, scaling
        )
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
    if seq_len_tensor.size(0) == 1:
        lora_b_2d = lora_b_weights.squeeze(0)
        torch.ops._C.lora_matmul_inplace(
            inputs,
            lora_b_2d,
            output_tensor,
            x_trans=False,
            w_trans=True,
            alpha=1.0,
            beta=1.0
        )
    else:
        torch.ops._C.sgmv_expand_lora(
            inputs, lora_b_weights, output_tensor, block_statistic,
            sorted_tokens_num_lod, moe_index, b_seq_start_loc, seq_len_tensor,
            lora_indices_tensor, batches, max_seq_length, token_nums, add_inputs
        )
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
    if seq_len_tensor.size(0) == 1:
        lora_b_2d = lora_b_weights.squeeze(0)
        output_slice = output_tensor.narrow(1, slice_offset, slice_size)
        beta = 1.0 if add_inputs else 0.0
        output_slice.addmm_(inputs, lora_b_2d.t(), beta=beta, alpha=1.0)
    else:
        torch.ops._C.sgmv_expand_slice_lora(
            inputs, lora_b_weights, output_tensor, block_statistic,
            sorted_tokens_num_lod, moe_index, normed_scale, b_seq_start_loc,
            seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
            token_nums, slice_offset, slice_size, add_inputs
        )
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
    if inputs.size(0) == 1:
        lora_idx = lora_indices_tensor[0].item()
        lora_a_2d = lora_a_weights[lora_idx].squeeze(0)
        torch.ops._C.lora_matmul_inplace(
            inputs,
            lora_a_2d,
            output_tensor,
            x_trans=False,
            w_trans=True,
            alpha=scaling,
            beta=1.0
        )
    else:
        torch.ops._C.bgmv_shrink_lora(
            inputs, lora_a_weights, output_tensor, block_statistic,
            sorted_tokens_num_lod, moe_index, expert_m, lora_indices_tensor, scaling
        )
    return output_tensor


def bgmv_expand(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                block_statistic: torch.Tensor,
                sorted_tokens_num_lod: torch.Tensor,
                moe_index: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                add_inputs: bool = True):
    """
    bgmv_expand
    """
    if inputs.size(0) == 1:
        lora_idx = lora_indices_tensor[0].item()
        lora_b_2d = lora_b_weights[lora_idx].squeeze(0)
        beta = 1.0 if add_inputs else 0.0
        torch.ops._C.lora_matmul_inplace(
            inputs,
            lora_b_2d,
            output_tensor,
            x_trans=False,
            w_trans=True,
            alpha=1.0,
            beta=beta
        )
    else:
        torch.ops._C.bgmv_expand_lora(
            inputs, lora_b_weights, output_tensor, block_statistic,
            sorted_tokens_num_lod, moe_index, lora_indices_tensor, add_inputs
        )
    return output_tensor


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
    if inputs.size(0) == 1:
        lora_idx = lora_indices_tensor[0].item()
        lora_b_2d = lora_b_weights[lora_idx].squeeze(0)
        output_slice = output_tensor.narrow(1, slice_offset, slice_size)
        beta = 1.0 if add_inputs else 0.0
        output_slice.addmm_(inputs, lora_b_2d.t(), beta=beta, alpha=1.0)
    else:
        torch.ops._C.bgmv_expand_slice_lora(
            inputs, lora_b_weights, output_tensor, block_statistic,
            sorted_tokens_num_lod, moe_index, normed_scale, lora_indices_tensor,
            slice_offset, slice_size, add_inputs
        )
    return output_tensor