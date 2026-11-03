"""kunlun_ops for lora"""

import torch

_KERNEL_DTYPE = torch.float16

# Pooled fp16 scratch for ``_accumulate`` (bf16-output path), keyed by
# (width, device); ``_ACC_RETIRED`` holds replaced blocks alive because a
# captured cudagraph may still reference their addresses.
_ACC_POOL: dict[tuple, torch.Tensor] = {}
_ACC_RETIRED: list[torch.Tensor] = []


def _to_kernel_dtype(tensor: torch.Tensor) -> torch.Tensor:
    """Cast to the dtype the xspeedgate LoRA kernels accept."""
    if tensor.dtype == _KERNEL_DTYPE:
        return tensor
    return tensor.to(_KERNEL_DTYPE)


def _kernel_weights(weights: torch.Tensor) -> torch.Tensor:
    """Stacked LoRA weights in the layout the kernels expect, in kernel dtype."""
    if weights.dim() == 4 and weights.size(1) == 1:
        weights = weights.squeeze(1)
    return _to_kernel_dtype(weights)


def _safe_indices(lora_indices_tensor: torch.Tensor) -> torch.Tensor:
    """Return int32 adapter indices that are always in range."""
    return lora_indices_tensor.to(torch.int32).clamp_min(0)


def _accumulate(output_tensor: torch.Tensor, launch) -> None:
    """Run ``launch(y)`` and accumulate the float16 result into ``output_tensor``."""
    if output_tensor.dtype == _KERNEL_DTYPE:
        launch(output_tensor)
        return
    width = output_tensor.shape[-1]
    rows = output_tensor.numel() // width
    key = (width, output_tensor.device)
    buf = _ACC_POOL.get(key)
    if buf is None or buf.shape[0] < rows:
        bucket = 1 << (rows - 1).bit_length() if rows > 1 else rows
        if buf is not None:
            _ACC_RETIRED.append(buf)
        buf = torch.empty((bucket, width), dtype=_KERNEL_DTYPE, device=key[1])
        _ACC_POOL[key] = buf
    scratch = buf[:rows].view(output_tensor.shape)
    scratch.zero_()
    launch(scratch)
    output_tensor.add_(scratch.to(output_tensor.dtype))


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
    _accumulate(
        output_tensor,
        lambda y: torch.ops.xspeedgate_ops.sgmv_shrink_sdnn(
            _to_kernel_dtype(inputs),
            _kernel_weights(lora_a_weights),
            seq_len_tensor.to(torch.int32),
            _safe_indices(lora_indices_tensor),
            y,
            scaling,
        ),
    )


def sgmv_expand(
    inputs: torch.Tensor,
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
    add_inputs: bool = False,
):
    """
    sgmv_expand
    """
    _accumulate(
        output_tensor,
        lambda y: torch.ops.xspeedgate_ops.sgmv_expand_sdnn(
            _to_kernel_dtype(inputs),
            _kernel_weights(lora_b_weights),
            seq_len_tensor.to(torch.int32),
            _safe_indices(lora_indices_tensor),
            y,
            0,
        ),
    )


def sgmv_expand_slice(
    inputs: torch.Tensor,
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
    add_inputs: bool = False,
):
    """
    sgmv_expand_slice
    """
    _accumulate(
        output_tensor,
        lambda y: torch.ops.xspeedgate_ops.sgmv_expand_sdnn(
            _to_kernel_dtype(inputs),
            _kernel_weights(lora_b_weights),
            seq_len_tensor.to(torch.int32),
            _safe_indices(lora_indices_tensor),
            y,
            slice_offset,
        ),
    )


def bgmv_shrink(
    inputs: torch.Tensor,  # [m, hidden_dim]
    lora_a_weights: torch.Tensor,  # [n, 1, r, hidden_dim]
    output_tensor: torch.Tensor,  # [m, r]
    block_statistic: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    moe_index: torch.Tensor,
    expert_m: torch.Tensor,
    lora_indices_tensor: torch.Tensor,  # [m]
    scaling: float = 1.0,
) -> torch.Tensor:
    """
    bgmv_shrink
    """
    _accumulate(
        output_tensor,
        lambda y: torch.ops.xspeedgate_ops.bgmv_shrink_cluster(
            _to_kernel_dtype(inputs),
            _kernel_weights(lora_a_weights),
            _safe_indices(lora_indices_tensor),
            y,
            scaling,
        ),
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    block_statistic: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    moe_index: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    """ "
    bgmv_expand
    """
    _accumulate(
        output_tensor,
        lambda y: torch.ops.xspeedgate_ops.bgmv_expand_cluster(
            _to_kernel_dtype(inputs),
            _kernel_weights(lora_b_weights),
            _safe_indices(lora_indices_tensor),
            y,
            0,
        ),
    )


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
    add_inputs: bool = True,
):
    """
    bgmv_expand_slice
    """
    _accumulate(
        output_tensor,
        lambda y: torch.ops.xspeedgate_ops.bgmv_expand_cluster(
            _to_kernel_dtype(inputs),
            _kernel_weights(lora_b_weights),
            _safe_indices(lora_indices_tensor),
            y,
            slice_offset,
        ),
    )
