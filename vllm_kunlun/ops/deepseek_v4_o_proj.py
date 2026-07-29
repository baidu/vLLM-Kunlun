import torch

from vllm_kunlun.ops.fp8 import dequantize_fp8_blocks

def deepseek_v4_bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: torch.nn.Module,
    wo_b: torch.nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """Correctness-only V4 output projection for Kunlun."""
    del einsum_recipe, tma_aligned_scales

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim >= nope_dim + rope_dim
    assert rope_dim % 2 == 0

    grouped = o.view(
        num_tokens, n_groups, heads_per_group, head_dim
    ).permute(1, 0, 2, 3)

    half_rope = rope_dim // 2
    cos_sin = torch.index_select(cos_sin_cache, 0, positions)
    cos = cos_sin[:, :half_rope].view(1, num_tokens, 1, half_rope)
    sin = cos_sin[:, half_rope:rope_dim].view(
        1, num_tokens, 1, half_rope
    )
    rope = grouped[..., nope_dim : nope_dim + rope_dim]
    rope_even = rope[..., 0::2]   # interleaved even indices
    rope_odd = rope[..., 1::2]    # interleaved odd indices
    inv_even = rope_even * cos + rope_odd * sin
    inv_odd = -rope_even * sin + rope_odd * cos
    inverse_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    grouped = torch.cat(
        (
            grouped[..., :nope_dim],
            inverse_rope,
            grouped[..., nope_dim + rope_dim :],
        ),
        dim=-1,
    )
    grouped = grouped.contiguous().view(n_groups, num_tokens, -1)

    # Cache dequantized weight on first call to avoid per-step CPU round-trip.
    # The FP8 weight never changes after loading, so this is safe.
    # For unquantized (BF16) attention (e.g. INT8 MoE model), weight_scale_inv
    # does not exist — use the weight as-is.
    if not hasattr(wo_a, '_bf16_weight_cache'):
        weight = wo_a.weight.data
        output_size_check = n_groups * o_lora_rank
        input_size_check = weight.shape[1]
        assert tuple(weight.shape) == (output_size_check, input_size_check), (
            f"weight shape {weight.shape} != ({output_size_check}, {input_size_check})"
        )
        if hasattr(wo_a, 'weight_scale_inv'):
            weight_scale = wo_a.weight_scale_inv.data
            if weight_scale.ndim < 2:
                weight_bf16 = weight.cpu().to(torch.bfloat16).to(weight.device) * weight_scale.float()
            else:
                weight_bf16 = dequantize_fp8_blocks(weight, weight_scale)
        else:
            weight_bf16 = weight.to(torch.bfloat16)
        wo_a._bf16_weight_cache = (
            weight_bf16
            .view(n_groups, o_lora_rank, input_size_check)
            .transpose(1, 2)
            .contiguous()
            .to(o.device)
        )
    output_size = n_groups * o_lora_rank
    input_size = grouped.shape[-1]
    weight_bf16 = wo_a._bf16_weight_cache
    projected = torch.bmm(grouped.to(torch.bfloat16), weight_bf16)
    projected = projected.permute(1, 0, 2).reshape(
        num_tokens, n_groups * o_lora_rank
    )
    return wo_b(projected)
