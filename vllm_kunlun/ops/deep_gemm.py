import torch
import xspeedgate_ops  # noqa

from vllm_kunlun.ops.fp8 import dequantize_fp8_blocks


def int8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    context_q_lens_xpu: torch.Tensor,
    context_q_lens_cpu: torch.Tensor,
    context_k_lens_xpu: torch.Tensor,
    context_k_lens_cpu: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    seq_len_q, seq_len_kv = q.shape[0], kv[0].shape[0]
    logits = torch.empty((seq_len_q, seq_len_kv), dtype=torch.float32, device=q.device)

    torch.ops._C.I8_mqa_logits(
        q=q,
        fused_kv_cache=kv,
        weights=weights,
        context_q_lens=(context_q_lens_cpu, context_q_lens_xpu),
        context_k_lens=(context_k_lens_cpu, context_k_lens_xpu),
        logits=logits,
        clean_logits=True,
        use_xfa_boost=False,
    )

    # mask参考 https://github.com/vllm-project/vllm/blob/v0.11.0/tests/kernels/attention/test_deepgemm_attention.py 的_ref_fp8_mqa_logits函数的实现
    torch.ops.xspeedgate_ops.mask_for_I8_mqa_logits(
        seq_len_kv=seq_len_kv,
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
        logits=logits,
    )

    return logits


def int8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    context_lens_cpu: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    batch_size, next_n, _, D = q_fp8.shape
    num_blocks, block_size, _, _ = kv_cache_fp8.shape

    kv_cache_fp8 = kv_cache_fp8.view(num_blocks, -1)
    k_val = kv_cache_fp8[:, : block_size * D].view(torch.int8)
    k_val = k_val.view(-1, block_size, 1, D)

    block_indices = block_tables.flatten()
    k_scale = (
        kv_cache_fp8[block_indices, block_size * D :].view(-1, 4).view(torch.float32)
    )
    k_scale = k_scale.view(-1, max_model_len)
    kv_cache = [k_val, k_scale]

    weights = weights.view(batch_size, next_n, -1)

    logits = torch.empty(
        (batch_size, next_n, max_model_len), dtype=torch.float32, device=q_fp8.device
    )

    torch.ops._C.I8_paged_mqa_logits(
        q=q_fp8,
        fused_kv_cache=kv_cache,
        weights=weights,
        context_lens=[context_lens_cpu, context_lens],
        block_table=block_tables,
        max_context_len=max_model_len,
        clean_logits=True,
        out=logits,
        use_xfa_boost=False,
    )
    logits = logits.view(-1, max_model_len)
    return logits


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
    if not hasattr(wo_a, '_bf16_weight_cache'):
        weight = wo_a.weight.data
        weight_scale = wo_a.weight_scale_inv.data
        output_size_check = n_groups * o_lora_rank
        input_size_check = weight.shape[1]
        assert tuple(weight.shape) == (output_size_check, input_size_check), (
            f"weight shape {weight.shape} != ({output_size_check}, {input_size_check})"
        )
        if weight_scale.ndim < 2:
            weight_bf16 = weight.cpu().to(torch.bfloat16).to(weight.device) * weight_scale.float()
        else:
            weight_bf16 = dequantize_fp8_blocks(weight, weight_scale)
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
