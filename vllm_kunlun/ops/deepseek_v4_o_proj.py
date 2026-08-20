"""DeepSeek-V4 attention output projection on Kunlun XPU.

Computes the projection as inverse-GPTJ-RoPE followed by a grouped BF16
bmm (``wo_a``) plus the ``wo_b`` tail; FP8 weights are dequantized once
and cached on the layer.
"""
import torch

from vllm_kunlun.ops.fp8 import dequantize_fp8_blocks

try:
    import kunlun_ops
except Exception:
    kunlun_ops = None

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

    # Native inverse GPT-J RoPE: the kernel only does the forward rotation,
    # so we pass a cos_sin table with the sine half negated to obtain the
    # inverse. Falls back to the torch formula when unavailable or failing.
    _warned_inv_rope_fallback = getattr(deepseek_v4_bf16_o_proj,
                                        "_warned_inv_rope_fallback", False)
    use_native_rotary = (
        kunlun_ops is not None
        and hasattr(kunlun_ops, "dpsk_decode_rotary_embedding_v3")
        and head_dim >= nope_dim + rope_dim
        
    )
    if use_native_rotary:
        try:
            # Key the cache by table identity, not (shape, dtype): V4 builds
            # two same-shaped tables (rope_theta / compress_rope_theta) and a
            # shape key would mix them up.
            caches = getattr(deepseek_v4_bf16_o_proj, "_global_inv_rope_caches", None)
            if caches is None:
                caches = {}
                deepseek_v4_bf16_o_proj._global_inv_rope_caches = caches
            cache_id = (
                cos_sin_cache.data_ptr(),
                tuple(cos_sin_cache.shape),
                str(cos_sin_cache.dtype),
            )
            inv_cache = caches.get(cache_id)
            if inv_cache is None or inv_cache.device != o.device:
                inv_cos_sin = cos_sin_cache.float().detach().clone()
                if inv_cos_sin.shape[-1] < half_rope * 2:
                    raise RuntimeError(f"rotary cache last dim {inv_cos_sin.shape[-1]} too small for rope dim {rope_dim}")
                neg_region_end = min(half_rope * 2, inv_cos_sin.shape[-1])
                inv_cos_sin[:, half_rope:neg_region_end].neg_()
                inv_cache = inv_cos_sin.contiguous().to(o.device)
                caches[cache_id] = inv_cache

            all_heads_3d = o.view(
                num_tokens, n_groups, heads_per_group, head_dim
            ).reshape(num_tokens, num_heads, head_dim)
            positions_xpu = (
                positions.to(o.device)
                if positions.device != o.device else positions
            )
            # One position per token; the kernel broadcasts it across heads.
            rotated_out = all_heads_3d.clone().contiguous()
            status_obj = kunlun_ops.dpsk_decode_rotary_embedding_v3(
                positions=positions_xpu.long(),
                cos_sin_cache=inv_cache,
                in_query=all_heads_3d.contiguous(),
                in_key=all_heads_3d.detach(),             # key side content unused by caller
                out_query=rotated_out,
                in_query_offset=nope_dim,
                in_key_offset=0,
                out_query_offset=nope_dim,
            )
            status_int = int(status_obj) if status_obj is not None else 0
            assert status_int == 0, f"dpsk_decode_rotary_embedding_v3 failed status={status_int}"
            grouped = (
    rotated_out.view_as(o)
        .to(torch.bfloat16)
        .view(num_tokens, n_groups, heads_per_group, head_dim)
        .permute(1, 0, 2, 3)
        .contiguous().view(n_groups, num_tokens, -1)
)
        except Exception as e:  # noqa: BLE001
            use_native_rotary = False
            if not _warned_inv_rope_fallback:
                import logging
                logging.getLogger("vllm_kunlun.ops.deepseek_v4_o_proj").warning(
                    "native dpsk_decode_rotary_embedding_v3 O-proj inverse-RoPE "
                    "failed (%s), falling back to torch formula", str(e)
                )
                deepseek_v4_bf16_o_proj._warned_inv_rope_fallback = True

    if not use_native_rotary:
        # Original elementwise fallback using selected rows from cos_sin_cache.
        cos_sin = torch.index_select(cos_sin_cache, 0, positions)
        cos = cos_sin[:, :half_rope].view(1, num_tokens, 1, half_rope)
        sin = cos_sin[:, half_rope:rope_dim].view(
            1, num_tokens, 1, half_rope
        )
        rope = grouped[..., nope_dim : nope_dim + rope_dim]
        rope_even = rope[..., 0::2]
        rope_odd = rope[..., 1::2]
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
        ).contiguous().view(n_groups, num_tokens, -1)

    # Dequantize once and cache: the weight never changes after loading.
    # Unquantized (BF16) layers have no weight_scale_inv; use the weight as-is.
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
