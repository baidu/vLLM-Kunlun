# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py
from typing import Optional, Tuple

import kunlun_ops
import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401

        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False

if current_platform.is_cuda():
    try:
        import vllm._flashmla_extension_C  # noqa: F401

        _flashmla_extension_C_AVAILABLE = True
    except ImportError:
        _flashmla_extension_C_AVAILABLE = False
else:
    _flashmla_extension_C_AVAILABLE = False


def is_flashmla_supported() -> Tuple[bool, Optional[str]]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    return True, None


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int = 1,
    num_heads_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Returns:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    # return flash_mla_cuda.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)
    cache_seqlens_cpu = cache_seqlens.cpu()
    return cache_seqlens_cpu, cache_seqlens


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    head_dim_v: int = 512,
    tile_scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    # V4 kwargs
    topk_length: Optional[torch.Tensor] = None,
    attn_sink=None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kunlun flash_mla_with_kvcache for DeepSeek-V4.
    Two paths:
    - Legacy V3: block_table provided -> kunlun_ops.paged_attention
    - V4: indices provided -> PyTorch naive MLA attention
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # ---- Legacy V3 path ----
    if block_table is not None and indices is None:
        output = torch.ones(
            q.size(0), q.size(1), q.size(2), head_dim_v, dtype=q.dtype, device=q.device
        )
        kv_lora_rank = head_dim_v
        qk_rope_head_dim = q.size(3) - head_dim_v
        head_dim = k_cache.shape[3]
        page_block_size = k_cache.shape[1]
        k_cache_view = k_cache.view(-1, 1, page_block_size, head_dim)

        kunlun_ops.paged_attention(
            output, q, k_cache_view, None, block_table,
            tile_scheduler_metadata, num_splits,
            False, causal, -1,
            kv_lora_rank, qk_rope_head_dim, softmax_scale, q_r=q,
        )
        return output, None

    # ---- V4 path: Vectorized sparse MLA (BF16 cache only; no per-token loop) ----
    batch_size = q.shape[0]
    seq_q = q.shape[1]
    num_heads = q.shape[2]
    d_qk = q.shape[3]
    kv_lora_rank = head_dim_v  # 512
    qk_rope_head_dim = d_qk - kv_lora_rank  # 64 (or 0 if d_qk==512)

    if out is not None:
        output = out
    else:
        output = torch.zeros(
            batch_size, seq_q, num_heads, head_dim_v,
            dtype=q.dtype, device=q.device,
        )

    if indices is None or topk_length is None:
        # Returning zeros here silently produces wrong attention output for the
        # rest of the run (a single warning is easy to miss). Fail loudly.
        raise RuntimeError(
            "[KunlunFlashMLA] V4 sparse MLA needs both `indices` and "
            f"`topk_length` (got indices={indices is not None}, "
            f"topk_length={topk_length is not None}); refusing to return zeros."
        )

    # Flatten paged caches to [total_slots, dim] bf16 (device-side, no D2H)
    def _flatten_cache(cache_tensor, target_dim):
        total = cache_tensor.shape[0] * cache_tensor.shape[1]
        flat = cache_tensor.reshape(total, -1)

        if cache_tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
            flat_kv = flat.to(torch.bfloat16)
            if flat_kv.shape[1] > target_dim:
                return flat_kv[:, :target_dim]
            elif flat_kv.shape[1] < target_dim:
                pad = torch.zeros(total, target_dim - flat_kv.shape[1],
                                  dtype=torch.bfloat16, device=flat.device)
                return torch.cat([flat_kv, pad], dim=1)
            return flat_kv
        elif cache_tensor.dtype == torch.uint8:
            # FP8 e4m3fn dequant (vectorized, no Python loop)
            fp8_data = flat[:, :target_dim].contiguous()
            sign = ((fp8_data >> 7) & 1).to(torch.int32)
            exp_bits = ((fp8_data >> 3) & 0x0F).to(torch.int32)
            mant_bits = (fp8_data & 0x07).to(torch.int32)
            is_normal = (exp_bits > 0)
            is_zero = (fp8_data == 0) | (fp8_data == 0x80)
            mantissa_n = (8 + mant_bits).float() / 8.0
            exponent_n = (exp_bits - 7).float()
            val_normal = mantissa_n * torch.pow(2.0, exponent_n)
            val_subnormal = mant_bits.float() / 8.0 * (2.0 ** -6.0)
            result = torch.where(is_normal, val_normal, val_subnormal)
            result = torch.where(is_zero, torch.zeros_like(result), result)
            result = result * (1 - 2 * sign.float())
            return result.to(torch.bfloat16)
        else:
            # Unknown dtype: try to cast
            flat_kv = flat.to(torch.bfloat16)
            if flat_kv.shape[1] > target_dim:
                return flat_kv[:, :target_dim]
            elif flat_kv.shape[1] < target_dim:
                pad = torch.zeros(total, target_dim - flat_kv.shape[1],
                                  dtype=torch.bfloat16, device=flat.device)
                return torch.cat([flat_kv, pad], dim=1)
            return flat_kv

    flat_swa = _flatten_cache(k_cache, d_qk)
    swa_total = flat_swa.shape[0]

    flat_comp = None
    if extra_k_cache is not None:
        flat_comp = _flatten_cache(extra_k_cache, d_qk)

    # Parse indices: [B, 1, N] -> [B, N]
    swa_idx = indices.squeeze(1) if indices.dim() == 3 else indices
    comp_idx = None
    if extra_indices_in_kvcache is not None:
        comp_idx = extra_indices_in_kvcache.squeeze(1) if extra_indices_in_kvcache.dim() == 3 else extra_indices_in_kvcache

    # Vectorized per-batch gather + attention (no .item(), no per-token loop)
    # For decode: seq_q=1, so we process q[:, 0] directly
    q_flat = q[:, 0].float()  # [B, H, d_qk]
    q_c = q_flat[:, :, :kv_lora_rank]  # [B, H, 512]
    q_r = q_flat[:, :, kv_lora_rank:kv_lora_rank + qk_rope_head_dim] if qk_rope_head_dim > 0 else None  # [B, H, 64]

    for b in range(batch_size):
        # Gather using tensor indexing (no .item() for lengths - use full index width)
        swa_len = topk_length[b]  # scalar tensor on device
        parts = []

        # SWA gather: use full swa_idx width (padded with 0s, clamped)
        swa_indices_b = swa_idx[b].long().clamp(0, swa_total - 1)
        parts.append(flat_swa[swa_indices_b])  # [swa_width, d_qk]

        # Compressed gather
        if flat_comp is not None and comp_idx is not None:
            comp_indices_b = comp_idx[b].long().clamp(0, flat_comp.shape[0] - 1)
            parts.append(flat_comp[comp_indices_b])  # [comp_width, d_qk]

        # Concatenate: [total_kv, d_qk]
        kv_all = torch.cat(parts, dim=0).float()
        kv_c = kv_all[:, :kv_lora_rank]  # [K, 512]
        kv_r = kv_all[:, kv_lora_rank:kv_lora_rank + qk_rope_head_dim] if qk_rope_head_dim > 0 else None

        # MLA score: [H, K] = [H, 512] @ [512, K] + [H, 64] @ [64, K]
        scores = torch.mm(q_c[b], kv_c.T)  # [H, K]
        if q_r is not None and kv_r is not None:
            scores = scores + torch.mm(q_r[b], kv_r.T)
        scores = scores * softmax_scale

        # Softmax + output
        attn_weights = torch.softmax(scores, dim=-1)  # [H, K]
        attn_out = torch.mm(attn_weights, kv_c)  # [H, 512]

        # Attention sink
        if attn_sink is not None:
            lse = torch.logsumexp(scores, dim=-1)  # [H]
            sink_scale = torch.sigmoid(lse - attn_sink[:num_heads].float())
            attn_out = attn_out * sink_scale.unsqueeze(-1)

        output[b, 0, :, :kv_lora_rank] = attn_out.to(output.dtype)

    return output, None


def kunlun_flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_seqlens_cpu: torch.Tensor,
    head_dim_v: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    max_seq_kv: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_tokens_kv, head_dim).
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.
        is_fp8_kvcache: bool. Whether the k_cache and v_cache are in fp8 format.
        indices: (batch_size, seq_len_q, topk), torch.int32. If not None, sparse attention will be enabled, and only tokens in the `indices` array will be attended to. Invalid indices should be set to -1 or numbers >= total_seq_len_kv.
        max_seq_kv: seq中最大的kv长度

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        max_logits:  (batch_size, seq_len_q, num_heads_q), torch.float32.
        p_sums:  (batch_size, seq_len_q, num_heads_q), torch.float32.
    """
    assert not is_fp8_kvcache, "By now, the kernel does not support uint8 kv cache."
    assert (
        q.shape[1] <= 2
    ), "kunlun_ops.fwd_kvcache_mla only support seq_len_q <= 2 for now."
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if indices is not None:
        # NOTE (zyongye): sparse attention is also causal
        # since it only attend to the tokens before
        # but here `causal` should not be specified
        assert not causal, "causal must be `false` if sparse attention is enabled."

    batch_size, seq_len_q, num_heads_q, head_dim = q.shape
    kv_lora_rank = head_dim_v

    out = torch.zeros(
        [batch_size, seq_len_q, num_heads_q, kv_lora_rank],
        dtype=q.dtype,
        device=q.device,
    )
    max_logits = torch.zeros(
        [batch_size, seq_len_q, num_heads_q], dtype=torch.float32, device=q.device
    )
    p_sums = torch.zeros(
        [batch_size, seq_len_q, num_heads_q], dtype=torch.float32, device=q.device
    )

    torch.ops._C.fwd_kvcache_mla(
        q_c=q,
        kv_cache=k_cache,
        indices=indices,
        kv_lod_cpu=cache_seqlens_cpu,
        max_seq_kv=max_seq_kv,
        softmax_scale=softmax_scale,
        # q_r=q_r,
        # pe_cache=pe_cache,
        out=out,
        max_logits=max_logits,
        p_sums=p_sums,
        kv_lod_xpu=cache_seqlens,
    )

    return out, max_logits, p_sums


def flash_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    q_lod_xpu: Optional[torch.Tensor] = None,
    q_lod_cpu: Optional[torch.Tensor] = None,
    d_v: int = 512,
    # V4 new kwargs
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
    - q: [s_q, h_q, d_qk], bfloat16
    - kv: [s_kv, d_qk], bfloat16
    - indices: [s_q, h_kv, topk], int32.
        Invalid indices should be set to -1 or numbers >= s_kv
    - sm_scale: float
    - q_lod_xpu: [batch+1], int32, q的每个seq长度的累加信息, 长度为batch_num + 1 (为空则表示q定长).
    - d_v: The dimension of value vectors. Can only be 512

    Returns:
    - (output, max_logits, lse)
        About the definition of output,
        max_logits and lse, please refer to README.md
    - output: [s_q, h_q, d_v], bfloat16
    - max_logits:  [s_q, h_q], float
    - lse: [s_q, h_q], float, 2-based log-sum-exp
    """
    s_q, h_q, d_qk = q.shape

    # [kunlun-patch] 3D-kv naive PyTorch MLA fallback
    #
    # DSv4 nvidia._forward_prefill calls this wrapper with
    #   kv     = kv_workspace.view(-1, 1, d_qk)   # 3-D
    #   indices= combined_indices.unsqueeze(1)    # [s_q, 1, topk]
    #   topk_length = combined_lens
    # The kunlun sparse_prefill_fwd_opt kernel asserts kv.dim() == 2 and
    # historically rejected this V4 layout on Kunlun, so we fall back to a
    # naive PyTorch MLA attention over the gathered bf16 workspace. This
    # matches the source-patched implementation that produced tokens on the
    # old pod. The 2-D fast path (V3.2 indexer sparse prefill) is unchanged.
    if kv.dim() == 3:
        import torch as _torch
        d_qk_v = q.shape[-1]
        s_q_v = q.shape[0]
        h_q_v = q.shape[1]
        if out is None:
            out = _torch.zeros(
                [s_q_v, h_q_v, d_v], dtype=q.dtype, device=q.device
            )
        kv_flat = kv.reshape(-1, d_qk_v)
        idx2d = indices.reshape(s_q_v, -1)
        kv_lora_rank = d_v
        rope_dim = d_qk_v - kv_lora_rank
        lens = topk_length
        for _t in range(s_q_v):
            valid = int(lens[_t].item()) if lens is not None else idx2d.shape[1]
            if valid <= 0:
                continue
            vi = idx2d[_t, :valid].long().clamp(0, kv_flat.shape[0] - 1)
            kvg = kv_flat[vi].float()
            kv_c = kvg[:, :kv_lora_rank]
            kv_r = kvg[:, kv_lora_rank:kv_lora_rank + rope_dim]
            q_tok = q[_t].float()
            q_c = q_tok[:, :kv_lora_rank]
            q_r = q_tok[:, kv_lora_rank:kv_lora_rank + rope_dim]
            scores = (q_c @ kv_c.T + q_r @ kv_r.T) * sm_scale
            w = _torch.softmax(scores, dim=-1)
            ao = w @ kv_c
            out[_t, :, :kv_lora_rank] = ao.to(out.dtype)
        max_logits = _torch.zeros(
            [s_q_v, h_q_v], dtype=_torch.float32, device=q.device
        )
        lse = _torch.zeros(
            [s_q_v, h_q_v], dtype=_torch.float32, device=q.device
        )
        return out, max_logits, lse

    if out is None:
        out = torch.zeros([s_q, h_q, d_v], dtype=q.dtype, device=q.device)
    max_logits = torch.zeros([s_q, h_q], dtype=torch.float32, device=q.device)
    lse = torch.zeros([s_q, h_q], dtype=torch.float32, device=q.device)

    # If q_lod not provided (V4 path), create a simple [0, s_q] lod
    if q_lod_xpu is None:
        q_lod_cpu = torch.tensor([0, s_q], dtype=torch.int32)
        q_lod_xpu = q_lod_cpu.to(q.device)
    if q_lod_cpu is None:
        q_lod_cpu = q_lod_xpu.cpu()

    torch.ops._C.sparse_prefill_fwd_opt(
        q=q,
        kv=kv,
        indices=indices,
        qlod_cpu=q_lod_cpu,
        qlod_xpu=q_lod_xpu,
        kvlod_cpu=q_lod_cpu,
        kvlod_xpu=q_lod_xpu,
        sm_scale=sm_scale,
        d_v=d_v,
        is_causal=True,  # aiak这个值为true，这是为啥
        out=out,
        max_logits=max_logits,
        lse=lse,
    )

    # NOTE: Compared with torch.ops._flashmla_C.sparse_prefill_fwd,
    # out_scale = 1 / math.log2(math.e)
    # gpu_max_logits * out_scale = kunlun_lse
    # gpu_lse * out_scale = kunlun_lse
    lse = lse.float()
    if isinstance(attn_sink, torch.Tensor):
        sink = attn_sink[:h_q].to(device=lse.device, dtype=torch.float32)
        sink_scale = torch.sigmoid(lse - sink.unsqueeze(0))
        out.mul_(sink_scale.unsqueeze(-1).to(out.dtype))
    return out, max_logits.float(), lse


#
# TODO: Add fake functions
#
# @register_fake("_flashmla_C::get_mla_metadata")
# def _get_mla_metadata_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#
# @register_fake("_flashmla_C::fwd_kvcache_mla")
# def _fwd_kvcache_mla_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#
