import torch
import kunlun_ops

def int8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
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
    logits = torch.empty((q.shape[0], kv[0].shape[0]), dtype=torch.float32, device=q.device)
    context_q_lens_xpu = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=cu_seqlen_ks.device)
    context_k_lens_xpu = torch.tensor([0, kv[0].shape[0]], dtype=torch.int32, device=cu_seqlen_ks.device)

    kunlun_ops.I8_mqa_logits(
            q=q,
            fused_kv_cache=kv,
            weights=weights,
            context_q_lens=(context_q_lens_xpu.cpu(), context_q_lens_xpu),
            context_k_lens=(context_k_lens_xpu.cpu(), context_k_lens_xpu),
            logits=logits,
            clean_logits=True,
            use_xfa_boost=False,
        ) 
    seq_len_kv = kv[0].shape[0]
    # mask参考 https://github.com/vllm-project/vllm/blob/v0.11.0/tests/kernels/attention/test_deepgemm_attention.py 的_ref_fp8_mqa_logits函数的实现
    mask_lo = (torch.arange(0, seq_len_kv, device=cu_seqlen_ks.device)[None, :]
               >= cu_seqlen_ks[:, None])
    mask_hi = (torch.arange(0, seq_len_kv, device=cu_seqlen_ke.device)[None, :]
               < cu_seqlen_ke[:, None])
    mask = mask_lo & mask_hi
    logits = logits.masked_fill(~mask, float("-inf"))

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
    
    kv_cache_fp8=kv_cache_fp8.view(num_blocks, -1)
    k_val = kv_cache_fp8[:,:block_size*D].view(torch.int8)
    k_val = k_val.view(-1,block_size, 1, D)
    k_scale_list = []
    for block_tables_idx in range(block_tables.shape[0]):
        k_scale_item = kv_cache_fp8[block_tables[block_tables_idx], block_size *
                                       D:].view(-1, 4)
        k_scale_list.append(k_scale_item)
    k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).view(-1,max_model_len)
    kv_cache = [k_val, k_scale]

    weights = weights.view(batch_size,next_n,-1)
            
    logits = torch.empty((batch_size, next_n, max_model_len), dtype=torch.float32, device=q_fp8.device)

    kunlun_ops.I8_paged_mqa_logits(
                q=q_fp8,
                fused_kv_cache=kv_cache,
                weights=weights,
                context_lens=[context_lens_cpu, context_lens],
                block_table=block_tables,
                max_context_len=max_model_len,
                clean_logits=True,
                out=logits,
                use_xfa_boost=False
            )
    logits = logits.view(-1, max_model_len)
    return logits