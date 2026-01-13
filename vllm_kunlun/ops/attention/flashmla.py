# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py
from typing import Optional, Tuple

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
import kunlun_ops

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
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
        softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    softmax_lse = None
    out = torch.ones(q.size(0), q.size(1), q.size(2), head_dim_v, dtype= q.dtype, device=q.device)
    kv_lora_rank = head_dim_v
    qk_rope_head_dim = q.size(3) - head_dim_v
    head_dim = k_cache.shape[3]
    page_block_size = k_cache.shape[1]
    k_cache = k_cache.view(-1, 1, page_block_size, head_dim)
    
    # todo: optimize memcp
    # q_c = q[..., : kv_lora_rank].contiguous()
    # q_r = q[..., kv_lora_rank :].contiguous()
    
    is_context = False
    vo_head_dim = -1
    
    kunlun_ops.paged_attention(out,
                               q,
                               k_cache, None,
                               block_table,
                               tile_scheduler_metadata, # context_lens_cpu
                               num_splits,              # context_lens_xpu
                               is_context,
                               causal,
                               vo_head_dim,
                               kv_lora_rank,
                               qk_rope_head_dim,
                               softmax_scale,
                               q_r=q)
    return out, softmax_lse
        
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
    assert q.shape[1] <= 2, "kunlun_ops.fwd_kvcache_mla only support seq_len_q <= 2 for now."
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if indices is not None:
        # NOTE (zyongye): sparse attention is also causal
        # since it only attend to the tokens before
        # but here `causal` should not be specified
        assert not causal, \
            "causal must be `false` if sparse attention is enabled."
    
    q_r, pe_cache = None, None # 当q_r和pe_cache为空时，为packed模式
    batch_size, seq_len_q, num_heads_q, head_dim = q.shape
    kv_lora_rank = head_dim_v
    rope_head_dim = head_dim - kv_lora_rank
    
    out = torch.zeros([batch_size, seq_len_q, num_heads_q, kv_lora_rank],
                        dtype=q.dtype, device=q.device)
    max_logits = torch.zeros([batch_size, seq_len_q, num_heads_q],
                                dtype=torch.float32, device=q.device)
    p_sums = torch.zeros([batch_size, seq_len_q, num_heads_q],
                            dtype=torch.float32, device=q.device)

    kunlun_ops.fwd_kvcache_mla(
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
    q_lod_xpu: torch.Tensor,
    q_lod_cpu: torch.Tensor,
    d_v: int = 512,
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
    
    out = torch.zeros([s_q, h_q, d_v], dtype=q.dtype, device=q.device)
    max_logits = torch.zeros([s_q, h_q], dtype=torch.float32, device=q.device)
    lse = torch.zeros([s_q, h_q], dtype=torch.float32, device=q.device)

    kunlun_ops.sparse_prefill_fwd_opt(
        q=q,
        kv=kv,
        indices=indices,
        qlod_cpu=q_lod_cpu,
        qlod_xpu=q_lod_xpu,
        kvlod_cpu=q_lod_cpu,
        kvlod_xpu=q_lod_xpu,
        sm_scale=sm_scale,
        d_v=d_v,
        is_causal=True, #aiak这个值为true，这是为啥
        out=out,
        max_logits=max_logits,
        lse=lse,
    )
    
    # NOTE: Compared with torch.ops._flashmla_C.sparse_prefill_fwd, 
    # out_scale = 1 / math.log2(math.e)
    # gpu_max_logits * out_scale = kunlun_lse
    # gpu_lse * out_scale = kunlun_lse
    return out, max_logits, lse


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