import torch
import xspeedgate_ops  # noqa


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


def pack_seq_native(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
) -> torch.Tensor:
    """Native replacement for vllm's ``pack_seq_triton``.

    Triton kernels cannot be launched on Kunlun XPU (the driver rejects
    load_binary with CUDA_ERROR_NOT_SUPPORTED), and the indexer hits this
    path whenever a decode batch has mixed query lengths — which is the norm
    with MTP, where spec-decode requests contribute 1 + num_spec_tokens
    tokens while freshly prefilled ones contribute 1.

    Args:
        x: [N, ...] where N is the total token count, laid out as the
            concatenation of each batch's tokens in order.
        lengths: [B] token count per batch.
        pad_value: fill for the padded tail of each row.

    Returns:
        [B, Lmax, ...] packed tensor.
    """
    B = lengths.numel()
    Lmax = int(lengths.max().item())
    N = x.shape[0]
    tail = x.shape[1:]

    starts = torch.cumsum(lengths, 0) - lengths  # [B]
    offsets = torch.arange(Lmax, device=x.device, dtype=starts.dtype)  # [Lmax]
    # Clamp so the gather stays in bounds; padded lanes are masked out below.
    idx = (starts[:, None] + offsets[None, :]).clamp_(max=max(N - 1, 0))
    gathered = x[idx.reshape(-1)].reshape((B, Lmax) + tail)

    valid = offsets[None, :] < lengths[:, None]  # [B, Lmax]
    valid = valid.reshape((B, Lmax) + (1,) * len(tail))
    return torch.where(valid, gathered, gathered.new_full((), pad_value))


def unpack_seq_native(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Native replacement for vllm's ``unpack_seq_triton``; inverse of
    :func:`pack_seq_native`.

    Args:
        packed_tensor: [B, Lmax, ...] as produced by ``pack_seq_native``.
        lengths: [B] token count per batch.

    Returns:
        [N, ...] where N == lengths.sum().
    """
    B, Lmax = packed_tensor.shape[:2]
    tail = packed_tensor.shape[2:]

    offsets = torch.arange(Lmax, device=packed_tensor.device, dtype=lengths.dtype)
    valid = offsets[None, :] < lengths[:, None]  # [B, Lmax]
    return packed_tensor.reshape((B * Lmax,) + tail)[valid.reshape(-1)]
