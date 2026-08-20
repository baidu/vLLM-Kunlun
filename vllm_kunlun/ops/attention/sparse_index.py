"""PyTorch fallbacks for the DSv4 top-K cache helpers.

Pure-torch equivalents of the upstream Triton kernels in
``vllm.models.deepseek_v4.common.ops.cache_utils`` (Kunlun Triton does not
support the constructs they use).
"""
from __future__ import annotations
import torch

_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


def compute_global_topk_indices_and_lens_pytorch(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local top-K indices to global KV cache slots; return slots + valid counts."""
    device = topk_indices.device
    num_tokens, topk = topk_indices.shape
    is_valid_idx = topk_indices >= 0

    safe_local = topk_indices.clamp(min=0).to(torch.int64)
    block_i = safe_local // block_size
    block_off = safe_local % block_size

    max_blocks = block_table.shape[1]
    block_i_safe = block_i.clamp(max=max_blocks - 1)
    req_idx = token_to_req_indices[:num_tokens].to(torch.int64).unsqueeze(1).expand(-1, topk)

    # Flat indexing: 2D advanced indexing crashes on Kunlun XPU.
    flat_idx = (req_idx * max_blocks + block_i_safe).reshape(-1)
    block_numbers = block_table.reshape(-1)[flat_idx].reshape(num_tokens, topk)
    slots = block_numbers * block_size + block_off
    slots_i32 = slots.to(torch.int32)
    neg_ones = torch.full_like(slots_i32, -1)
    global_slots = torch.where(is_valid_idx, slots_i32, neg_ones)

    counts = is_valid_idx.sum(dim=1).to(torch.int32)
    topk_lens = torch.where(
        is_valid_token.bool(),
        counts,
        torch.zeros_like(counts),
    )
    return global_slots, topk_lens


def combine_topk_swa_indices_pytorch(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate top-K slots and sliding-window slots into one combined index.

    Per-request loop; prefill chunk query lengths are small so the loop
    overhead is acceptable.
    """
    device = topk_indices.device
    num_tokens, _source_topk = topk_indices.shape
    num_reqs = int(seq_lens.shape[0])

    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
    ) * _SPARSE_PREFILL_TOPK_ALIGNMENT

    combined_indices = torch.full(
        (num_tokens, combined_topk), -1, dtype=torch.int32, device=device
    )
    combined_lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    if num_reqs == 0 or num_tokens == 0:
        return combined_indices, combined_lens

    # Rebase query_start_loc to chunk-local offsets.
    qs_cpu = (query_start_loc - query_start_loc[0]).to("cpu").tolist()
    seq_lens_cpu = seq_lens.to("cpu").tolist()
    gather_lens_cpu = gather_lens.to("cpu").tolist()

    topk_i32 = topk_indices.to(torch.int32)

    for b in range(num_reqs):
        q_start = int(qs_cpu[b])
        q_end = int(qs_cpu[b + 1])
        q_len = q_end - q_start
        if q_len <= 0:
            continue
        seq_len = int(seq_lens_cpu[b])
        gather_len = int(gather_lens_cpu[b])
        start_pos = seq_len - q_len
        gather_start = seq_len - gather_len
        positions = torch.arange(q_len, device=device) + start_pos  # int64
        tl_topk = torch.clamp((positions + 1) // compress_ratio, max=topk).to(torch.int32)
        tl_swa = torch.clamp(positions + 1, max=window_size).to(torch.int32)
        combined_lens[q_start:q_end] = tl_topk + tl_swa

        for t in range(q_len):
            token_idx = q_start + t
            n_top = int(tl_topk[t].item())
            n_swa = int(tl_swa[t].item())
            if n_top > 0:
                combined_indices[token_idx, :n_top] = topk_i32[token_idx, :n_top] + (M * b)
            if n_swa > 0:
                pos = int(positions[t].item())
                swa_offsets = torch.arange(n_swa, dtype=torch.int32, device=device)
                combined_indices[
                    token_idx, n_top : n_top + n_swa
                ] = M * b + N + swa_offsets + (pos - n_swa + 1 - gather_start)

    return combined_indices, combined_lens
