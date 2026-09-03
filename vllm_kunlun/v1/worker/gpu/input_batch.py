# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun torch-native replacement for ``vllm.v1.worker.gpu.input_batch``.

Reuses the upstream ``InputBuffers`` / ``InputBatch`` dataclasses (loaded via
``load_upstream``) and reimplements the seven Triton-backed module functions
with torch-native equivalents. The functions operate on jagged per-request
segments; the correctness-first implementation loops per request in Python
(``num_reqs`` is small, typically <= a few hundred) and pulls the small
metadata index tensors to host once per call. Optimizing the hot ones with
``kunlun_ops`` is a later step.
"""

import torch

from vllm_kunlun.v1.worker.gpu._upstream import load_upstream

_up = load_upstream("vllm.v1.worker.gpu.input_batch")

InputBuffers = _up.InputBuffers
InputBatch = _up.InputBatch


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    idx = idx_mapping.tolist()
    qsl = query_start_loc.tolist()
    pl = prefill_len.tolist()
    nc = num_computed_tokens.tolist()
    for b in range(num_reqs):
        rs = idx[b]
        prefill = pl[rs]
        computed = nc[rs]
        if computed >= prefill:
            # Not prefilling this step.
            continue
        qs = qsl[b]
        qe = qsl[b + 1]
        query_len = qe - qs
        input_ids[qs:qe] = all_token_ids[rs, computed : computed + query_len]
        next_pos = computed + query_len
        if next_pos < prefill:
            next_prefill_tokens[rs] = all_token_ids[rs, next_pos]


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    # Pad unused seq_lens as 0 (full CUDA graph parity).
    seq_lens[num_reqs:] = 0
    idx = idx_mapping.tolist()
    qsl = query_start_loc.tolist()
    nc = num_computed_tokens.tolist()
    for b in range(num_reqs):
        computed = nc[idx[b]]
        qs = qsl[b]
        qe = qsl[b + 1]
        query_len = qe - qs
        seq_lens[b] = computed + query_len
        if query_len > 0:
            pos[qs:qe] = torch.arange(
                computed, computed + query_len, dtype=pos.dtype, device=pos.device
            )


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
    num_new_sampled_tokens: int = 1,
) -> torch.Tensor:
    assert num_new_sampled_tokens in (
        0,
        1,
    ), f"num_new_sampled_tokens must be 0 or 1, got {num_new_sampled_tokens}"
    num_reqs = idx_mapping.shape[0]
    device = input_ids.device
    logits_indices = torch.empty(num_logits, dtype=torch.int64, device=device)

    idx = idx_mapping.tolist()
    qsl = query_start_loc.tolist()
    cnl = cu_num_logits.tolist()
    sl = seq_lens.tolist()
    pl = prefill_len.tolist()
    for b in range(num_reqs):
        rs = idx[b]
        cl_start = cnl[b]
        cl_end = cnl[b + 1]
        nlog = cl_end - cl_start
        query_end = qsl[b + 1]
        logits_start = query_end - nlog
        logits_indices[cl_start:cl_end] = torch.arange(
            logits_start, logits_start + nlog, dtype=torch.int64, device=device
        )

        seq_len = sl[b]
        prefill = pl[rs]
        if seq_len <= prefill:
            # Prefill tokens: no sampled/draft tokens to splice in.
            continue

        num_draft = nlog - num_new_sampled_tokens
        first_logit_seq_pos = seq_len - nlog
        if num_new_sampled_tokens > 0 and first_logit_seq_pos >= prefill:
            input_ids[logits_start] = last_sampled_tokens[rs]
        if num_draft > 0:
            input_ids[query_end - num_draft : query_end] = draft_tokens[rs, :num_draft]
    return logits_indices


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    idx_long = idx_mapping.to(torch.long)
    prefill = prefill_len[idx_long]
    is_chunked = seq_lens[:num_reqs] < prefill
    num_logits = cu_num_logits[1 : num_reqs + 1] - cu_num_logits[:num_reqs]

    zero = torch.zeros_like(num_sampled[:num_reqs])
    num_sampled[:num_reqs] = torch.where(is_chunked, zero, num_sampled[:num_reqs])
    num_rejected = torch.empty_like(num_sampled)
    rejected = num_logits - num_sampled[:num_reqs]
    num_rejected[:num_reqs] = torch.where(
        is_chunked, torch.zeros_like(rejected), rejected
    )
    return num_sampled, num_rejected


def post_update(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor | None,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    idx = idx_mapping.tolist()
    ns = num_sampled.tolist()
    nr = num_rejected.tolist()
    tl_list = total_len.tolist()
    qsl = query_start_loc.tolist() if query_start_loc is not None else None
    for b in range(num_reqs):
        rs = idx[b]
        if rs < 0:
            # Filtered row.
            continue
        num_s = ns[b]
        if num_s > 0:
            base = tl_list[rs]
            last_sampled_tokens[rs] = sampled_tokens[b, num_s - 1]
            tokens = sampled_tokens[b, :num_s]
            all_token_ids[rs, base : base + num_s] = tokens
            total_len[rs] = base + num_s
            if output_bin_counts is not None:
                toks = tokens.to(torch.long)
                output_bin_counts[rs].scatter_add_(
                    0, toks, torch.ones_like(toks, dtype=output_bin_counts.dtype)
                )
        query_len = 0 if qsl is None else (qsl[b + 1] - qsl[b])
        computed_delta = query_len - nr[b]
        if computed_delta != 0:
            num_computed_tokens[rs] += computed_delta


def post_update_num_computed_tokens(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    idx_long = idx_mapping.to(torch.long)
    query_len = (query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]).to(
        num_computed_tokens.dtype
    )
    # Each request appears once per batch, so plain indexed add is correct.
    num_computed_tokens[idx_long] += query_len


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    device = idx_mapping.device
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(total_num_logits, dtype=torch.int32, device=device)
    cnl = cu_num_logits.tolist()
    idx = idx_mapping.tolist()
    for r in range(num_reqs):
        s = cnl[r]
        e = cnl[r + 1]
        n = e - s
        if n <= 0:
            continue
        expanded_idx_mapping[s:e] = idx[r]
        expanded_local_pos[s:e] = torch.arange(n, dtype=torch.int32, device=device)
    return expanded_idx_mapping, expanded_local_pos
