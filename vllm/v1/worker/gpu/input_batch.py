# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils import random_uuid


class InputBuffers:
    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device

        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # DCP: per-request local seq_lens buffer
        self.dcp_local_seq_lens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )


@dataclass
class InputBatch:
    # batch_idx -> req_id
    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    # Identical to idx_mapping except for spec decoding.
    expanded_idx_mapping: torch.Tensor
    # [total_num_logits] position within request for each logit
    expanded_local_pos: torch.Tensor

    # [num_reqs]
    # batch_idx -> num_scheduled_tokens
    num_scheduled_tokens: np.ndarray
    # sum(num_scheduled_tokens)
    num_tokens: int
    num_tokens_after_padding: int
    # Sum of draft tokens scheduled across requests.
    num_draft_tokens: int
    # [num_reqs] number of draft tokens scheduled for each request, if any.
    num_draft_tokens_per_req: np.ndarray | None

    # [num_reqs + 1]
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    # [num_reqs]
    seq_lens: torch.Tensor
    # [num_reqs] CPU upper bound on seq_lens (see CommonAttentionMetadata).
    seq_lens_cpu_upper_bound: torch.Tensor
    # [num_reqs]
    dcp_local_seq_lens: torch.Tensor | None
    # [num_reqs]
    num_computed_tokens_np: np.ndarray
    # [num_reqs]
    prefill_len_np: np.ndarray
    # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray
    # [num_reqs] CPU bool array == (num_computed_prefill_tokens_np < prefill_len_np).
    is_prefilling_np: np.ndarray

    # [num_reqs] only populated when pipeline parallelism is enabled.
    max_seq_len_np: np.ndarray | None

    # [num_tokens_after_padding]
    input_ids: torch.Tensor
    # [num_tokens_after_padding]
    positions: torch.Tensor
    # [num_tokens_after_padding]
    is_padding: torch.Tensor

    # [total_num_logits]
    logits_indices: torch.Tensor
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor
    cu_num_logits_np: np.ndarray

    # Whether any requests in batch use structured output.
    has_structured_output_reqs: bool

    # [num_reqs] per-request prompt length, only populated for R-SWA.
    prompt_lens: torch.Tensor | None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: InputBuffers,
    ) -> "InputBatch":
        assert 0 < num_reqs <= num_tokens
        device = input_buffers.device

        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        num_scheduled_tokens = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
        num_scheduled_tokens[-1] += num_tokens % num_reqs
        assert int(num_scheduled_tokens.sum()) == num_tokens

        # seq_len equals to query_len
        input_buffers.seq_lens[:num_reqs] = num_tokens // num_reqs
        input_buffers.seq_lens[num_reqs - 1] += num_tokens % num_reqs
        # Pad for full CUDA graph mode.
        input_buffers.seq_lens[num_reqs:] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs]

        query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        input_buffers.query_start_loc[:1] = 0
        torch.cumsum(
            seq_lens, dim=0, out=input_buffers.query_start_loc[1 : num_reqs + 1]
        )
        # Pad for full CUDA graph mode.
        input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

        input_ids = input_buffers.input_ids[:num_tokens].zero_()
        positions = input_buffers.positions[:num_tokens].zero_()

        input_buffers.is_padding[:num_tokens].fill_(True)
        is_padding = input_buffers.is_padding[:num_tokens]

        logits_indices = query_start_loc[1:] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        # Dummy: seq_len == query_len (fresh-prefill shape).
        seq_lens_cpu_upper_bound = torch.from_numpy(num_scheduled_tokens.copy())
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
            num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
            max_seq_len_np=None,
            input_ids=input_ids,
            positions=positions,
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
            prompt_lens=None,
        )


@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= prefill_len:
        # Not prefill.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(request_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    next_pos = num_computed + query_len
    if next_pos < prefill_len:
        next_token = tl.load(request_ptr + next_pos)
        tl.store(next_prefill_tokens_ptr + req_state_idx, next_token)


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
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        prefill_len,
        num_computed_tokens,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    # NOTE(woosuk): We do +1 because the last thread block is used
    # to pad unused seq_lens as 0 for full CUDA graphs.
    _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
        pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        seq_lens.shape[0],
        BLOCK_SIZE=1024,
    )


@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    # Get the number of logits and draft tokens.
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS

    # Compute the logits indices.
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Handling prefill tokens. No sampled or draft tokens.
        return

    # Keep prompt-tail slots intact; only rewrite generated-token slots.
    first_logit_seq_pos = seq_len - num_logits
    if NUM_NEW_SAMPLED_TOKENS > 0 and first_logit_seq_pos >= prefill_len:
        # Write the last sampled token ID to input_ids.
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + logits_start, last_token_id)

    # Write the draft tokens (if any) to input_ids.
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
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
    num_new_sampled_tokens: int = 1,  # excl accepted draft tokens, a.k.a bonus tokens
) -> torch.Tensor:
    assert num_new_sampled_tokens in (0, 1), (
        f"num_new_sampled_tokens must be 0 or 1, got {num_new_sampled_tokens}"
    )
    # use idx_mapping.shape[0] for actual request count
    num_reqs = idx_mapping.shape[0]
    num_speculative_steps = draft_tokens.shape[-1]

    logits_indices = torch.empty(
        num_logits,
        dtype=torch.int64,
        device=input_ids.device,
    )
    _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        # NOTE(woosuk): Add num_new_sampled_tokens to ensure the block covers the
        # last sampled token in addition to all draft tokens.
        BLOCK_SIZE=triton.next_power_of_2(
            num_speculative_steps + num_new_sampled_tokens
        ),
    )
    return logits_indices


@triton.jit
def _get_num_sampled_and_rejected_kernel(
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    is_chunked_prefilling = seq_len < prefill_len

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    tl.store(num_sampled_ptr + batch_idx, num_sampled)

    logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = logits_end - logits_start

    num_rejected = num_logits - num_sampled
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(num_rejected_ptr + batch_idx, num_rejected)


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    num_rejected = torch.empty_like(num_sampled)
    _get_num_sampled_and_rejected_kernel[(num_reqs,)](
        num_sampled,
        num_rejected,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )
    return num_sampled, num_rejected


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
):
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    if req_state_idx < 0:
        # Filter rows with negative index entries.
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    num_sampled = tl.load(num_sampled_ptr + req_id)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + req_id * sampled_tokens_stride + i)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

        if output_bin_counts_ptr is not None:
            token_ptr = (
                output_bin_counts_ptr
                + req_state_idx * output_bin_counts_stride
                + token_id
            )
            count = tl.load(token_ptr)
            tl.store(token_ptr, count + 1)

    if query_start_loc_ptr is None:
        query_len = 0
    else:
        query_start = tl.load(query_start_loc_ptr + req_id)
        query_end = tl.load(query_start_loc_ptr + req_id + 1)
        query_len = query_end - query_start
    num_rejected = tl.load(num_rejected_ptr + req_id)

    computed_delta = query_len - num_rejected
    if computed_delta != 0:
        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + computed_delta)


def post_update(
    # [num_reqs] batch_idx -> req_state_idx; negative index means skip.
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [max_num_reqs, vocab_size]
    output_bin_counts: torch.Tensor | None,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor | None,
    # [max_num_reqs, max_model_len]
    all_token_ids: torch.Tensor,
    # [max_num_reqs]
    total_len: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    _post_update_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens,
        sampled_tokens.stride(0),
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        num_warps=1,
    )


@triton.jit
def _post_update_num_computed_tokens_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
):
    batch_id = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_id)
    query_end = tl.load(query_start_loc_ptr + batch_id + 1)
    query_len = query_end - query_start

    req_state_idx = tl.load(idx_mapping_ptr + batch_id)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + query_len)


def post_update_num_computed_tokens(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    _post_update_num_computed_tokens_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
    )


@triton.jit
def _expand_idx_mapping_kernel(
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    tl.store(expanded_idx_mapping_ptr + start_idx + block, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(
        total_num_logits, dtype=torch.int32, device=idx_mapping.device
    )
    _expand_idx_mapping_kernel[(num_reqs,)](
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        cu_num_logits,
        BLOCK_SIZE=triton.next_power_of_2(max_expand_len),
    )
    return expanded_idx_mapping, expanded_local_pos


# === KUNLUN_NATIVE_INPUTBATCH_PATCH ===
def _get_num_sampled_and_rejected_native(
    num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
):
    n = idx_mapping.shape[0]
    req_state_idx = idx_mapping.to(torch.long)
    prefill = prefill_len[req_state_idx]
    seq_lens = seq_lens[:n]
    is_chunked = seq_lens < prefill
    num_sampled.copy_(torch.where(is_chunked, torch.zeros_like(num_sampled), num_sampled))
    num_logits = cu_num_logits[1:] - cu_num_logits[:-1]
    num_rejected = num_logits.to(num_sampled.dtype) - num_sampled
    num_rejected = torch.where(is_chunked, torch.zeros_like(num_rejected), num_rejected)
    return num_sampled, num_rejected


get_num_sampled_and_rejected = _get_num_sampled_and_rejected_native  # noqa: F811


# === KUNLUN_NATIVE_INPUTBATCH_PATCH2 ===
def _prepare_prefill_inputs_native(
    input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
    all_token_ids, prefill_len, num_computed_tokens,
):
    num_reqs = idx_mapping.shape[0]
    qsl = query_start_loc
    for b in range(num_reqs):
        rsi = int(idx_mapping[b])
        pfl = int(prefill_len[rsi])
        nc = int(num_computed_tokens[rsi])
        if nc >= pfl:
            continue
        qs = int(qsl[b])
        qe = int(qsl[b + 1])
        ql = qe - qs
        if ql > 0:
            input_ids[qs:qe] = all_token_ids[rsi, nc:nc + ql]
        next_pos = nc + ql
        if next_pos < pfl:
            next_prefill_tokens[rsi] = all_token_ids[rsi, next_pos]


def _prepare_pos_seq_lens_native(
    idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens,
):
    num_reqs = idx_mapping.shape[0]
    qsl = query_start_loc
    for b in range(num_reqs):
        rsi = int(idx_mapping[b])
        nc = int(num_computed_tokens[rsi])
        start = int(qsl[b])
        end = int(qsl[b + 1])
        ql = end - start
        seq_lens[b] = nc + ql
        if ql > 0:
            pos[start:end] = nc + torch.arange(ql, device=pos.device, dtype=pos.dtype)
    if seq_lens.shape[0] > num_reqs:                 # pad unused rows for full graphs
        seq_lens[num_reqs:] = 0


def _combine_sampled_and_draft_tokens_native(
    input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
    prefill_len, draft_tokens, cu_num_logits, num_logits, num_new_sampled_tokens=1,
):
    assert num_new_sampled_tokens in (0, 1)
    num_reqs = idx_mapping.shape[0]
    dev = input_ids.device
    logits_indices = torch.empty(num_logits, dtype=torch.int64, device=dev)
    qsl = query_start_loc
    for b in range(num_reqs):
        rsi = int(idx_mapping[b])
        cs = int(cu_num_logits[b])
        ce = int(cu_num_logits[b + 1])
        nl = ce - cs
        nd = nl - num_new_sampled_tokens
        qe = int(qsl[b + 1])
        ls = qe - nl
        logits_indices[cs:ce] = ls + torch.arange(nl, device=dev, dtype=torch.int64)
        sl = int(seq_lens[b])
        pl = int(prefill_len[rsi])
        if sl <= pl:                                 # prefill: no sampled/draft
            continue
        first = sl - nl
        if num_new_sampled_tokens > 0 and first >= pl:
            input_ids[ls] = last_sampled_tokens[rsi]
        if nd > 0:
            input_ids[qe - nd:qe] = draft_tokens[rsi, :nd]
    return logits_indices


def _post_update_native(
    idx_mapping, num_computed_tokens, last_sampled_tokens, output_bin_counts,
    sampled_tokens, num_sampled, num_rejected, query_start_loc, all_token_ids, total_len,
):
    num_reqs = idx_mapping.shape[0]
    if query_start_loc is not None:
        qlens = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
    else:
        qlens = [0] * num_reqs
    for b in range(num_reqs):
        rsi = int(idx_mapping[b])
        if rsi < 0:
            continue
        base = int(total_len[rsi])
        n = int(num_sampled[b])
        if n > 0:
            last_sampled_tokens[rsi] = sampled_tokens[b, n - 1]
            total_len[rsi] = base + n
            toks = sampled_tokens[b, :n]
            all_token_ids[rsi, base:base + n] = toks
            if output_bin_counts is not None:
                output_bin_counts[rsi].scatter_add_(
                    0, toks.long(),
                    torch.ones(n, dtype=output_bin_counts.dtype,
                               device=output_bin_counts.device))
        delta = qlens[b] - int(num_rejected[b])
        if delta != 0:
            num_computed_tokens[rsi] += delta


def _post_update_num_computed_tokens_native(idx_mapping, num_computed_tokens, query_start_loc):
    qlen = (query_start_loc[1:] - query_start_loc[:-1]).to(num_computed_tokens.dtype)
    num_computed_tokens.index_add_(0, idx_mapping.to(torch.long), qlen)


def _expand_idx_mapping_native(idx_mapping, total_num_logits, cu_num_logits, max_expand_len):
    num_reqs = idx_mapping.shape[0]
    dev = idx_mapping.device
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(total_num_logits, dtype=torch.int32, device=dev)
    for b in range(num_reqs):
        s = int(cu_num_logits[b])
        e = int(cu_num_logits[b + 1])
        n = e - s
        if n > 0:
            expanded_idx_mapping[s:e] = int(idx_mapping[b])
            expanded_local_pos[s:e] = torch.arange(n, device=dev, dtype=torch.int32)
    return expanded_idx_mapping, expanded_local_pos


prepare_prefill_inputs = _prepare_prefill_inputs_native  # noqa: F811
# === ABTEST_VEC_INPUTPREP ===
def _prepare_pos_seq_lens_vec(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens):
    import torch as _t
    num_reqs = idx_mapping.shape[0]
    qsl = query_start_loc
    idx = idx_mapping.to(_t.long)
    nc = num_computed_tokens.to(_t.long)[idx]                 # [num_reqs]
    starts = qsl[:num_reqs].to(_t.long)
    ends = qsl[1:num_reqs + 1].to(_t.long)
    qlens = ends - starts                                     # [num_reqs]
    seq_lens[:num_reqs] = (nc + qlens).to(seq_lens.dtype)
    if seq_lens.shape[0] > num_reqs:
        seq_lens[num_reqs:] = 0
    if num_reqs > 0:
        total = int(ends[-1].item())
        if total > 0:
            tok = _t.arange(total, device=pos.device, dtype=_t.long)
            rep_base = _t.repeat_interleave(nc, qlens)        # [total]
            rep_start = _t.repeat_interleave(starts, qlens)   # [total]
            pos[:total] = (rep_base + (tok - rep_start)).to(pos.dtype)


prepare_pos_seq_lens = _prepare_pos_seq_lens_vec  # noqa: F811
# === ABTEST_XPU_COMBINE ===
import kunlun_ops as _abtest_kops  # noqa: E402
def _combine_sampled_and_draft_tokens_xpu(
    input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
    prefill_len, draft_tokens, cu_num_logits, num_logits, num_new_sampled_tokens=1,
):
    assert num_new_sampled_tokens == 1, num_new_sampled_tokens
    import torch as _t
    lst = last_sampled_tokens
    if lst.dim() > 1:
        lst = lst.reshape(-1)
    _i32 = _t.int32
    return _abtest_kops.combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping.to(_i32),
        lst.to(_i32),
        query_start_loc,
        seq_lens.to(_i32),
        prefill_len.to(_i32),
        draft_tokens.to(_i32),
        cu_num_logits.to(_i32),
        num_logits,
    )
combine_sampled_and_draft_tokens = _combine_sampled_and_draft_tokens_xpu  # noqa: F811
post_update = _post_update_native  # noqa: F811
post_update_num_computed_tokens = _post_update_num_computed_tokens_native  # noqa: F811
def _expand_idx_mapping_vec(idx_mapping, total_num_logits, cu_num_logits, max_expand_len):
    import torch as _t
    num_reqs = idx_mapping.shape[0]
    dev = idx_mapping.device
    cu = cu_num_logits.to(_t.long)
    starts = cu[:num_reqs]
    counts = cu[1:num_reqs + 1] - starts                      # [num_reqs]
    expanded_idx_mapping = _t.repeat_interleave(idx_mapping, counts)
    tok = _t.arange(total_num_logits, device=dev, dtype=_t.long)
    rep_start = _t.repeat_interleave(starts, counts)
    expanded_local_pos = (tok - rep_start).to(_t.int32)
    return expanded_idx_mapping, expanded_local_pos


expand_idx_mapping = _expand_idx_mapping_vec  # noqa: F811


# === KUNLUN_POSTUPDATE_VEC_PATCH (whk) ===
def _post_update_vec(
    idx_mapping, num_computed_tokens, last_sampled_tokens, output_bin_counts,
    sampled_tokens, num_sampled, num_rejected, query_start_loc, all_token_ids, total_len,
):
    import torch as _t
    if output_bin_counts is not None:
        return _post_update_native(
            idx_mapping, num_computed_tokens, last_sampled_tokens, output_bin_counts,
            sampled_tokens, num_sampled, num_rejected, query_start_loc, all_token_ids, total_len)
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    dev = all_token_ids.device
    idx = idx_mapping.to(_t.long)
    valid = idx >= 0
    n = num_sampled.to(_t.long)
    if query_start_loc is not None:
        qlens = (query_start_loc[1:num_reqs + 1] - query_start_loc[:num_reqs]).to(_t.long)
    else:
        qlens = _t.zeros(num_reqs, dtype=_t.long, device=dev)
    delta = qlens - num_rejected.to(_t.long)
    m = valid & (delta != 0)
    if bool(m.any()):
        num_computed_tokens.index_add_(0, idx[m], delta[m].to(num_computed_tokens.dtype))
    prod = valid & (n > 0)
    if not bool(prod.any()):
        return
    b_idx = _t.nonzero(prod, as_tuple=True)[0]
    rsi_p = idx[b_idx]
    n_p = n[b_idx]
    base_p = total_len.to(_t.long)[rsi_p]
    last_sampled_tokens[rsi_p] = sampled_tokens[b_idx, n_p - 1].to(last_sampled_tokens.dtype)
    total_len[rsi_p] = (base_p + n_p).to(total_len.dtype)
    total_writes = int(n_p.sum().item())
    if total_writes > 0:
        P = b_idx.shape[0]
        seg = _t.repeat_interleave(_t.arange(P, device=dev), n_p)
        starts_cum = _t.cumsum(n_p, 0) - n_p
        j = _t.arange(total_writes, device=dev) - _t.repeat_interleave(starts_cum, n_p)
        dst_rsi = rsi_p[seg]
        dst_col = base_p[seg] + j
        src_val = sampled_tokens[b_idx[seg], j]
        all_token_ids[dst_rsi, dst_col] = src_val.to(all_token_ids.dtype)

post_update = _post_update_vec
# === KUNLUN_V2_HOSTVEC_PATCH ===
import os as _hv_os  # noqa: E402
import torch as _hv_t  # noqa: E402
def _prepare_pos_seq_lens_vec(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens):
    num_reqs = idx_mapping.shape[0]
    qsl = query_start_loc
    idx = idx_mapping.to(_hv_t.long)
    nc = num_computed_tokens.to(_hv_t.long)[idx]
    starts = qsl[:num_reqs].to(_hv_t.long)
    ends = qsl[1:num_reqs + 1].to(_hv_t.long)
    qlens = ends - starts
    seq_lens[:num_reqs] = (nc + qlens).to(seq_lens.dtype)
    if seq_lens.shape[0] > num_reqs:
        seq_lens[num_reqs:] = 0
    if num_reqs > 0:
        total = int(ends[-1].item())
        if total > 0:
            tok = _hv_t.arange(total, device=pos.device, dtype=_hv_t.long)
            rep_base = _hv_t.repeat_interleave(nc, qlens)
            rep_start = _hv_t.repeat_interleave(starts, qlens)
            pos[:total] = (rep_base + (tok - rep_start)).to(pos.dtype)

def _post_update_vec2(
    idx_mapping, num_computed_tokens, last_sampled_tokens, output_bin_counts,
    sampled_tokens, num_sampled, num_rejected, query_start_loc, all_token_ids, total_len,
):
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    dev = all_token_ids.device
    idx = idx_mapping.to(_hv_t.long)
    valid = idx >= 0
    n = num_sampled.to(_hv_t.long)
    if query_start_loc is not None:
        qlens = (query_start_loc[1:num_reqs + 1] - query_start_loc[:num_reqs]).to(_hv_t.long)
    else:
        qlens = _hv_t.zeros(num_reqs, dtype=_hv_t.long, device=dev)
    delta = qlens - num_rejected.to(_hv_t.long)
    m = valid & (delta != 0)
    if bool(m.any()):
        num_computed_tokens.index_add_(0, idx[m], delta[m].to(num_computed_tokens.dtype))
    prod = valid & (n > 0)
    if not bool(prod.any()):
        return
    b_idx = _hv_t.nonzero(prod, as_tuple=True)[0]
    rsi_p = idx[b_idx]
    n_p = n[b_idx]
    base_p = total_len.to(_hv_t.long)[rsi_p]
    S = sampled_tokens.shape[1]
    last_vals = sampled_tokens.reshape(-1).index_select(0, b_idx * S + (n_p - 1))
    last_sampled_tokens.view(-1).index_copy_(0, rsi_p, last_vals.to(last_sampled_tokens.dtype))
    total_len.index_copy_(0, rsi_p, (base_p + n_p).to(total_len.dtype))
    total_writes = int(n_p.sum().item())
    if total_writes > 0:
        P = b_idx.shape[0]
        seg = _hv_t.repeat_interleave(_hv_t.arange(P, device=dev), n_p)
        starts_cum = _hv_t.cumsum(n_p, 0) - n_p
        j = _hv_t.arange(total_writes, device=dev) - _hv_t.repeat_interleave(starts_cum, n_p)
        dst_rsi = rsi_p[seg]
        dst_col = base_p[seg] + j
        src_val = sampled_tokens.reshape(-1).index_select(0, b_idx[seg] * S + j)
        W = all_token_ids.shape[1]
        all_token_ids.reshape(-1).index_copy_(0, dst_rsi * W + dst_col, src_val.to(all_token_ids.dtype))
        if output_bin_counts is not None:
            Vbin = output_bin_counts.shape[1]
            ones = _hv_t.ones(total_writes, dtype=output_bin_counts.dtype, device=dev)
            output_bin_counts.view(-1).index_add_(0, dst_rsi * Vbin + src_val.to(_hv_t.long), ones)

def _combine_sampled_and_draft_tokens_vec(
    input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
    prefill_len, draft_tokens, cu_num_logits, num_logits, num_new_sampled_tokens=1,
):
    assert num_new_sampled_tokens in (0, 1)
    num_reqs = idx_mapping.shape[0]
    dev = input_ids.device
    logits_indices = _hv_t.empty(num_logits, dtype=_hv_t.int64, device=dev)
    if num_reqs == 0:
        return logits_indices
    idxl = idx_mapping.to(_hv_t.long)
    qsl = query_start_loc
    cnl = cu_num_logits.to(_hv_t.long)
    nls = cnl[1:num_reqs + 1] - cnl[:num_reqs]
    qe = qsl[1:num_reqs + 1].to(_hv_t.long)
    ls = qe - nls
    within = _hv_t.arange(num_logits, device=dev) - _hv_t.repeat_interleave(cnl[:num_reqs], nls)
    logits_indices[:] = _hv_t.repeat_interleave(ls, nls) + within
    sl = seq_lens[:num_reqs].to(_hv_t.long)
    pl = prefill_len.to(_hv_t.long)[idxl]
    first = sl - nls
    decode = sl > pl
    if num_new_sampled_tokens > 0:
        mask_s = decode & (first >= pl)
        if bool(mask_s.any()):
            rows = _hv_t.nonzero(mask_s, as_tuple=True)[0]
            input_ids[ls[rows]] = last_sampled_tokens[idxl[rows]].to(input_ids.dtype)
    nd = nls - num_new_sampled_tokens
    draft_rows = _hv_t.nonzero(decode & (nd > 0), as_tuple=True)[0]
    if draft_rows.numel() > 0:
        for b in draft_rows.tolist():
            ndb = int(nd[b])
            qeb = int(qe[b])
            rsi = int(idxl[b])
            input_ids[qeb - ndb:qeb] = draft_tokens[rsi, :ndb]
    return logits_indices

if _hv_os.environ.get("KUNLUN_HOSTVEC", "1") != "0":
    if _hv_os.environ.get("KUNLUN_HOSTVEC_POSSEQ", "1") != "0":
        prepare_pos_seq_lens = _prepare_pos_seq_lens_vec
    if _hv_os.environ.get("KUNLUN_HOSTVEC_POSTUPDATE", "1") != "0":
        post_update = _post_update_vec2
    if _hv_os.environ.get("KUNLUN_HOSTVEC_COMBINE", "1") != "0":
        combine_sampled_and_draft_tokens = _combine_sampled_and_draft_tokens_vec


# === KUNLUN_POSTUPD3_PATCH ===
def _post_update_vec3(
    idx_mapping, num_computed_tokens, last_sampled_tokens, output_bin_counts,
    sampled_tokens, num_sampled, num_rejected, query_start_loc, all_token_ids, total_len,
):
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    dev = all_token_ids.device
    idx = idx_mapping.to(_hv_t.long)
    valid = idx >= 0
    safe_idx = _hv_t.where(valid, idx, _hv_t.zeros_like(idx))
    n = num_sampled.to(_hv_t.long)
    if query_start_loc is not None:
        qlens = (query_start_loc[1:num_reqs + 1] - query_start_loc[:num_reqs]).to(_hv_t.long)
    else:
        qlens = _hv_t.zeros(num_reqs, dtype=_hv_t.long, device=dev)
    # dense masked update: invalid rows contribute 0 (safe_idx=0), valid rows
    # have unique idx -> no index_add collision. Removes m.any() + gather sync.
    delta = (qlens - num_rejected.to(_hv_t.long)) * valid.to(_hv_t.long)
    num_computed_tokens.index_add_(0, safe_idx, delta.to(num_computed_tokens.dtype))
    prod = valid & (n > 0)
    b_idx = _hv_t.nonzero(prod, as_tuple=True)[0]
    if b_idx.numel() == 0:
        return
    rsi_p = idx[b_idx]
    n_p = n[b_idx]
    base_p = total_len.to(_hv_t.long)[rsi_p]
    S = sampled_tokens.shape[1]
    last_vals = sampled_tokens.reshape(-1).index_select(0, b_idx * S + (n_p - 1))
    last_sampled_tokens.view(-1).index_copy_(0, rsi_p, last_vals.to(last_sampled_tokens.dtype))
    total_len.index_copy_(0, rsi_p, (base_p + n_p).to(total_len.dtype))
    if S == 1:
        # no-spec / bonus-only: exactly one token per prod row, no .item() sync.
        dst_rsi = rsi_p
        dst_col = base_p
        src_val = last_vals
    else:
        total_writes = int(n_p.sum().item())
        if total_writes <= 0:
            return
        P = b_idx.shape[0]
        seg = _hv_t.repeat_interleave(_hv_t.arange(P, device=dev), n_p)
        starts_cum = _hv_t.cumsum(n_p, 0) - n_p
        j = _hv_t.arange(total_writes, device=dev) - _hv_t.repeat_interleave(starts_cum, n_p)
        dst_rsi = rsi_p[seg]
        dst_col = base_p[seg] + j
        src_val = sampled_tokens.reshape(-1).index_select(0, b_idx[seg] * S + j)
    W = all_token_ids.shape[1]
    all_token_ids.reshape(-1).index_copy_(0, dst_rsi * W + dst_col, src_val.to(all_token_ids.dtype))
    if output_bin_counts is not None:
        Vbin = output_bin_counts.shape[1]
        ones = _hv_t.ones(dst_rsi.shape[0], dtype=output_bin_counts.dtype, device=dev)
        output_bin_counts.view(-1).index_add_(0, dst_rsi * Vbin + src_val.to(_hv_t.long), ones)


if (
    _hv_os.environ.get("KUNLUN_HOSTVEC", "1") != "0"
    and _hv_os.environ.get("KUNLUN_HOSTVEC_POSTUPDATE", "1") != "0"
    and _hv_os.environ.get("KUNLUN_POSTUPD3", "1") != "0"
):
    post_update = _post_update_vec3
