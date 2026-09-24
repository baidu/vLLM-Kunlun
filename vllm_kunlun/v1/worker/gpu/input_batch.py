# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacements for ``vllm.v1.worker.gpu.input_batch``.

Every function below launches a Triton kernel upstream and is replaced by its
xspeedgate_ops equivalent.

Preconditions the ops enforce (adapted here):
  * ``combine_sampled_and_draft_tokens`` needs int32 token tensors and allocates
    + returns ``logits_indices`` itself. The op reads ``draft_tokens[req, 0]``
    unconditionally, so the 0-column (no-spec-decode) buffer is swapped for a
    1-column scratch by ``_draft_tokens_for_op``.
  * ``prepare_pos_seq_lens`` requires ``query_start_loc`` to be exactly
    ``[num_reqs + 1]``; the runner passes the CUDA-graph-padded slice, so only
    the first ``num_reqs + 1`` entries are handed over.
  * ``post_update`` wants int32 token tensors; ``last_sampled_tokens`` is mutated
    in place, so it runs on an int32 copy that is written back.
  * ``prepare_prefill_inputs`` dispatches on ``next_prefill_tokens.dim()``: the
    runner's tensor is 2-D ``[num_prefill_lookahead, max_num_reqs]``, i.e. the
    v0.27.0+ contract where an out-of-range lookahead slot is written as 0 (the
    1-D v0.26.0 contract would keep the old value instead).

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)

logger = logging.getLogger("vllm_kunlun")


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    torch.ops.xspeedgate_ops.prepare_prefill_inputs(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    )


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    # The op requires `query_start_loc` to be exactly [num_reqs + 1]; the runner
    # passes the CUDA-graph-padded slice [num_reqs_padded + 1]. Only the first
    # num_reqs entries are semantically relevant, so slice.
    num_reqs = idx_mapping.shape[0]
    torch.ops.xspeedgate_ops.prepare_pos_seq_lens(
        idx_mapping,
        query_start_loc[: num_reqs + 1],
        num_computed_tokens,
        pos,
        seq_lens,
    )


# Scratch drafts for the native op: it reads `draft_tokens[req, 0]`
# unconditionally and therefore faults on a 0-column buffer (the no-spec-decode
# shape). The extra column is never read: `num_draft_tokens` stays 0.
_OP_DRAFT_PAD: dict[tuple[int, torch.device], torch.Tensor] = {}


def _draft_tokens_for_op(draft_tokens: torch.Tensor) -> torch.Tensor:
    if draft_tokens.shape[-1] > 0:
        if draft_tokens.dtype != torch.int32:
            return draft_tokens.to(torch.int32)
        return draft_tokens
    key = (draft_tokens.shape[0], draft_tokens.device)
    buf = _OP_DRAFT_PAD.get(key)
    if buf is None:
        buf = torch.zeros(
            (draft_tokens.shape[0], 1), dtype=torch.int32, device=draft_tokens.device
        )
        _OP_DRAFT_PAD[key] = buf
    return buf


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
    # The op needs int32 token tensors and allocates + returns logits_indices.
    return torch.ops.xspeedgate_ops.combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping,
        last_sampled_tokens.to(torch.int32),
        query_start_loc,
        seq_lens,
        prefill_len,
        _draft_tokens_for_op(draft_tokens),
        cu_num_logits,
        num_logits,
        num_new_sampled_tokens,
    )


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.xspeedgate_ops.get_num_sampled_and_rejected(
        num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
    )


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
    # The op wants int32 token tensors. `last_sampled_tokens` is mutated in
    # place, so run on an int32 copy and write it back.
    if last_sampled_tokens.dtype == torch.int32:
        last32 = last_sampled_tokens
    else:
        last32 = last_sampled_tokens.to(torch.int32)
    torch.ops.xspeedgate_ops.post_update(
        idx_mapping,
        num_computed_tokens,
        last32,
        output_bin_counts,
        sampled_tokens.to(torch.int32),
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    )
    if last32 is not last_sampled_tokens:
        last_sampled_tokens.copy_(last32)


def post_update_num_computed_tokens(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
) -> None:
    torch.ops.xspeedgate_ops.post_update_num_computed_tokens(
        idx_mapping, num_computed_tokens, query_start_loc
    )


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.xspeedgate_ops.expand_idx_mapping(
        idx_mapping, total_num_logits, cu_num_logits, max_expand_len
    )


prepare_prefill_inputs._kunlun_patched = True
prepare_pos_seq_lens._kunlun_patched = True
combine_sampled_and_draft_tokens._kunlun_patched = True
get_num_sampled_and_rejected._kunlun_patched = True
post_update._kunlun_patched = True
post_update_num_computed_tokens._kunlun_patched = True
expand_idx_mapping._kunlun_patched = True
