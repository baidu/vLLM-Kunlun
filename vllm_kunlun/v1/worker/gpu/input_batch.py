# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op overrides for ``vllm.v1.worker.gpu.input_batch``.

Leaves the upstream ``InputBuffers`` / ``InputBatch`` dataclasses alone and
reimplements the seven Triton-backed module functions on top of the Kunlun
native operators ``torch.ops.xspeedgate_ops.*``. Every function here is a thin
wrapper that forwards to the matching device op, so the whole input-preparation
and postprocess path is sync-free: no ``.tolist()`` / ``.item()`` / boolean-mask
select, no per-request Python loop, and nothing that pulls a metadata tensor to
host. The torch-native reference implementations these replace still live in
``_kernels`` (``post_update``) and are used as the CPU parity oracle by the unit
tests; see ``tests/ut/test_mrv2_input_batch.py``.

Dtype note: the native ``combine_sampled_and_draft_tokens`` and ``post_update``
ops require their *token* tensors to be int32, but upstream keeps
``last_sampled_tokens`` / ``draft_tokens`` (states.py:64-77) and the sampler's
``sampled_tokens`` in int64. Token ids are bounded by the vocab size and so fit
losslessly in int32, so the wrappers cast those inputs to int32 device
temporaries before the call. ``post_update`` additionally mutates
``last_sampled_tokens`` in place, so its int32 view is copied back into the
int64 buffer afterwards. Both casts are pure device-side copies of small
``[max_num_reqs, *]`` tensors -- no host sync.

Boundary note: FULL graph metadata may pad ``query_start_loc`` beyond the real
request count, while the native preparation ops require exactly
``idx_mapping.numel() + 1`` entries, so those wrappers pass the corresponding
prefix view. Empty batches are handled without launching zero-work native
kernels. When speculative decoding is disabled, upstream owns a zero-width
``draft_tokens`` tensor; the combine wrapper supplies a valid non-empty device
pointer because the optimized native kernel may issue a speculative prefetch
even when the logical draft-token count is zero.

Sentinel invariant: every function here except ``post_update`` may assume its
``idx_mapping`` / ``expanded_idx_mapping`` is non-negative, so none of them
needs a ``-1`` guard. Upstream builds ``InputBatch.idx_mapping`` from
``req_id_to_index.get`` over the scheduled request ids (model_runner.py:862),
which cannot yield ``-1``, and derives ``expanded_idx_mapping`` from it. The
``-1`` sentinel is produced in exactly one place on the non-spec-decode path --
``PPHandler.get_prev_sampled_outputs`` (pp_utils.py:115) masking rows on
non-last pipeline-parallel ranks -- and that tensor is local to
``GPUModelRunner.postprocess_sampled`` (model_runner.py:1082), reaching only
``post_update`` and ``model_state.postprocess_state``. The native ``post_update``
op handles the sentinel internally; the other ops are never handed it.
"""

import logging

import torch
import vllm.v1.worker.gpu.input_batch as _up

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
    if idx_mapping.numel() == 0:
        return
    torch.ops.xspeedgate_ops.prepare_prefill_inputs(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc[: idx_mapping.numel() + 1],
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
    if idx_mapping.numel() == 0:
        seq_lens.zero_()
        return
    torch.ops.xspeedgate_ops.prepare_pos_seq_lens(
        idx_mapping,
        query_start_loc[: idx_mapping.numel() + 1],
        num_computed_tokens,
        pos,
        seq_lens,
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
    if idx_mapping.numel() == 0:
        return input_ids.new_empty((num_logits,), dtype=torch.int64)
    # Native op wants int32 token tensors; upstream buffers are int64. Casting
    # is lossless (token id < vocab) and read-only here, so no write-back.
    last32 = last_sampled_tokens.to(torch.int32)
    if draft_tokens.numel() == 0:
        # RequestState uses [max_reqs, 0] without speculative decoding. The
        # optimized kernel may still prefetch from the draft pointer, so pass
        # valid storage that cannot be observed by the logical no-draft path.
        # Avoid converting the empty tensor on this common decode path.
        draft32 = last32.reshape(-1, 1)
    else:
        draft32 = draft_tokens.to(torch.int32)
    return torch.ops.xspeedgate_ops.combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping,
        last32,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft32,
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
    if idx_mapping.numel() == 0:
        return (num_sampled, torch.empty_like(num_sampled))
    # Mutates num_sampled in place and returns (num_sampled, num_rejected).
    return torch.ops.xspeedgate_ops.get_num_sampled_and_rejected(
        num_sampled,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )


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
    if idx_mapping.numel() == 0:
        return
    # Native op wants int32 tokens. ``last_sampled_tokens`` is mutated in place,
    # so operate on an int32 view and copy the result back into the int64 buffer.
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
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    if idx_mapping.numel() == 0:
        return
    torch.ops.xspeedgate_ops.post_update_num_computed_tokens(
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
    )


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if total_num_logits == 0:
        return (idx_mapping.new_empty((0,)), idx_mapping.new_empty((0,)))
    return torch.ops.xspeedgate_ops.expand_idx_mapping(
        idx_mapping,
        total_num_logits,
        cu_num_logits,
        max_expand_len,
    )


# Install into the upstream module's globals. These are all module-level
# functions that consumers bind by name at import time (model_runner.py:75-83,
# sample/sampler.py:15, spec_decode/rejection_sampler.py:9), which is why the
# patch has to be in place before those modules are executed -- the post-import
# dispatcher in vllm_kunlun/registration/import_hooks.py guarantees that.
_up.prepare_prefill_inputs = prepare_prefill_inputs
_up.prepare_pos_seq_lens = prepare_pos_seq_lens
_up.combine_sampled_and_draft_tokens = combine_sampled_and_draft_tokens
_up.get_num_sampled_and_rejected = get_num_sampled_and_rejected
_up.post_update = post_update
_up.post_update_num_computed_tokens = post_update_num_computed_tokens
_up.expand_idx_mapping = expand_idx_mapping
logger.info("[KunlunPlugin] V2 input_batch patched (xspeedgate_ops native)")
