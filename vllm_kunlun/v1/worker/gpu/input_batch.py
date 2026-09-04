# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun torch-native overrides for ``vllm.v1.worker.gpu.input_batch``.

Leaves the upstream ``InputBuffers`` / ``InputBatch`` dataclasses alone and
reimplements the seven Triton-backed module functions with torch-native
equivalents. They operate on jagged per-request segments.

``post_update`` and ``get_num_sampled_and_rejected`` are fully vectorised and
perform no device-to-host transfer, which takes the postprocess phase to zero
syncs -- that is where a sync hurts most, because it directly negates the
``AsyncOutput`` copy-stream design (see ``_kernels.post_update``).

Sentinel invariant: every function here except ``post_update`` may assume its
``idx_mapping`` / ``expanded_idx_mapping`` is non-negative, so none of them
needs a ``-1`` guard. Upstream builds ``InputBatch.idx_mapping`` from
``req_id_to_index.get`` over the scheduled request ids (model_runner.py:862),
which cannot yield ``-1``, and derives ``expanded_idx_mapping`` from it. The
``-1`` sentinel is produced in exactly one place on the non-spec-decode path --
``PPHandler.get_prev_sampled_outputs`` (pp_utils.py:115) masking rows on
non-last pipeline-parallel ranks -- and that tensor is local to
``GPUModelRunner.postprocess_sampled`` (model_runner.py:1082, whose signature
documents it), reaching only ``post_update`` and
``model_state.postprocess_state``. Both handle it; see ``_kernels.post_update``
and ``_kernels.scatter_num_accepted``. Adding guards to the rest would not just
be dead code: the natural spelling (``x[idx_mapping >= 0]``) is a boolean-mask
selection, i.e. a host sync, which is precisely what the TODO below is about.

Pipeline parallelism is not a declared feature of the plugin (the README lists
tensor parallelism only, and there is no PP gate or test in vllm_kunlun), but
it does run: V2 with PP=2/TP=2 has been exercised on Kunlun. A sentinel needs
more than that -- a request freed or no longer needing sampled output while a
broadcast is in flight -- so the ``-1`` branch is covered by the unit tests
rather than by any hardware run so far. Both functions keep the handling
because in their vectorised form it costs nothing (one extra term in a
``torch.where`` over a mask that is computed either way), and because matching
upstream semantics is what keeps a filtered row from silently corrupting
another request's state.

TODO: the remaining five still loop per request in Python and pull their
metadata index tensors to host with ``.tolist()``. Each such call is a
main-stream sync, so the input-preparation phase stalls once per step and the
V2 runner's CPU/GPU overlap is lost there. They are correct but not optimised.
Vectorising them is deliberately deferred rather than done piecemeal: one
remaining sync stalls the whole phase just as thoroughly as five, and
``BlockTables.compute_slot_mappings`` cannot drop its own ``.item()`` until
``kunlun_ops.compute_slot_mappings`` can take num_tokens as a device scalar
(see the TODO there). When that lands, these should go in the same change.
``_kernels.segment_ids`` is the sync-free token-to-request mapping they need;
note ``torch.searchsorted`` is not usable on Kunlun XPU.
"""

import torch
import vllm.v1.worker.gpu.input_batch as _up

from vllm_kunlun.v1.worker.gpu._kernels import post_update


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


# ``post_update`` is fully vectorised and sync-free; it lives in ``_kernels``
# with the other torch-only kernel stand-ins so it can be unit-tested on CPU.


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
    # ``cu_num_logits`` may cover fewer entries than ``total_num_logits`` for
    # padded CUDA-graph buffers. Initialize the padding so callers never see
    # uninitialized device memory.
    expanded_idx_mapping = idx_mapping.new_zeros(total_num_logits)
    expanded_local_pos = torch.zeros(total_num_logits, dtype=torch.int32, device=device)
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
