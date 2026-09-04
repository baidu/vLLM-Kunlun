# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests for the vectorised sync-free ``_kernels`` replacements.

Each reference implementation below is a transcription of the upstream Triton
kernel it stands in for -- ``input_batch._post_update_kernel`` and
``model_states.mamba_hybrid._scatter_num_accepted_kernel``, one Python program
per batch row. The vectorised versions must match them field for field.

Both run in ``GPUModelRunner.postprocess_sampled`` and are the two functions
that actually receive the ``-1`` sentinel, so the filtered-row cases here are
the ones that matter.

``_kernels`` depends on ``torch`` alone, so these run on CPU with no vLLM and no
XPU hardware.
"""

import pytest
import torch

from vllm_kunlun.v1.worker.gpu._kernels import (
    post_update,
    prepare_rope_positions,
    scatter_num_accepted,
    segment_ids,
)


@pytest.mark.parametrize(
    "cu_seqlens",
    [
        [0, 5],  # single segment
        [0, 3, 5],  # two segments
        [0, 3, 3, 5],  # interior zero-length segment
        [0, 3, 3],  # trailing zero-length segment
        [0, 3, 3, 3],  # two trailing zero-length segments
        [0, 0, 5],  # leading zero-length segment
        [0, 0, 0],  # everything empty
    ],
)
def test_segment_ids_matches_searchsorted(cu_seqlens):
    """segment_ids replaces searchsorted, which falls back to CPU on Kunlun.

    torch.searchsorted is the reference here only because it is the definition
    being reproduced; the shipped code must not call it.
    """
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    num_segments = len(cu_seqlens) - 1
    length = int(cu[-1])
    positions = torch.arange(length)
    expected = torch.searchsorted(cu[1 : num_segments + 1], positions, right=True)
    assert torch.equal(segment_ids(cu, num_segments, length), expected.to(torch.long))


def reference_post_update(
    idx_mapping,
    num_computed_tokens,
    last_sampled_tokens,
    output_bin_counts,
    sampled_tokens,
    num_sampled,
    num_rejected,
    query_start_loc,
    all_token_ids,
    total_len,
):
    """One Python program per batch row, mirroring the upstream kernel."""
    for b in range(idx_mapping.shape[0]):
        s = int(idx_mapping[b])
        if s < 0:
            continue
        base = int(total_len[s])
        n = int(num_sampled[b])
        if n > 0:
            last_sampled_tokens[s] = sampled_tokens[b, n - 1]
            total_len[s] = base + n
        for i in range(n):
            token_id = int(sampled_tokens[b, i])
            all_token_ids[s, base + i] = token_id
            if output_bin_counts is not None:
                output_bin_counts[s, token_id] += 1
        if query_start_loc is None:
            query_len = 0
        else:
            query_len = int(query_start_loc[b + 1]) - int(query_start_loc[b])
        delta = query_len - int(num_rejected[b])
        if delta != 0:
            num_computed_tokens[s] += delta


def make_state(max_num_reqs, max_model_len, vocab_size, seed, with_bins=True):
    g = torch.Generator().manual_seed(seed)
    return {
        "num_computed_tokens": torch.randint(
            0, 8, (max_num_reqs,), dtype=torch.int32, generator=g
        ),
        "last_sampled_tokens": torch.randint(
            0, vocab_size, (max_num_reqs,), dtype=torch.int64, generator=g
        ),
        "output_bin_counts": (
            torch.randint(
                0, 3, (max_num_reqs, vocab_size), dtype=torch.int32, generator=g
            )
            if with_bins
            else None
        ),
        "all_token_ids": torch.randint(
            0, vocab_size, (max_num_reqs, max_model_len), dtype=torch.int32, generator=g
        ),
        "total_len": torch.randint(
            1, 5, (max_num_reqs,), dtype=torch.int32, generator=g
        ),
    }


def run_both(state_kwargs, **call_kwargs):
    """Run reference and vectorised versions on identical clones."""
    ref = {k: (v.clone() if v is not None else None) for k, v in state_kwargs.items()}
    got = {k: (v.clone() if v is not None else None) for k, v in state_kwargs.items()}
    reference_post_update(**call_kwargs, **ref)
    post_update(**call_kwargs, **got)
    for name in ref:
        if ref[name] is None:
            continue
        assert torch.equal(
            ref[name], got[name]
        ), f"{name} differs\n  ref={ref[name]}\n  got={got[name]}"


@pytest.mark.parametrize("with_bins", [True, False])
@pytest.mark.parametrize("with_qsl", [True, False])
def test_matches_reference_decode(with_bins, with_qsl):
    """Steady-state decode: every row valid, exactly one sampled token."""
    num_reqs, max_num_reqs, max_model_len, vocab = 5, 8, 64, 32
    idx_mapping = torch.tensor([3, 0, 6, 1, 5], dtype=torch.int32)
    sampled = torch.randint(0, vocab, (num_reqs, 1), dtype=torch.int64)
    run_both(
        make_state(max_num_reqs, max_model_len, vocab, seed=1, with_bins=with_bins),
        idx_mapping=idx_mapping,
        sampled_tokens=sampled,
        num_sampled=torch.ones(max_num_reqs, dtype=torch.int32),
        num_rejected=torch.zeros(max_num_reqs, dtype=torch.int32),
        query_start_loc=(
            torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32) if with_qsl else None
        ),
    )


def test_matches_reference_ragged_spec_decode():
    """Speculative decoding: per-row num_sampled varies, including zero."""
    num_reqs, max_num_reqs, max_model_len, vocab = 4, 8, 64, 32
    num_sampled = torch.zeros(max_num_reqs, dtype=torch.int32)
    num_sampled[:num_reqs] = torch.tensor([3, 0, 1, 2], dtype=torch.int32)
    run_both(
        make_state(max_num_reqs, max_model_len, vocab, seed=2),
        idx_mapping=torch.tensor([2, 5, 0, 7], dtype=torch.int32),
        sampled_tokens=torch.randint(0, vocab, (num_reqs, 3), dtype=torch.int64),
        num_sampled=num_sampled,
        num_rejected=torch.tensor([0, 3, 2, 1, 0, 0, 0, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8, 11, 14], dtype=torch.int32),
    )


def test_matches_reference_filtered_rows():
    """Pipeline parallelism: several rows carry the -1 sentinel.

    The vectorised version folds those onto destination 0, so this also covers
    the case where a filtered row shares a destination with a real one
    (idx_mapping contains 0 as a genuine target).
    """
    num_reqs, max_num_reqs, max_model_len, vocab = 5, 8, 64, 32
    num_sampled = torch.zeros(max_num_reqs, dtype=torch.int32)
    num_sampled[:num_reqs] = torch.tensor([1, 2, 1, 1, 2], dtype=torch.int32)
    run_both(
        make_state(max_num_reqs, max_model_len, vocab, seed=3),
        idx_mapping=torch.tensor([0, -1, 4, -1, -1], dtype=torch.int32),
        sampled_tokens=torch.randint(0, vocab, (num_reqs, 2), dtype=torch.int64),
        num_sampled=num_sampled,
        num_rejected=torch.tensor([0, 1, 1, 2, 0, 0, 0, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4, 6, 8, 10], dtype=torch.int32),
    )


def test_matches_reference_repeated_tokens():
    """The same token id sampled twice in a row must bump its bin count twice."""
    num_reqs, max_num_reqs, max_model_len, vocab = 2, 4, 32, 8
    num_sampled = torch.zeros(max_num_reqs, dtype=torch.int32)
    num_sampled[:num_reqs] = torch.tensor([3, 2], dtype=torch.int32)
    run_both(
        make_state(max_num_reqs, max_model_len, vocab, seed=4),
        idx_mapping=torch.tensor([1, 3], dtype=torch.int32),
        sampled_tokens=torch.tensor([[5, 5, 5], [2, 2, 0]], dtype=torch.int64),
        num_sampled=num_sampled,
        num_rejected=torch.zeros(max_num_reqs, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
    )


def test_empty_batch_is_a_noop():
    state = make_state(4, 32, 8, seed=5)
    before = {k: (v.clone() if v is not None else None) for k, v in state.items()}
    post_update(
        idx_mapping=torch.zeros(0, dtype=torch.int32),
        sampled_tokens=torch.zeros((0, 1), dtype=torch.int64),
        num_sampled=torch.zeros(4, dtype=torch.int32),
        num_rejected=torch.zeros(4, dtype=torch.int32),
        query_start_loc=torch.zeros(1, dtype=torch.int32),
        **state,
    )
    for name, value in before.items():
        if value is not None:
            assert torch.equal(value, state[name]), name


def test_does_not_mutate_its_inputs():
    """sampled_tokens and friends are the caller's tensors; keep them intact."""
    max_num_reqs, max_model_len, vocab = 4, 32, 8
    sampled = torch.tensor([[5, 1], [2, 7], [0, 3]], dtype=torch.int64)
    num_sampled = torch.tensor([2, 1, 2, 0], dtype=torch.int32)
    num_rejected = torch.tensor([0, 1, 0, 0], dtype=torch.int32)
    idx_mapping = torch.tensor([1, 3, 0], dtype=torch.int32)
    qsl = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    inputs = [sampled, num_sampled, num_rejected, idx_mapping, qsl]
    originals = [t.clone() for t in inputs]

    post_update(
        idx_mapping=idx_mapping,
        sampled_tokens=sampled,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        query_start_loc=qsl,
        **make_state(max_num_reqs, max_model_len, vocab, seed=6),
    )
    for original, current in zip(originals, inputs):
        assert torch.equal(original, current)


@pytest.mark.parametrize("fn", [post_update, scatter_num_accepted])
def test_no_host_sync_ops_are_used(fn):
    """Guard the property that motivated the rewrites.

    A ``.item()`` / ``.tolist()`` / boolean-mask select would each force a
    device-to-host transfer. On CPU tensors they are harmless, so instead of
    timing anything this asserts on the source: the function must not call them.
    ``.numel()`` is banned too -- it is how a masked-select result gets tested
    for emptiness, which is the sync ``scatter_num_accepted`` used to have.
    """
    import inspect

    source = inspect.getsource(fn)
    body = source.split('"""')[2]  # drop the docstring, which discusses them
    for banned in (
        ".item()",
        ".tolist()",
        ".nonzero(",
        ".numel()",
        "masked_select",
        ".cpu()",
    ):
        assert banned not in body, f"{banned} reintroduces a host sync"


def reference_scatter_num_accepted(idx_mapping, num_sampled, num_accepted):
    """One Python program per batch row, mirroring the upstream kernel.

    Upstream: ``model_states/mamba_hybrid.py:337-348``.
    """
    for row in range(idx_mapping.shape[0]):
        req = int(idx_mapping[row])
        if req < 0:
            continue
        num_accepted[req] = max(int(num_sampled[row]), 1)


def run_both_scatter(idx_mapping, num_sampled, max_num_reqs, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = torch.randint(0, 9, (max_num_reqs,), dtype=torch.int32, generator=g)
    ref, got = state.clone(), state.clone()
    reference_scatter_num_accepted(idx_mapping, num_sampled, ref)
    scatter_num_accepted(idx_mapping, num_sampled, got)
    assert torch.equal(ref, got), f"\n  ref={ref}\n  got={got}"


def test_scatter_matches_reference_all_valid():
    run_both_scatter(
        idx_mapping=torch.tensor([3, 0, 6, 1, 5], dtype=torch.int32),
        num_sampled=torch.tensor([1, 2, 1, 3, 1], dtype=torch.int32),
        max_num_reqs=8,
    )


def test_scatter_clamps_zero_to_one():
    """Chunked prefill samples nothing; mamba's neutral value is 1, not 0."""
    run_both_scatter(
        idx_mapping=torch.tensor([2, 4, 1], dtype=torch.int32),
        num_sampled=torch.tensor([0, 0, 2], dtype=torch.int32),
        max_num_reqs=8,
        seed=1,
    )


def test_scatter_skips_filtered_rows_colliding_on_zero():
    """PP filtered rows fold onto destination 0, where a real row also lands.

    The collision is the whole reason the vectorised form accumulates rather
    than assigns, so index 0 must still come out as the *valid* row's value.
    """
    run_both_scatter(
        idx_mapping=torch.tensor([0, -1, 4, -1, -1], dtype=torch.int32),
        num_sampled=torch.tensor([3, 7, 1, 9, 0], dtype=torch.int32),
        max_num_reqs=8,
        seed=2,
    )


def test_scatter_all_rows_filtered_is_a_noop():
    """Destination 0 must be untouched when no valid row targets it."""
    run_both_scatter(
        idx_mapping=torch.tensor([-1, -1, -1], dtype=torch.int32),
        num_sampled=torch.tensor([4, 5, 6], dtype=torch.int32),
        max_num_reqs=8,
        seed=3,
    )


def test_scatter_empty_batch_is_a_noop():
    run_both_scatter(
        idx_mapping=torch.zeros(0, dtype=torch.int32),
        num_sampled=torch.zeros(0, dtype=torch.int32),
        max_num_reqs=4,
        seed=4,
    )


def test_scatter_does_not_mutate_its_inputs():
    idx_mapping = torch.tensor([1, -1, 3], dtype=torch.int32)
    num_sampled = torch.tensor([2, 5, 0], dtype=torch.int32)
    originals = [idx_mapping.clone(), num_sampled.clone()]
    scatter_num_accepted(idx_mapping, num_sampled, torch.zeros(4, dtype=torch.int32))
    for original, current in zip(originals, [idx_mapping, num_sampled]):
        assert torch.equal(original, current)


def test_prepare_rope_positions_handles_prefill_and_decode_rows():
    """Prefill rows gather model positions; decode rows use offset positions."""
    num_dims, max_model_len = 2, 8
    positions = torch.full((num_dims, 6), -1, dtype=torch.int64)
    prefill_positions = torch.arange(2 * 2 * max_model_len, dtype=torch.int32).view(
        4, max_model_len
    )
    prefill_delta = torch.tensor([2, 0], dtype=torch.int32)
    idx_mapping = torch.tensor([1, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    prefill_lens = torch.tensor([3, 3], dtype=torch.int32)
    num_computed_tokens = torch.tensor([3, 1], dtype=torch.int32)

    prepare_rope_positions(
        positions,
        prefill_positions,
        prefill_delta,
        idx_mapping,
        query_start_loc,
        prefill_lens,
        num_computed_tokens,
        num_dims,
        max_model_len,
    )

    # Request 1 is prefilling at original positions 1 and 2.
    assert torch.equal(positions[:, :2], prefill_positions[2:4, 1:3])
    # Request 0 is decoding: computed position 3 plus its delta of 2.
    assert torch.equal(positions[:, 2], torch.tensor([5, 5]))


def test_prepare_rope_positions_clamps_prefill_lookup_and_handles_empty_batch():
    positions = torch.full((1, 3), -1, dtype=torch.int64)
    prefill_positions = torch.tensor([[10, 20, 30]], dtype=torch.int32)
    prepare_rope_positions(
        positions,
        prefill_positions,
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([10], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
        1,
        3,
    )
    assert torch.equal(positions[0, :2], torch.tensor([30, 30]))

    empty = torch.full((1, 3), -1, dtype=torch.int64)
    prepare_rope_positions(
        empty,
        prefill_positions,
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(0, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        1,
        3,
    )
    assert torch.equal(empty, torch.full_like(empty, -1))
