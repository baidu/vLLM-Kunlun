# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-device tests for the Kunlun native-op input_batch wrappers.

Every function in ``vllm_kunlun/v1/worker/gpu/input_batch.py`` is now a thin
forwarder to a ``torch.ops.xspeedgate_ops.*`` op, which only registers for the
CUDA/XLA/XPU backends (Kunlun XPU presents as CUDA via ``torch_xmlir``). There
is no CPU kernel, so these tests are gated on device availability. The
torch-native reference the wrappers replaced still lives in ``_kernels`` and is
exercised on CPU by ``test_mrv2_kernels.py`` -- that file is the platform-free
parity oracle; this file checks the native ops themselves plus the wrappers'
dtype adaptation.

Tensor dtypes below mirror the live path: the upstream ``InputBuffers``
(input_batch.py:23-31) keeps ``input_ids`` / metadata int32 and ``positions``
int64, and the native ops enforce that (``input_ids must be int32``,
``metadata must be int32``). The token buffers upstream (``last_sampled_tokens``
/ ``draft_tokens``, states.py) stay int64; the wrappers cast those to int32
before the call, so the tests pass them int64 to cover that cast.
"""

import pytest
import torch

_CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not _CUDA,
    reason="MRV2 native ops (torch.ops.xspeedgate_ops.*) require Kunlun XPU",
)

DEVICE = "cuda"

if _CUDA:
    from vllm_kunlun.v1.worker.gpu.input_batch import (
        combine_sampled_and_draft_tokens,
        expand_idx_mapping,
        get_num_sampled_and_rejected,
        post_update,
        post_update_num_computed_tokens,
        prepare_pos_seq_lens,
        prepare_prefill_inputs,
    )


def _i32(data):
    return torch.tensor(data, dtype=torch.int32, device=DEVICE)


def _i64(data):
    return torch.tensor(data, dtype=torch.int64, device=DEVICE)


def test_prepare_prefill_inputs_updates_prompt_and_next_token():
    input_ids = torch.full((4,), -1, dtype=torch.int32, device=DEVICE)
    next_tokens = torch.full((2,), -1, dtype=torch.int32, device=DEVICE)
    prepare_prefill_inputs(
        input_ids,
        next_tokens,
        _i32([1, 0]),
        _i32([0, 2, 4]),
        _i32([[10, 11, 12, 13], [20, 21, 22, 23]]),
        _i32([4, 3]),
        _i32([1, 0]),
    )
    assert torch.equal(
        input_ids.cpu(), torch.tensor([20, 21, 11, 12], dtype=torch.int32)
    )
    assert torch.equal(next_tokens.cpu(), torch.tensor([13, 22], dtype=torch.int32))


def test_prepare_pos_seq_lens_writes_positions_and_clears_padding():
    # positions are int64 upstream; the int32 metadata is what the op requires.
    pos = torch.full((4,), -1, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.full((4,), -1, dtype=torch.int32, device=DEVICE)
    prepare_pos_seq_lens(
        _i32([1, 0]),
        _i32([0, 2, 3]),
        _i32([4, 1]),
        pos,
        seq_lens,
    )
    assert torch.equal(pos[:3].cpu(), torch.tensor([1, 2, 4]))
    assert torch.equal(seq_lens.cpu(), torch.tensor([3, 5, 0, 0], dtype=torch.int32))


def test_combine_sampled_and_draft_tokens_splices_decode_tokens():
    input_ids = torch.tensor([7, 8, 9, 0, 0, 0], dtype=torch.int32, device=DEVICE)
    indices = combine_sampled_and_draft_tokens(
        input_ids,
        _i32([0]),
        _i64([42]),  # last_sampled_tokens: int64 upstream, wrapper casts to int32
        _i32([0, 3]),
        _i32([5]),
        _i32([3]),
        _i64([[51, 52]]),  # draft_tokens: int64 upstream, wrapper casts to int32
        _i32([0, 3]),
        num_logits=3,
    )
    assert torch.equal(indices.cpu(), torch.tensor([0, 1, 2]))
    assert torch.equal(
        input_ids.cpu(), torch.tensor([7, 51, 52, 0, 0, 0], dtype=torch.int32)
    )


def test_sampled_rejected_and_expanded_mappings():
    sampled = _i32([2, 4, 9, 9])
    result, rejected = get_num_sampled_and_rejected(
        sampled,
        _i32([5, 2, 7]),
        _i32([0, 3, 5]),
        _i32([1, 0]),
        _i32([4, 2]),
    )
    # Pin the expected dtype to int32 explicitly rather than reading it back
    # off the result: ``dtype=result.dtype`` would make the assertion pass for
    # whatever the op happens to return, defeating the check. The native ops
    # return int32 metadata here.
    assert torch.equal(result[:2].cpu(), torch.tensor([2, 0], dtype=torch.int32))
    assert torch.equal(rejected[:2].cpu(), torch.tensor([1, 0], dtype=torch.int32))

    expanded, local = expand_idx_mapping(
        _i32([3, 1]), 5, _i32([0, 2, 2]), max_expand_len=4
    )
    assert torch.equal(expanded[:2].cpu(), torch.tensor([3, 3], dtype=torch.int32))
    assert torch.equal(local[:2].cpu(), torch.tensor([0, 1], dtype=torch.int32))


def test_post_update_num_computed_tokens_uses_request_mapping():
    computed = _i32([10, 20, 30])
    post_update_num_computed_tokens(_i32([2, 0]), computed, _i32([0, 3, 4]))
    assert torch.equal(computed.cpu(), torch.tensor([11, 20, 33], dtype=torch.int32))


def test_prepare_wrappers_accept_graph_padded_query_start_loc():
    """Live FULL-graph metadata can be longer than the real request count."""
    idx_mapping = _i32([1, 0])
    padded_query_start_loc = _i32([0, 2, 3, 3, 3])
    input_ids = torch.full((4,), -1, dtype=torch.int32, device=DEVICE)
    next_tokens = torch.full((2,), -1, dtype=torch.int32, device=DEVICE)
    prepare_prefill_inputs(
        input_ids,
        next_tokens,
        idx_mapping,
        padded_query_start_loc,
        _i32([[10, 11, 12, 13], [20, 21, 22, 23]]),
        _i32([4, 3]),
        _i32([1, 0]),
    )
    pos = torch.full((4,), -1, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.full((4,), -1, dtype=torch.int32, device=DEVICE)
    prepare_pos_seq_lens(
        idx_mapping,
        padded_query_start_loc,
        _i32([4, 1]),
        pos,
        seq_lens,
    )
    assert torch.equal(
        input_ids.cpu(), torch.tensor([20, 21, 11, -1], dtype=torch.int32)
    )
    assert torch.equal(pos.cpu(), torch.tensor([1, 2, 4, -1], dtype=torch.int64))
    assert torch.equal(seq_lens.cpu(), torch.tensor([3, 5, 0, 0], dtype=torch.int32))


def test_combine_uses_valid_placeholder_for_zero_width_draft_buffer():
    """No-spec RequestState owns a [max_reqs, 0] draft tensor."""
    input_ids = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    indices = combine_sampled_and_draft_tokens(
        input_ids,
        _i32([0]),
        _i64([[42]]),
        _i32([0, 1]),
        _i32([4]),
        _i32([3]),
        torch.empty((1, 0), dtype=torch.int64, device=DEVICE),
        _i32([0, 1]),
        num_logits=1,
    )
    assert torch.equal(indices.cpu(), torch.tensor([0], dtype=torch.int64))
    assert torch.equal(input_ids.cpu(), torch.tensor([42, 0], dtype=torch.int32))


def test_empty_batch_boundaries_are_well_defined():
    empty_idx = torch.empty(0, dtype=torch.int32, device=DEVICE)
    padded_qsl = _i32([0, 0, 0])
    pos = torch.full((3,), 777, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.full((3,), 777, dtype=torch.int32, device=DEVICE)

    prepare_prefill_inputs(
        torch.empty(0, dtype=torch.int32, device=DEVICE),
        torch.empty(0, dtype=torch.int32, device=DEVICE),
        empty_idx,
        padded_qsl,
        torch.empty((0, 0), dtype=torch.int32, device=DEVICE),
        torch.empty(0, dtype=torch.int32, device=DEVICE),
        torch.empty(0, dtype=torch.int32, device=DEVICE),
    )
    prepare_pos_seq_lens(
        empty_idx,
        padded_qsl,
        torch.empty(0, dtype=torch.int32, device=DEVICE),
        pos,
        seq_lens,
    )
    expanded, local = expand_idx_mapping(empty_idx, 0, _i32([0]), 0)

    assert expanded.numel() == 0
    assert local.numel() == 0
    assert bool((seq_lens == 0).all())
    assert bool((pos == 777).all())


def test_post_update_preserves_native_int32_state_shape(monkeypatch):
    """The thin wrapper must not flatten an already ABI-compatible buffer."""
    observed = {}

    def fake_post_update(
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
        observed["shape"] = last_sampled_tokens.shape
        observed["data_ptr"] = last_sampled_tokens.data_ptr()

    monkeypatch.setattr(torch.ops.xspeedgate_ops, "post_update", fake_post_update)
    last_sampled_tokens = torch.zeros((3, 1), dtype=torch.int32, device=DEVICE)
    post_update(
        _i32([0]),
        _i32([0, 0, 0]),
        last_sampled_tokens,
        None,
        _i32([[7]]),
        _i32([1]),
        _i32([0]),
        _i32([0, 1]),
        torch.zeros((3, 4), dtype=torch.int32, device=DEVICE),
        _i32([0, 0, 0]),
    )

    assert observed["shape"] == torch.Size((3, 1))
    assert observed["data_ptr"] == last_sampled_tokens.data_ptr()


def test_post_update_casts_tokens_writes_back_and_skips_negative_mapping():
    idx_mapping = _i32([1, -1])
    num_computed_tokens = _i32([10, 20, 30])
    last_sampled_tokens = torch.full((3, 1), -1, dtype=torch.int64, device=DEVICE)
    all_token_ids = torch.zeros((3, 8), dtype=torch.int32, device=DEVICE)
    total_len = _i32([1, 2, 3])

    post_update(
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        None,
        _i64([[7, 8], [9, 10]]),
        _i32([2, 1]),
        _i32([0, 0]),
        _i32([0, 2, 3]),
        all_token_ids,
        total_len,
    )

    assert torch.equal(
        num_computed_tokens.cpu(), torch.tensor([10, 22, 30], dtype=torch.int32)
    )
    assert torch.equal(
        last_sampled_tokens.cpu(),
        torch.tensor([[-1], [8], [-1]], dtype=torch.int64),
    )
    assert torch.equal(total_len.cpu(), torch.tensor([1, 4, 3], dtype=torch.int32))
    assert torch.equal(
        all_token_ids[1].cpu(),
        torch.tensor([0, 0, 7, 8, 0, 0, 0, 0], dtype=torch.int32),
    )
    assert bool((all_token_ids[0] == 0).all())
    assert bool((all_token_ids[2] == 0).all())
