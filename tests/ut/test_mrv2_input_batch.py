# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input-batch wrapper arguments, dtype adaptation and writeback on CPU."""
from unittest.mock import Mock
import pytest
import torch


@pytest.fixture
def input_batch(load_kunlun_override, native_ops):
    return load_kunlun_override(
        "v1/worker/gpu/input_batch.py", "vllm.v1.worker.gpu.input_batch"
    )


def test_wrappers_are_registered(input_batch):
    (module, upstream) = input_batch
    for name in [
        "prepare_prefill_inputs",
        "prepare_pos_seq_lens",
        "combine_sampled_and_draft_tokens",
        "get_num_sampled_and_rejected",
        "post_update",
        "post_update_num_computed_tokens",
        "expand_idx_mapping",
    ]:
        assert getattr(upstream, name) is getattr(module, name)


@pytest.mark.parametrize("name", ["prepare_prefill_inputs", "prepare_pos_seq_lens"])
def test_prepare_slices_graph_padded_starts(input_batch, native_ops, name):
    (module, _) = input_batch
    mapping = torch.tensor([1, 0], dtype=torch.int32)
    starts = torch.tensor([0, 2, 3, 3, 3], dtype=torch.int32)
    buffers = [torch.zeros(4, dtype=torch.int32) for _ in range(5)]
    native = Mock()
    setattr(native_ops, name, native)
    if name == "prepare_prefill_inputs":
        args = [buffers[0], buffers[1], mapping, starts, *buffers[2:]]
        starts_index = 3
    else:
        args = [mapping, starts, *buffers[:3]]
        starts_index = 1
    getattr(module, name)(*args)
    native.assert_called_once()
    forwarded = native.call_args.args
    assert len(forwarded) == len(args)
    for index, (actual, wanted) in enumerate(zip(forwarded, args)):
        if index == starts_index:
            assert actual.tolist() == [0, 2, 3]
            assert actual.data_ptr() == starts.data_ptr()
        else:
            assert actual is wanted


@pytest.mark.parametrize(
    "dtype,empty_drafts",
    [(torch.int64, False), (torch.int64, True), (torch.int32, False)],
)
def test_combine_converts_readonly_tokens_and_returns_native_result(
    input_batch, native_ops, dtype, empty_drafts
):
    (module, _) = input_batch
    tokens = torch.zeros(6, dtype=torch.int32)
    mapping = torch.tensor([1, 0], dtype=torch.int32)
    last = torch.tensor([[41], [42]], dtype=dtype)
    drafts = (
        torch.empty((2, 0), dtype=dtype)
        if empty_drafts
        else torch.tensor([[51, 52], [61, 62]], dtype=dtype)
    )
    (original_last, original_drafts) = (last.clone(), drafts.clone())
    starts = torch.tensor([0, 3, 6], dtype=torch.int32)
    seq_lens = torch.tensor([4, 5], dtype=torch.int32)
    prefill = torch.tensor([1, 1], dtype=torch.int32)
    logits = torch.tensor([0, 3, 6], dtype=torch.int32)
    result = torch.arange(6, dtype=torch.int64)
    native_ops.combine_sampled_and_draft_tokens = Mock(return_value=result)
    actual = module.combine_sampled_and_draft_tokens(
        tokens, mapping, last, starts, seq_lens, prefill, drafts, logits, 6, 0
    )
    assert actual is result
    native_ops.combine_sampled_and_draft_tokens.assert_called_once()
    args = native_ops.combine_sampled_and_draft_tokens.call_args.args
    assert args[0] is tokens and args[1] is mapping
    assert args[2].dtype == torch.int32 and torch.equal(args[2], last.int())
    assert all((a is b for (a, b) in zip(args[3:6], [starts, seq_lens, prefill])))
    assert args[6].dtype == torch.int32
    if empty_drafts:
        assert args[6].shape == (2, 1) and args[6].data_ptr() == args[2].data_ptr()
    else:
        assert torch.equal(args[6], drafts.int())
    assert args[7] is logits and args[8:] == (6, 0)
    assert torch.equal(last, original_last) and torch.equal(drafts, original_drafts)


@pytest.mark.parametrize(
    "dtype,shape,with_bins",
    [
        (torch.int32, (3, 1), False),
        (torch.int64, (3, 1), True),
        (torch.int64, (3,), False),
    ],
)
def test_post_update_preserves_shape_and_copies_back(
    input_batch, native_ops, dtype, shape, with_bins
):
    (module, _) = input_batch
    mapping = torch.tensor([2, -1], dtype=torch.int32)
    computed = torch.zeros(3, dtype=torch.int32)
    last = torch.full(shape, -1, dtype=dtype)
    bins = torch.zeros((3, 100), dtype=torch.int32) if with_bins else None
    sampled = torch.tensor([[7], [8]], dtype=torch.int64)
    counts = torch.tensor([1, 1], dtype=torch.int32)
    rejected = torch.zeros(2, dtype=torch.int32)
    starts = torch.tensor([0, 1, 2], dtype=torch.int32)
    all_tokens = torch.zeros((3, 8), dtype=torch.int32)
    total = torch.zeros(3, dtype=torch.int32)

    def update(*args):
        assert args[2].dtype == torch.int32 and args[2].shape == shape
        args[2].view(-1)[2] = 73

    native_ops.post_update = Mock(side_effect=update)
    module.post_update(
        mapping,
        computed,
        last,
        bins,
        sampled,
        counts,
        rejected,
        starts,
        all_tokens,
        total,
    )
    native_ops.post_update.assert_called_once()
    args = native_ops.post_update.call_args.args
    assert args[0] is mapping and args[1] is computed and (args[3] is bins)
    assert args[4].dtype == torch.int32 and torch.equal(args[4], sampled.int())
    assert all(
        (
            a is b
            for (a, b) in zip(args[5:], [counts, rejected, starts, all_tokens, total])
        )
    )
    if dtype == torch.int32:
        assert args[2] is last
    assert last.shape == shape and last.dtype == dtype
    assert last.view(-1).tolist() == [-1, -1, 73]


def test_remaining_wrappers_forward_arguments_and_results(input_batch, native_ops):
    (module, _) = input_batch
    mapping = torch.tensor([1, 0], dtype=torch.int32)
    values = [torch.zeros(3, dtype=torch.int32) for _ in range(4)]
    (sampled, seq_lens, starts, prefill) = values
    result = (sampled, torch.ones_like(sampled))
    native_ops.get_num_sampled_and_rejected = Mock(return_value=result)
    assert (
        module.get_num_sampled_and_rejected(sampled, seq_lens, starts, mapping, prefill)
        is result
    )
    args = native_ops.get_num_sampled_and_rejected.call_args.args
    assert all(
        (a is b for (a, b) in zip(args, [sampled, seq_lens, starts, mapping, prefill]))
    )
    native_ops.post_update_num_computed_tokens = Mock()
    module.post_update_num_computed_tokens(mapping, seq_lens, starts)
    args = native_ops.post_update_num_computed_tokens.call_args.args
    assert all((a is b for (a, b) in zip(args, [mapping, seq_lens, starts])))
    expanded = (mapping.clone(), mapping.clone())
    native_ops.expand_idx_mapping = Mock(return_value=expanded)
    assert module.expand_idx_mapping(mapping, 5, starts, 4) is expanded
    args = native_ops.expand_idx_mapping.call_args.args
    assert (
        args[0] is mapping and args[1] == 5 and (args[2] is starts) and (args[3] == 4)
    )


def test_empty_batches_do_not_call_native_operators(input_batch, native_ops):
    (module, _) = input_batch
    empty = torch.empty(0, dtype=torch.int32)
    starts = torch.zeros(3, dtype=torch.int32)
    pos = torch.full((3,), 777, dtype=torch.int64)
    lengths = torch.full((3,), 777, dtype=torch.int32)
    module.prepare_prefill_inputs(empty, empty, empty, starts, empty, empty, empty)
    module.prepare_pos_seq_lens(empty, starts, empty, pos, lengths)
    combined = module.combine_sampled_and_draft_tokens(
        empty, empty, empty, starts, empty, empty, empty, starts, 0
    )
    (sampled, rejected) = module.get_num_sampled_and_rejected(
        empty, empty, starts, empty, empty
    )
    module.post_update(
        empty, empty, empty, None, empty, empty, empty, None, empty, empty
    )
    module.post_update_num_computed_tokens(empty, empty, starts)
    (expanded, local) = module.expand_idx_mapping(empty, 0, starts, 0)
    assert combined.shape == (0,) and combined.dtype == torch.int64
    assert sampled is empty and rejected.shape == (0,)
    assert expanded.numel() == local.numel() == 0
    assert bool((lengths == 0).all()) and bool((pos == 777).all())
    assert vars(native_ops) == {}


def test_combine_rejects_invalid_sample_count(input_batch):
    (module, _) = input_batch
    with pytest.raises(AssertionError, match="0 or 1"):
        module.combine_sampled_and_draft_tokens(*[None] * 8, 0, 2)
