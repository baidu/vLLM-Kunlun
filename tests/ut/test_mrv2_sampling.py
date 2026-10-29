# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling wrapper dispatch and output contracts on CPU."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def sampling(load_kunlun_override, native_ops):
    gumbel, gumbel_upstream = load_kunlun_override(
        "v1/worker/gpu/sample/gumbel.py", "vllm.v1.worker.gpu.sample.gumbel"
    )
    min_p, min_p_upstream = load_kunlun_override(
        "v1/worker/gpu/sample/min_p.py", "vllm.v1.worker.gpu.sample.min_p"
    )
    return SimpleNamespace(
        gumbel=gumbel,
        min_p=min_p,
        gumbel_upstream=gumbel_upstream,
        min_p_upstream=min_p_upstream,
    )


def test_sampling_wrappers_are_registered(sampling):
    assert (
        sampling.gumbel_upstream.apply_temperature is sampling.gumbel.apply_temperature
    )
    assert sampling.gumbel_upstream.gumbel_sample is sampling.gumbel.gumbel_sample
    assert sampling.min_p_upstream.apply_min_p is sampling.min_p.apply_min_p


def test_temperature_restores_excluded_tokens(sampling, native_ops):
    logits = torch.tensor([[1.0, float("-inf"), 3.0]])
    mapping = torch.tensor([0], dtype=torch.int32)
    temperature = torch.tensor([0.5])
    native_ops.apply_temperature = Mock(
        side_effect=lambda target, *_: target.fill_(6.0)
    )
    sampling.gumbel.apply_temperature(logits, mapping, temperature)
    native_ops.apply_temperature.assert_called_once()
    args = native_ops.apply_temperature.call_args.args
    assert all((a is b for (a, b) in zip(args, [logits, mapping, temperature])))
    assert logits[0, [0, 2]].tolist() == [6.0, 6.0]
    assert torch.isneginf(logits[0, 1]) and (not bool(torch.isnan(logits).any()))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_fp64,apply_temperature", [(False, False), (True, True)])
def test_gumbel_without_cache_reuses_existing_operator(
    sampling, native_ops, dtype, use_fp64, apply_temperature
):
    logits = torch.tensor(
        [[1.0, 4.0, 2.0], [7.0, 1.0, 3.0], [1.0, 2.0, 9.0]], dtype=dtype
    )
    mapping = torch.tensor([1, 0, -1], dtype=torch.int32)
    temperature = torch.tensor([0.0, 0.5])
    seed = torch.tensor([17, 29], dtype=torch.int64)
    pos = torch.tensor([3, 4, 5], dtype=torch.int64)
    expected = torch.tensor([[2], [0], [2]], dtype=torch.int64)
    native_ops.gumbel_sample = Mock(return_value=expected)
    result = sampling.gumbel.gumbel_sample(
        logits, mapping, temperature, seed, pos, apply_temperature, use_fp64=use_fp64
    )
    native_ops.gumbel_sample.assert_called_once()
    args = native_ops.gumbel_sample.call_args.args
    processed = logits.float()
    if apply_temperature:
        processed = processed * torch.tensor([2.0, 1.0, 1.0])[:, None]
    assert args[0].dtype == torch.float32 and args[0].is_contiguous()
    torch.testing.assert_close(args[0], processed, rtol=0, atol=0)
    assert args[1].tolist() == [1, 0, 2]
    assert args[2].tolist() == [0.0, 0.5, 0.0]
    assert args[3].tolist() == [17, 29, 0]
    assert args[4] is pos and args[5:] == (False, None, None, use_fp64)
    assert result.shape == (3,) and result.dtype == torch.int64
    assert result.data_ptr() == expected.data_ptr()
    assert "apply_write" not in vars(native_ops)


def test_min_p_forwards_inputs_and_preserves_native_write(sampling, native_ops):
    logits = torch.tensor([[1.0, 4.0, 2.0]])
    mapping = torch.tensor([0], dtype=torch.int32)
    threshold = torch.tensor([0.5])
    native_ops.min_p_inplace = Mock(side_effect=lambda target, *_: target.fill_(-99))
    sampling.min_p.apply_min_p(logits, mapping, threshold)
    native_ops.min_p_inplace.assert_called_once()
    args = native_ops.min_p_inplace.call_args.args
    assert all((a is b for (a, b) in zip(args, [logits, mapping, threshold])))
    assert bool((logits == -99).all())


def test_empty_batches_do_not_call_sampling_operators(sampling, native_ops):
    logits = torch.empty((0, 4))
    mapping = torch.empty(0, dtype=torch.int32)
    values = torch.empty(0)
    integers = torch.empty(0, dtype=torch.int64)
    sampling.gumbel.apply_temperature(logits, mapping, values)
    sampling.min_p.apply_min_p(logits, mapping, values)
    result = sampling.gumbel.gumbel_sample(
        logits, mapping, values, integers, integers, False
    )
    assert result.shape == (0,) and result.dtype == torch.int64
    assert result.device.type == "cpu" and vars(native_ops) == {}


@pytest.fixture
def rejection_wrappers(load_kunlun_override, stub_module_tree, monkeypatch, native_ops):
    utilities, utilities_upstream = load_kunlun_override(
        "v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils",
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_kunlun.v1.worker.gpu.spec_decode.rejection_sampler_utils",
        utilities,
    )
    stub_module_tree(
        "vllm.v1.worker.gpu.input_batch",
        InputBatch=type("InputBatch", (), {}),
        get_num_sampled_and_rejected=Mock(),
    )
    stub_module_tree("vllm.v1.worker.gpu.metrics.logits", get_num_nans=Mock())
    stub_module_tree("vllm.v1.worker.gpu.sample.output", SamplerOutput=SimpleNamespace)
    sampler, upstream = load_kunlun_override(
        "v1/worker/gpu/spec_decode/rejection_sampler.py",
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
        RejectionSampler=type("RejectionSampler", (), {}),
    )
    return SimpleNamespace(
        utilities=utilities,
        utilities_upstream=utilities_upstream,
        sampler=sampler,
        upstream=upstream,
    )


def _apply_cached_logits(output, indices, starts, contents, ends):
    begin = 0
    for row, start, end in zip(indices.tolist(), starts.tolist(), ends.tolist()):
        output[row, start : start + end - begin] = contents[begin:end]
        begin = end


@pytest.mark.parametrize("column", [None, 2, "vector"])
@pytest.mark.parametrize("apply_temperature", [False, True])
def test_mtp_gumbel_writes_cache_with_existing_operator(
    sampling, native_ops, column, apply_temperature
):
    logits = torch.tensor([[1, 4, 2], [2, 1, 3], [1, 5, 2]], dtype=torch.float16)
    mapping = torch.tensor([2, 0, -1], dtype=torch.int32)
    temperature = torch.tensor([0.5, 1.0, 0.0])
    seed, pos = torch.tensor([17, 29, 41]), torch.tensor([3, 4, 5])
    cache = torch.full((3, 3, 3), 77.0)
    columns = (
        None
        if column is None
        else torch.tensor([2, 0, 1]) if column == "vector" else torch.tensor(column)
    )
    expected_cache = cache.clone()
    first_col = 2 if column in (2, "vector") else 0
    second_col = 2 if column == 2 else 0
    expected_cache[2, first_col] = torch.tensor([1.0, 4.0, 2.0])
    expected_cache[0, second_col] = torch.tensor(
        [4.0, 2.0, 6.0] if apply_temperature else [2.0, 1.0, 3.0]
    )
    native_ops.gumbel_sample = Mock(return_value=torch.tensor([[1], [2], [1]]))
    native_ops.apply_write = Mock(side_effect=_apply_cached_logits)
    result = sampling.gumbel.gumbel_sample(
        logits, mapping, temperature, seed, pos, apply_temperature, cache, columns, True
    )
    native_ops.gumbel_sample.assert_called_once()
    sample_args = native_ops.gumbel_sample.call_args.args
    assert sample_args[0].dtype == torch.float32
    assert sample_args[1].tolist() == [2, 0, 3]
    assert sample_args[2].tolist() == [0.5, 1.0, 0.0, 0.0]
    assert sample_args[3].tolist() == [17, 29, 41, 0]
    assert sample_args[4] is pos and sample_args[5:] == (False, None, None, True)
    native_ops.apply_write.assert_called_once()
    write_args = native_ops.apply_write.call_args.args
    assert write_args[0].data_ptr() == cache.data_ptr()
    assert (
        write_args[1].dtype == write_args[2].dtype == write_args[4].dtype == torch.int32
    )
    assert write_args[4].tolist() == [3, 6, 6]
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    assert result.tolist() == [1, 2, 1]


def test_mtp_gumbel_padding_preserves_cache(sampling, native_ops):
    logits = torch.tensor([[1.0, 4.0, 2.0], [7.0, 1.0, 3.0]])
    cache = torch.full((2, 3, 3), 77.0)
    native_ops.gumbel_sample = Mock(
        side_effect=lambda values, *_: values.argmax(-1)[:, None]
    )
    native_ops.apply_write = Mock(side_effect=_apply_cached_logits)
    result = sampling.gumbel.gumbel_sample(
        logits,
        torch.full((2,), -1, dtype=torch.int32),
        torch.tensor([0.5, 1.0]),
        torch.tensor([17, 29]),
        torch.tensor([3, 4]),
        True,
        cache,
        torch.tensor(1),
    )
    assert result.tolist() == [1, 0]
    torch.testing.assert_close(cache, torch.full_like(cache, 77.0), rtol=0, atol=0)
    assert native_ops.apply_write.call_args.args[4].tolist() == [0, 0]


@pytest.mark.parametrize("all_greedy", [None, True, False])
@pytest.mark.parametrize("temperature_values", [[0.0, 0.0], [0.5, 0.0]])
def test_mtp_rejection_forwards_modes_and_native_result(
    rejection_wrappers, native_ops, all_greedy, temperature_values
):
    module = rejection_wrappers.utilities
    arrays = [torch.zeros(2, dtype=torch.int32) for _ in range(8)]
    arrays[1] = None
    arrays[5] = torch.tensor([1, 0], dtype=torch.int32)
    temperature = torch.tensor(temperature_values)
    seed = torch.tensor([17, 29])
    result = (torch.tensor([[1, -1]]), torch.tensor([1]))
    native_ops.rejection_sample_fused_kernel = Mock(return_value=result)
    actual = module.rejection_sample(
        *arrays, temperature, seed, 3, use_fp64=True, all_greedy=all_greedy
    )
    native_ops.rejection_sample_fused_kernel.assert_called_once()
    args = native_ops.rejection_sample_fused_kernel.call_args.args
    assert all(a is b for a, b in zip(args[:10], [*arrays, temperature, seed]))
    expected_greedy = (
        all(value == 0 for value in temperature_values)
        if all_greedy is None
        else all_greedy
    )
    assert args[10:] == (3, expected_greedy, True, 0) and actual is result
    assert (
        rejection_wrappers.utilities_upstream.rejection_sample
        is module.rejection_sample
    )


@pytest.mark.parametrize("steps", [0, 33])
def test_mtp_rejection_rejects_unsupported_step_counts(rejection_wrappers, steps):
    with pytest.raises(ValueError, match="1 to 32"):
        rejection_wrappers.utilities.rejection_sample(*([None] * 10), steps)


@pytest.mark.parametrize(
    "options",
    [{"synthetic_conditional_rates": [0.5]}, {"use_block_verification": True}],
)
def test_mtp_rejection_rejects_unsupported_modes(rejection_wrappers, options):
    with pytest.raises(NotImplementedError, match="standard rejection only"):
        rejection_wrappers.utilities.rejection_sample(*([None] * 10), 3, **options)


def test_mtp_flatten_passes_padded_stride_to_native(rejection_wrappers, native_ops):
    sampled = torch.zeros((2, 5), dtype=torch.int64)[:, :4]
    flat = torch.full((6,), -999, dtype=torch.int64)
    counts = torch.tensor([2, 1], dtype=torch.int32)
    starts = torch.tensor([0, 4, 6], dtype=torch.int32)
    native_ops.flatten_sampled_kernel = Mock()
    kernel = rejection_wrappers.upstream._flatten_sampled_kernel
    kernel[(1,)](flat, sampled, sampled.stride(0), counts, starts)
    native_ops.flatten_sampled_kernel.assert_called_once()
    args = native_ops.flatten_sampled_kernel.call_args.args
    assert all(a is b for a, b in zip(args[:4], [flat, sampled, counts, starts]))
    assert args[4:] == (2, 5)
