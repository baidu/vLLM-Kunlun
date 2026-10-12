# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-device tests for the Kunlun native-op sampling wrappers.

``apply_temperature`` / ``gumbel_sample`` / ``apply_min_p`` forward to
``torch.ops.xspeedgate_ops.*``, which register for CUDA/XLA/XPU only (Kunlun XPU
presents as CUDA via ``torch_xmlir``), so these tests are device-gated.

Live-path metadata uses int32 request mappings and int64 seed/position tensors.
The wrapper also owns three contracts beyond the native kernels: the sample
result is flattened from ``[n, 1]`` to ``[n]``, existing ``-inf`` masks survive
temperature scaling, and negative request mappings select argmax as upstream
does. Those local compatibility paths are asserted here as well.
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
    from vllm_kunlun.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
    from vllm_kunlun.v1.worker.gpu.sample.min_p import apply_min_p
    from vllm_kunlun.v1.worker.gpu.spec_decode.rejection_sampler import _flatten_sampled
    from vllm_kunlun.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )


def _i32(data):
    return torch.tensor(data, dtype=torch.int32, device=DEVICE)


def _i64(data):
    return torch.tensor(data, dtype=torch.int64, device=DEVICE)


def _f32(data):
    return torch.tensor(data, dtype=torch.float32, device=DEVICE)


def test_apply_temperature_keeps_greedy_rows_unchanged():
    logits = _f32([[2.0, 4.0], [2.0, 4.0]])
    apply_temperature(logits, _i32([0, 1]), _f32([0.0, 2.0]))
    assert torch.equal(logits[0].cpu(), torch.tensor([2.0, 4.0]))
    assert torch.equal(logits[1].cpu(), torch.tensor([1.0, 2.0]))


def test_apply_temperature_preserves_negative_infinity_masks():
    logits = _f32([[1.0, float("-inf"), 3.0]])
    apply_temperature(logits, _i32([0]), _f32([0.5]))
    assert torch.equal(logits[0, [0, 2]].cpu(), torch.tensor([2.0, 6.0]))
    assert torch.isneginf(logits[0, 1])
    assert not torch.isnan(logits).any()


def test_gumbel_sample_is_argmax_for_greedy_requests():
    result = gumbel_sample(
        _f32([[1.0, 4.0, 2.0]]),
        _i32([0]),
        _f32([0.0]),
        _i64([17]),
        _i64([3]),
        apply_temperature=True,
    )
    assert result.dim() == 1
    assert torch.equal(result.cpu(), torch.tensor([1]))


def test_gumbel_sample_uses_argmax_for_negative_request_mapping():
    result = gumbel_sample(
        _f32([[1.0, 4.0, 2.0]]),
        _i32([-1]),
        _f32([1.0]),
        _i64([17]),
        _i64([3]),
        apply_temperature=False,
    )
    assert torch.equal(result.cpu(), torch.tensor([1]))


def test_gumbel_sample_is_reproducible_for_same_inputs():
    args = dict(
        logits=torch.zeros((2, 4), dtype=torch.float32, device=DEVICE),
        expanded_idx_mapping=_i32([0, 1]),
        temperature=_f32([1.0, 1.0]),
        seed=_i64([10, 20]),
        pos=_i64([0, 1]),
        apply_temperature=False,
    )
    assert torch.equal(gumbel_sample(**args).cpu(), gumbel_sample(**args).cpu())


def test_apply_min_p_filters_logits_below_relative_threshold():
    logits = _f32([[1.0, 4.0, 2.0]])
    apply_min_p(logits, _i32([0]), _f32([0.5]))
    assert torch.isneginf(logits[0, 0])
    assert torch.isneginf(logits[0, 2])
    assert logits[0, 1] == 4.0


def test_apply_min_p_zero_is_noop():
    logits = _f32([[1.0, 4.0, 2.0]])
    original = logits.clone()
    apply_min_p(logits, _i32([0]), _f32([0.0]))
    assert torch.equal(logits, original)


def test_empty_sampling_batches_are_well_defined():
    logits = torch.empty((0, 4), dtype=torch.float32, device=DEVICE)
    mapping = torch.empty(0, dtype=torch.int32, device=DEVICE)
    values = torch.empty(0, dtype=torch.float32, device=DEVICE)

    apply_temperature(logits, mapping, values)
    apply_min_p(logits, mapping, values)
    result = gumbel_sample(
        logits,
        mapping,
        values,
        torch.empty(0, dtype=torch.int64, device=DEVICE),
        torch.empty(0, dtype=torch.int64, device=DEVICE),
        apply_temperature=False,
    )

    assert result.shape == (0,)
    assert result.dtype == torch.int64
    assert result.device == logits.device


@pytest.mark.parametrize("use_fp64", [False, True])
def test_mtp_gumbel_updates_only_selected_request_columns(use_fp64):
    logits = _f32([[1, 4, 2], [2, 1, 3], [3, 2, 1]]).half()
    mapping = _i32([1, 0, -1])
    temperature = _f32([0.0, 0.5])
    processed = torch.full((2, 3, 3), 77.0, device=DEVICE)
    args = dict(
        logits=logits,
        expanded_idx_mapping=mapping,
        temperature=temperature,
        seed=_i64([17, 29]),
        pos=_i64([3, 4, 5]),
        apply_temperature=True,
        output_processed_logits=processed,
        output_processed_logits_col=_i64([2, 0, 1]),
        use_fp64=use_fp64,
    )
    result = gumbel_sample(**args)
    assert result.shape == (3,)
    assert result[1:].cpu().tolist() == [2, 0]
    expected = torch.full((2, 3, 3), 77.0)
    expected[1, 2] = logits[0].float().cpu() / 0.5
    expected[0, 0] = logits[1].float().cpu()
    torch.testing.assert_close(processed.cpu(), expected)
    assert torch.equal(result.cpu(), gumbel_sample(**args).cpu())


@pytest.mark.parametrize("steps", [1, 3, 32])
@pytest.mark.parametrize("all_greedy", [None, True])
def test_mtp_rejection_accepts_matching_drafts_and_stops_at_mismatch(steps, all_greedy):
    length = steps + 1
    # Request 0 accepts every draft; request 1 rejects its first draft.
    logits = _f32([[0, 3, 1]] * (2 * length))
    drafts = _i32([1] * (2 * length))
    drafts[length + 1] = 2
    sampled, counts = rejection_sample(
        logits,
        None,
        drafts,
        _i32([0, length, 2 * length]),
        _i64(list(range(2 * length))),
        _i32([0, 1]),
        _i32([0] * length + [1] * length),
        _i32(list(range(length)) * 2),
        _f32([0, 0]),
        _i64([17, 29]),
        steps,
        all_greedy=all_greedy,
    )
    assert counts.cpu().tolist() == [length, 1]
    assert sampled[0].cpu().tolist() == [1] * length
    assert sampled[1].cpu().tolist() == [1] + [-1] * steps


@pytest.mark.parametrize("steps", [0, 33])
def test_mtp_rejection_rejects_unsupported_step_counts(steps):
    with pytest.raises(ValueError, match="1 to 32"):
        rejection_sample(*([None] * 10), steps)


@pytest.mark.parametrize(
    "options",
    [{"synthetic_conditional_rates": [0.5]}, {"use_block_verification": True}],
)
def test_mtp_rejection_rejects_unsupported_modes(options):
    with pytest.raises(NotImplementedError, match="standard rejection only"):
        rejection_sample(*([None] * 10), 3, **options)


def test_mtp_flatten_preserves_live_tokens_and_zeros_rejected_slots():
    sampled = _i64([[11, 12, -1, -1, 99], [21, -1, -1, -1, 99]])[:, :4]
    flat = _i64([-999] * 6)
    _flatten_sampled(flat, sampled, sampled.stride(0), _i32([2, 1]), _i32([0, 4, 6]))
    assert flat.cpu().tolist() == [11, 12, 0, 0, 21, 0]
