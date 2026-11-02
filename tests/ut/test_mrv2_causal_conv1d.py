# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convolution wrapper arguments, input protection and padding on CPU."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def convolution(load_kunlun_override, stub_module_tree):
    native = Mock(return_value=0)
    stub_module_tree("kunlun_ops", causal_conv1d_update=native)
    module, _ = load_kunlun_override(
        "ops/mamba/causal_conv1d.py", "vllm.v1.attention.backends.utils", PAD_SLOT_ID=-1
    )
    return module.causal_conv1d_update, native


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("with_bias,silu", [(False, False), (True, True)])
def test_speculative_update_adapts_inputs_and_masks_padding(
    convolution, dtype, with_bias, silu
):
    update, native = convolution
    batch, steps, channels = 3, 2, 4
    x = torch.arange(batch * steps * channels, dtype=dtype).reshape(-1, channels)
    before = x.clone()
    state = torch.zeros(5, steps + 2, channels, dtype=dtype)
    weight = torch.ones(channels, 4, dtype=dtype)
    bias = torch.ones(channels, dtype=dtype) if with_bias else None
    indices = torch.tensor([2, 0, -1], dtype=torch.int32)
    accepted = torch.tensor([2, 0, 0], dtype=torch.int32)

    def write_output(value, *_args, **_kwargs):
        value.copy_(torch.linspace(-1, 1, value.numel()).reshape_as(value))
        return 0

    native.side_effect = write_output
    result = update(
        x,
        state,
        weight,
        bias,
        silu,
        conv_state_indices=indices,
        num_accepted_tokens=accepted,
        max_query_len=steps,
    )
    native.assert_called_once()
    args, kwargs = native.call_args
    assert args[0].shape == (batch, steps, channels)
    assert args[0].is_contiguous() and args[0].data_ptr() != x.data_ptr()
    assert args[1] is state and args[2] is weight
    assert kwargs["conv_state_indices"].tolist() == [2, 0, 0]
    assert kwargs["num_accepted_tokens"] is accepted
    assert kwargs["silu_activation"] is False and kwargs["pad_slot_id"] == 0
    assert kwargs["is_ncw"] is False
    assert (
        kwargs["cache_seqlens"] is None and kwargs["intermediate_conv_window"] is None
    )
    if with_bias:
        assert kwargs["bias"].dtype == torch.float32
        torch.testing.assert_close(kwargs["bias"], bias.float())
    else:
        assert kwargs["bias"] is None
    expected = (
        torch.linspace(-1, 1, x.numel()).reshape(batch, steps, channels).to(dtype)
    )
    if silu:
        expected = F.silu(expected)
    expected[1:] = 0
    torch.testing.assert_close(result, expected.reshape_as(x), rtol=0, atol=0)
    assert torch.equal(x, before) and indices.tolist() == [2, 0, -1]


def test_empty_speculative_batch_does_not_call_native(convolution):
    update, native = convolution
    x = torch.empty((0, 4))
    result = update(
        x,
        torch.zeros(3, 4, 4),
        torch.ones(4, 3),
        conv_state_indices=torch.empty(0, dtype=torch.int32),
        num_accepted_tokens=torch.empty(0, dtype=torch.int32),
        max_query_len=2,
    )
    assert result.shape == x.shape and result.dtype == x.dtype
    native.assert_not_called()


def test_speculative_update_reports_native_failure(convolution):
    update, native = convolution
    native.return_value = 7
    with pytest.raises(RuntimeError, match="causal_conv1d_update failed: 7"):
        update(
            torch.ones(2, 4),
            torch.zeros(3, 4, 4),
            torch.ones(4, 3),
            conv_state_indices=torch.tensor([1], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
            max_query_len=2,
        )


def test_decode_preserves_native_inplace_output(convolution):
    update, native = convolution
    x = torch.zeros(2, 4)
    state, weight = torch.zeros(3, 4, 4), torch.ones(4, 3)
    indices = torch.tensor([1, 2], dtype=torch.int32)

    def write_output(value, *_args, **_kwargs):
        value.fill_(9)
        return 0

    native.side_effect = write_output
    result = update(x, state, weight, activation=True, conv_state_indices=indices)
    native.assert_called_once()
    args, kwargs = native.call_args
    assert args[0].shape == (2, 1, 4) and args[0].data_ptr() == x.data_ptr()
    assert args[1] is state and args[2] is weight
    assert kwargs["conv_state_indices"] is indices
    assert kwargs["silu_activation"] is True and kwargs["pad_slot_id"] == -1
    assert "num_accepted_tokens" not in kwargs
    assert result.shape == x.shape and result.data_ptr() == x.data_ptr()
    assert torch.equal(result, torch.full_like(x, 9))
