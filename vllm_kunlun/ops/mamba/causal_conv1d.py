# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py

from typing import Optional, Union

import kunlun_ops
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    query_start_loc_cpu: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    cache_indices_cpu: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    has_initial_state_cpu: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    if not x.is_contiguous():
        x = x.contiguous()

    out = torch.empty_like(x)

    x_shape = x.shape
    dim = x_shape[-1]
    cu_seqlen = x_shape[-2]
    width = weight.shape[-1]

    assert (
        conv_states is not None
    ), "conv_states is required for kunlun causal_conv1d_fn"
    num_cache_lines = conv_states.shape[0]
    state_width = conv_states.shape[-2]
    stride = conv_states.stride(0)
    assert (
        query_start_loc is not None
    ), "query_start_loc is required for kunlun causal_conv1d_fn"
    batch_size = query_start_loc.shape[0] - 1

    kunlun_ops.causal_conv1d_fn(
        x,
        out,
        dim,
        cu_seqlen,
        weight,
        width,
        conv_states,
        num_cache_lines,
        state_width,
        query_start_loc_cpu,
        query_start_loc,
        batch_size,
        bias,
        cache_indices_cpu=cache_indices_cpu,
        cache_indices_xpu=cache_indices,
        has_initial_state_cpu=has_initial_state_cpu,
        has_initial_state_xpu=has_initial_state,
        act="SWISH",
        state_seq_stride=stride,
    )

    return out


def causal_conv1d_update_spec_graphsafe(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    conv_state_indices_cpu: Optional[torch.Tensor] = None,
    num_accepted_tokens_cpu: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if hidden_states.dim() != 3:
        raise ValueError(
            "causal_conv1d_update_spec_graphsafe expects "
            "[batch, seq_len, hidden] hidden_states."
        )
    if (
        conv_state_indices is None
        or conv_state_indices_cpu is None
        or num_accepted_tokens is None
        or num_accepted_tokens_cpu is None
    ):
        raise ValueError(
            "causal_conv1d_update_spec_graphsafe requires CPU and XPU "
            "conv_state_indices and num_accepted_tokens."
        )
    out = torch.empty_like(hidden_states)
    kunlun_ops.causal_conv1d_update(
        hidden_states,
        weight,
        out,
        conv_state,
        None,
        bias,
        conv_state_indices_cpu=conv_state_indices_cpu,
        conv_state_indices_xpu=conv_state_indices,
        num_accepted_tokens_cpu=num_accepted_tokens_cpu,
        num_accepted_tokens_xpu=num_accepted_tokens.to(torch.int32),
        act="SWISH",
        state_seq_stride=conv_state.stride(0),
        is_ncw=False,
    )
    return out.view(-1, hidden_states.shape[-1])


def _pad_spec_hidden_states(
    hidden_states: torch.Tensor,
    max_query_len: int,
    num_accepted_tokens_cpu: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    dim = hidden_states.shape[-1]
    num_spec_decodes = num_accepted_tokens_cpu.shape[0]
    padded_num_tokens = num_spec_decodes * max_query_len

    if hidden_states.shape[0] == padded_num_tokens:
        return hidden_states.view(num_spec_decodes, max_query_len, dim), []

    lengths = [int(length) for length in num_accepted_tokens_cpu.tolist()]
    if sum(lengths) != hidden_states.shape[0]:
        raise ValueError(
            "spec conv token count does not match num_accepted_tokens_cpu: "
            f"got {hidden_states.shape[0]} tokens, expected {sum(lengths)}."
        )

    first_length = lengths[0] if lengths else 0
    if all(length == first_length for length in lengths):
        return hidden_states.view(num_spec_decodes, first_length, dim), []

    padded = hidden_states.new_zeros((num_spec_decodes, max_query_len, dim))
    offset = 0
    for index, length in enumerate(lengths):
        if length > max_query_len:
            raise ValueError(
                f"spec conv length {length} exceeds max_query_len " f"{max_query_len}."
            )
        padded[index, :length].copy_(hidden_states[offset : offset + length])
        offset += length
    return padded, lengths


def _unpad_spec_hidden_states(
    hidden_states: torch.Tensor,
    lengths: list[int],
) -> torch.Tensor:
    if not lengths:
        return hidden_states.view(-1, hidden_states.shape[-1])

    dim = hidden_states.shape[-1]
    total_num_tokens = sum(lengths)
    unpadded = hidden_states.new_empty((total_num_tokens, dim))
    offset = 0
    for index, length in enumerate(lengths):
        unpadded[offset : offset + length].copy_(hidden_states[index, :length])
        offset += length
    return unpadded


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    conv_state_indices_cpu: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    num_accepted_tokens_cpu: Optional[torch.Tensor] = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if False and validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    spec_lengths: list[int] = []
    if num_accepted_tokens is None:
        x = x.squeeze(-1).unsqueeze(1)
    else:
        if max_query_len <= 0:
            max_query_len = seqlen
        if num_accepted_tokens_cpu is None:
            raise ValueError(
                "spec conv requires num_accepted_tokens_cpu to handle "
                "variable scheduled token counts."
            )
        x, spec_lengths = _pad_spec_hidden_states(
            x.squeeze(-1),
            max_query_len,
            num_accepted_tokens_cpu,
        )

    if num_accepted_tokens is None:
        out = torch.empty_like(x)

        stride = conv_state.stride()[0]
        kunlun_ops.causal_conv1d_update(
            x,
            weight,
            out,
            conv_state,
            None,
            bias,
            conv_state_indices_cpu=conv_state_indices_cpu,
            conv_state_indices_xpu=conv_state_indices,
            act="SWISH",
            state_seq_stride=stride,
            is_ncw=False,
        )
        out = out.squeeze(1)
        return out
    else:
        out = causal_conv1d_update_spec_graphsafe(
            x,
            conv_state,
            weight,
            bias,
            conv_state_indices=conv_state_indices,
            conv_state_indices_cpu=conv_state_indices_cpu,
            num_accepted_tokens=num_accepted_tokens,
            num_accepted_tokens_cpu=num_accepted_tokens_cpu,
        )
        return _unpad_spec_hidden_states(
            out.view(x.shape[0], x.shape[1], dim),
            spec_lengths,
        )
