# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py

from typing import Optional, Union

import kunlun_ops
import torch
import torch.nn.functional as F
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):

    x = x.contiguous()
    out = torch.empty_like(x)
    has_initial_state = has_initial_state.to(torch.int32)
    dim = x.shape[-1]
    cu_seqlen = x.shape[-2]
    width = weight.shape[-1]
    num_cache_lines = conv_states.shape[0]
    state_width = conv_states.shape[-2]
    batch_size = query_start_loc.shape[0] - 1
    stride = conv_states.stride()[0]

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
        query_start_loc.cpu(),
        query_start_loc,
        batch_size,
        bias,
        cache_indices_cpu=cache_indices.cpu(),
        cache_indices_xpu=cache_indices,
        has_initial_state_cpu=has_initial_state.cpu(),
        has_initial_state_xpu=has_initial_state,
        act="SWISH",
        state_seq_stride=stride,
    )
    # out = torch.nn.functional.silu(out)
    return out


def torch_causal_conv1d_update_spec(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
):
    out = torch.empty_like(hidden_states)
    _, seq_len, hidden_size = hidden_states.shape
    for i in range(hidden_states.shape[0]):
        tmp_conv_state = conv_state[conv_state_indices[i]]
        state_len = tmp_conv_state.shape[-2]
        hidden_states_i = hidden_states[i]
        hidden_states_new = torch.cat(
            [tmp_conv_state[: (2 + num_accepted_tokens[i]), :], hidden_states_i], dim=0
        ).to(weight.dtype)

        hidden_states_new = hidden_states_new.unsqueeze(0)

        conv_state[conv_state_indices[i]] = hidden_states_new[:, -state_len:, :]
        for j in range(seq_len):
            if j == seq_len - 1:
                hidden_states_new_j = hidden_states_new
            else:
                hidden_states_new_j = hidden_states_new[:, : (1 - seq_len + j)]
            hidden_states_new_j = hidden_states_new_j.transpose(-1, -2).contiguous()
            out_i = F.conv1d(
                hidden_states_new_j,
                weight.unsqueeze(1),
                bias,
                padding=0,
                groups=hidden_size,
            )
            out_i = F.silu(out_i[:, :, -1:])
            out_i = out_i.to(hidden_states.dtype).squeeze(-1).unsqueeze(0)
            out[i, j] = out_i
    return out.view(-1, hidden_size)


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

    if num_accepted_tokens is None:
        x = x.squeeze(-1).unsqueeze(1)
    else:
        x = x.squeeze(-1).view(-1, max_query_len, dim)
    if num_accepted_tokens is None:
        out = torch.empty_like(x)
        import kunlun_ops

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
        return torch_causal_conv1d_update_spec(
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
        )
