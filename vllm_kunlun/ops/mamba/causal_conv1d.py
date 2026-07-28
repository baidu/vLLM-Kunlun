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


def torch_causal_conv1d_update_spec(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
):
    if bias is not None:
        bias = bias.to(weight.dtype)
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

        width = weight.shape[-1]
        # The convolution below needs at least ``width`` input rows to produce a
        # non-negative output length (F.conv1d output len = in_len - width + 1).
        # On the real spec-decode path the concatenated stream is always long
        # enough. During the dummy/warmup run (e.g. cudagraph capture) the
        # constructed ``num_accepted_tokens`` can make ``2 + accepted + seq_len``
        # shorter than ``width``, which would crash F.conv1d with a negative
        # dimension. Left-pad (on the oldest side) with zeros to the minimum
        # required length; this mirrors the community Triton kernel, which reads
        # conv history through a bounded mask and treats out-of-range history as
        # zero. The real path never hits this branch, so its numerics are
        # unchanged.
        min_len = width - 1 + seq_len
        if hidden_states_new.shape[1] < min_len:
            hidden_states_new = F.pad(
                hidden_states_new, (0, 0, min_len - hidden_states_new.shape[1], 0)
            )

        # Roll the conv_state in a sliding-window manner, matching the
        # community Triton kernel (``_causal_conv1d_update_kernel``). That
        # kernel writes ``new_conv_state`` back with a masked store bounded by
        # ``idx_tokens < state_len``, i.e. it fills only the leading valid rows
        # of the cache slot and leaves the rest untouched -- it never requires
        # the written window to exactly equal the physical ``state_len``.
        #
        # Here ``hidden_states_new[:, -state_len:, :]`` is the sliding window
        # (history tail + new tokens). On the real spec-decode path its length
        # is exactly ``state_len``; during the dummy/warmup run (seq_len==1,
        # dummy num_accepted_tokens) it can be shorter. Writing into the front
        # slice ``[:n]`` reproduces the kernel's masked store for both cases
        # without zero-padding the cache (left-padding would corrupt the state
        # with spurious zeros).
        new_conv_state = hidden_states_new[:, -state_len:, :]
        n = new_conv_state.shape[1]
        conv_state[conv_state_indices[i]][:n, :] = new_conv_state[0]
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

    if num_accepted_tokens is None:
        x = x.squeeze(-1).unsqueeze(1)
    else:
        x = x.squeeze(-1).view(-1, max_query_len, dim)
    # if num_accepted_tokens is None:
    # New ``causal_conv1d_update`` writes its output in-place into x.
    # Drop the legacy ``state_seq_stride`` / ``act="SWISH"`` / paired
    # ``*_cpu`` + ``*_xpu`` arguments.
    silu_activation = activation in ("silu", "swish")
    kunlun_ops.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias=bias,
        silu_activation=silu_activation,
        cache_seqlens=None,
        conv_state_indices=conv_state_indices,
        is_ncw=False,
        pad_slot_id=pad_slot_id,
        num_accepted_tokens=num_accepted_tokens,
    )
    # if (
    #     num_accepted_tokens is not None
    # ):
    #     # Binary-search: route the spec conv through the pure-torch reference.
    #     x = torch_causal_conv1d_update_spec(
    #         hidden_states=x,
    #         conv_state=conv_state,
    #         weight=weight,
    #         bias=bias,
    #         activation="silu" if silu_activation else None,
    #         conv_state_indices=conv_state_indices,
    #         num_accepted_tokens=num_accepted_tokens,
    #     )
    # else:
    #     kunlun_ops.causal_conv1d_update(
    #         x,
    #         conv_state,
    #         weight,
    #         bias=bias,
    #         silu_activation=silu_activation,
    #         cache_seqlens=None,
    #         conv_state_indices=conv_state_indices,
    #         is_ncw=False,
    #         pad_slot_id=pad_slot_id,
    #         num_accepted_tokens=num_accepted_tokens,
    #     )
    if num_accepted_tokens is None:
        # non-spec decode: x is [batch, 1, dim] -> [batch, dim]
        return x.squeeze(1)
    # spec (MTP) decode: x is [num_spec_decodes, max_query_len, dim];
    # flatten back to [num_tokens, dim] to match the caller's layout
    # (mixed_qkv_spec[:actual_num] = <return>).
    return x.reshape(-1, dim)
