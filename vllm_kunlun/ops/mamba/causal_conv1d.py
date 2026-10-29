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
        )
        return x.squeeze(1)
    else:
        # kunlun_ops 原地写输入，保留调用者的张量。
        x = x.clone(memory_format=torch.contiguous_format)
        if x.size(0) == 0:
            return x.reshape(-1, dim)
        # NULL block 和 padding 都映射到 slot 0。
        indices = conv_state_indices.clamp_min(0)
        status = kunlun_ops.causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias=None if bias is None else bias.float(),
            silu_activation=False,
            cache_seqlens=None,
            conv_state_indices=indices,
            intermediate_conv_window=None,
            is_ncw=False,
            pad_slot_id=0,
            num_accepted_tokens=num_accepted_tokens,
        )
        if status != 0:
            raise RuntimeError(f"kunlun causal_conv1d_update failed: {status}")
        # 卷积先按输出 dtype 舍入，再做 SiLU。
        if activation in ("silu", "swish"):
            x = F.silu(x)
        x.masked_fill_((conv_state_indices <= 0).view(-1, 1, 1), 0)
        return x.reshape(-1, dim)
