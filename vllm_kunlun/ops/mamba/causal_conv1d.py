import kunlun_ops
import torch
import torch.nn.functional as F
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    query_start_loc_cpu: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    cache_indices_cpu: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    has_initial_state_cpu: torch.Tensor | None = None,
    conv_states: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    if not x.is_contiguous():
        x = x.contiguous()
    # kunlun_ops accepts raw host/device pointers for the metadata arrays and
    # currently assumes unit stride. In MTP align mode, selecting the first
    # column of a multi-column block table produces a non-contiguous view such
    # as logical [45, 61] backed by physical [45, 46, ..., 61, ...].
    if cache_indices is not None and not cache_indices.is_contiguous():
        cache_indices = cache_indices.contiguous()
    if cache_indices_cpu is not None and not cache_indices_cpu.is_contiguous():
        cache_indices_cpu = cache_indices_cpu.contiguous()

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
    """CUDA/XPU-graph-safe pure-torch reference for the spec (MTP) conv update.

    hidden_states: (batch, seq_len, dim)
    conv_state:    (num_cache_lines, state_len, dim)  [is_ncw=False layout]
    weight:        (dim, width), bias: (dim,)

    ``num_accepted_tokens`` is only ever used to build *gather indices* and a
    *mask* -- never a Python slice bound, an ``if`` condition or a tensor
    shape. Every shape below is a function of (batch, seq_len, dim, width)
    only, so the traced graph stays valid for any accepted-token count and
    there is no device->host sync.
    """
    batch, seq_len, dim = hidden_states.shape
    state_len = conv_state.shape[-2]
    width = weight.shape[-1]
    # The caller guarantees a sliding window that exactly fits the cache slot
    # (state_len == conv_kernel_size - 1 + num_spec, seq_len == num_spec + 1).
    assert state_len == width - 2 + seq_len, (
        f"spec conv expects state_len == width - 2 + seq_len, got "
        f"state_len={state_len}, width={width}, seq_len={seq_len}"
    )

    idx = conv_state_indices.long()
    # Read the selected cache lines *before* the write-back below.
    hist = conv_state.index_select(0, idx)  # (batch, state_len, dim)

    # Request i has ``2 + accepted_i`` valid history rows, so the ``width - 1``
    # rows the convolution needs start at ``2 + accepted_i - (width - 1)``.
    # Negative rows lie outside the history -- only reachable on the
    # dummy/warmup run where accepted can be 0. They are clamped for the gather
    # and then zeroed by ``row_ok``, which reproduces the zero-padded history of
    # the community Triton kernel (it reads history through a bounded mask).
    rows = num_accepted_tokens.long().view(batch, 1) + (3 - width)
    rows = rows + torch.arange(width - 1, device=hidden_states.device).view(1, -1)
    row_ok = (rows >= 0) & (rows < state_len)
    rows = rows.clamp_(0, state_len - 1)
    hist_tail = hist.gather(1, rows.unsqueeze(-1).expand(batch, width - 1, dim))
    hist_tail = hist_tail * row_ok.unsqueeze(-1).to(hist_tail.dtype)

    # (batch, width - 1 + seq_len, dim): history tail followed by the new
    # tokens. Output token j is the conv over the fixed window [j, j + width).
    stream = torch.cat([hist_tail, hidden_states], dim=1).to(weight.dtype)

    out = F.conv1d(
        stream.transpose(1, 2).contiguous(),
        weight.unsqueeze(1),
        bias.to(weight.dtype) if bias is not None else None,
        padding=0,
        groups=dim,
    )  # (batch, dim, seq_len)
    # Unconditional silu, matching the previous reference and the GDN caller,
    # which always requests silu.
    out = F.silu(out).transpose(1, 2).to(hidden_states.dtype)

    # Slide the cache window: drop the oldest row of the stream, keeping the
    # newest ``state_len`` rows.
    conv_state.index_copy_(0, idx, stream[:, 1:, :].to(conv_state.dtype).contiguous())
    return out.reshape(-1, dim)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    cache_seqlens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    conv_state_indices_cpu: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    num_accepted_tokens_cpu: torch.Tensor | None = None,
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
    if num_accepted_tokens is not None:
        state_len = width - 1 + (max_query_len - 1)  # spec
    else:
        state_len = width - 1
    kunlun_ops.causal_conv1d_update(
        x,
        conv_state[:, :state_len, :],
        weight,
        bias=bias,
        silu_activation=silu_activation,
        cache_seqlens=None,
        conv_state_indices=conv_state_indices,
        is_ncw=False,
        pad_slot_id=pad_slot_id,
        num_accepted_tokens=num_accepted_tokens,
    )
    if num_accepted_tokens is None:
        # non-spec decode: x is [batch, 1, dim] -> [batch, dim]
        return x.squeeze(1)
    # spec (MTP) decode: x is [num_spec_decodes, max_query_len, dim];
    # flatten back to [num_tokens, dim] to match the caller's layout
    # (mixed_qkv_spec[:actual_num] = <return>).
    return x.reshape(-1, dim)
