import os
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


# __Q38_CONV_SPEC_KERNEL__: spec(MTP) 路径改走原生内核的开关与 bias fp32 缓存

_Q38_CONV_SPEC_KERNEL = os.environ.get("Q38_CONV_SPEC_KERNEL", "1") == "1"
_Q38_BIAS_F32_CACHE: dict = {}


def torch_causal_conv1d_update_spec_graphsafe(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
):
    """Vectorized, CUDA-graph-safe equivalent of
    ``torch_causal_conv1d_update_spec``.

    The loop version slices ``conv_state[: 2 + num_accepted[i]]`` per request,
    which is data-dependent shape logic and cannot be captured in a CUDA
    graph (and crashes when the capture pads the batch). This rewrite keeps
    the exact same semantics with static shapes: per-token windows and the
    state writeback are expressed as ``gather`` over a request-aligned
    [history | current] buffer, with the accepted-token rollback folded into
    the gather indices. Only static python loops over the spec length L and
    the conv width W remain. Verified bit-equivalent on the state and within
    1-2 bf16 ulps on the output vs the loop implementation (/tmp/conv_spec_vec.py).
    """
    N, L, dim = hidden_states.shape
    W = weight.shape[1]
    # The callers may pass padded persistent buffers (CUDA-graph capture);
    # only the first N entries belong to real spec requests.
    raw_idx = conv_state_indices.long()[:N]
    # Graph replays pad the spec batch; padded rows carry PAD_SLOT_ID (-1).
    # Read from line 0 for them and mask their writeback below.
    valid = raw_idx >= 0
    idx = torch.where(valid, raw_idx, torch.zeros_like(raw_idx))
    # index_select instead of fancy indexing (XPU materializes the whole
    # paged pool on advanced indexing over a non-contiguous view).
    from vllm_kunlun.models.qwen4_exp.nvidia.ple_xpu_state import (
        gather_rows as _q38_gather_rows,
    )
    from vllm_kunlun.models.qwen4_exp.nvidia.ple_xpu_state import (
        scatter_rows_ as _q38_scatter_rows_,
    )

    state = _q38_gather_rows(conv_state, idx)  # (N, state_len, dim)
    state_len = state.shape[1]
    acc = num_accepted_tokens.to(hidden_states.device).long()[:N].view(N, 1)
    keep = (2 + acc).clamp(max=state_len)  # (N,1) valid history rows
    combined = torch.cat(
        [state.to(weight.dtype), hidden_states.to(weight.dtype)], dim=1
    )  # (N, state_len+L, dim)
    arange_w = torch.arange(W, device=hidden_states.device).view(1, W)
    wt = weight.t().view(1, W, dim)
    outs = []
    for j in range(L):  # static loop: graph-safe
        logical = keep + j - (W - 1) + arange_w  # (N, W)
        phys = torch.where(logical < keep, logical, state_len + logical - keep)
        phys = phys.clamp(min=0)
        win = combined.gather(1, phys.unsqueeze(-1).expand(N, W, dim))
        o = (win.float() * wt.float()).sum(1)
        if bias is not None:
            o = o + bias.float()
        outs.append(F.silu(o.to(weight.dtype)))
    out = torch.stack(outs, dim=1).to(hidden_states.dtype)
    total = keep + L
    p = torch.arange(state_len, device=hidden_states.device).view(1, state_len)
    logical = total - state_len + p
    phys = torch.where(logical < keep, logical, state_len + logical - keep)
    phys = phys.clamp(min=0)
    new_state = combined.gather(1, phys.unsqueeze(-1).expand(N, state_len, dim))
    _q38_scatter_rows_(
        conv_state,
        idx,
        torch.where(valid.view(N, 1, 1), new_state.to(conv_state.dtype), state),
    )
    return out.view(-1, dim)


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
        # __Q38_CONV_SPEC_KERNEL__: 原生 spec 内核。
        # /tmp/ut_conv_spec.py + ut_conv_view.py 验证：状态位级一致、输出
        # fp16/bf16 舍入级差异；kv_cache 切片视图与 storage-offset 池均正确；
        # pad 行内核会写 0 号行（FULL 图 padding 即 NULL 页 0，行为兼容）。
        # bias 内核要求 fp32，按权重 id 缓存转换结果。
        # Q38_CONV_SPEC_KERNEL=0 回退 torch graphsafe 兜底。
        if _Q38_CONV_SPEC_KERNEL:
            # 元数据里的 num_accepted_tokens/conv_state_indices 是按 token 数
            # 填充的持久 buffer（FULL 图 staging），内核要求 numel == 请求数，
            # 切到 x 的 batch 维（切片是静态形状视图，图安全）。
            _n = x.shape[0]
            if num_accepted_tokens.shape[0] != _n:
                num_accepted_tokens = num_accepted_tokens[:_n]
            if conv_state_indices is not None and conv_state_indices.shape[0] != _n:
                conv_state_indices = conv_state_indices[:_n]
            # __Q38_CONV_CONTIG_FIX__: spec 路径传入的是
            # spec_state_indices_tensor[:, 0] 列切片（stride=num_spec+1，
            # 非连续）。原生内核按扁平 int32 指针线性读取，bs=1 只读
            # 元素 0 恰好正确；bs>=2 时第 i 行读到 row0 的第 i 个 spec
            # 槽位 -> 跨请求状态串用（并发精度灾难：空 think 塌缩/
            # 重复 token/提前 EOS/乱码）。contiguous 拷贝为 gather
            # kernel，FULL 图捕获后回放读持久 staging buffer，图安全。
            if (
                conv_state_indices is not None
                and not conv_state_indices.is_contiguous()
            ):
                conv_state_indices = conv_state_indices.contiguous()
            _b = bias
            if _b is not None and _b.dtype != torch.float32:
                _bc = _Q38_BIAS_F32_CACHE.get(id(_b))
                if _bc is None:
                    _bc = _b.float()
                    _Q38_BIAS_F32_CACHE[id(_b)] = _bc
                _b = _bc
            _spec_state_len = width - 1 + (max_query_len - 1)
            kunlun_ops.causal_conv1d_update(
                x,
                conv_state[:, :_spec_state_len, :],
                weight,
                bias=_b,
                silu_activation=activation in ("silu", "swish"),
                cache_seqlens=None,
                conv_state_indices=conv_state_indices,
                is_ncw=False,
                pad_slot_id=pad_slot_id,
                num_accepted_tokens=num_accepted_tokens,
            )
            return x.reshape(-1, dim)
        return torch_causal_conv1d_update_spec_graphsafe(
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
        )
