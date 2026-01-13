# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import warnings
from typing import Optional
import torch.nn.functional as F

import torch
import torch.distributed as dist
from einops import rearrange

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril
from .utils import SUPPRESS_LEVEL, input_guard
from .wy_fast import recompute_w_u_fwd
from vllm.distributed import get_tensor_model_parallel_rank


def torch_solve_tril(A: torch.Tensor, cu_seqlens: Optional[torch.LongTensor] = None, output_dtype: torch.dtype = torch.float,):
    chunk_size=64
    A = -A.transpose(1,2)
    sequence_length = A.shape[-2]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    A = F.pad(A, (0, 0, 0, pad_size))
    A = A.reshape(A.shape[0], A.shape[1], -1, chunk_size, A.shape[-1])

    for i in range(1, chunk_size):
        row = A[..., i, :i].clone()
        sub = A[..., :i, :i].clone()
        A[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    A = A + torch.eye(chunk_size, dtype=A.dtype, device=A.device)
    return A.reshape(A.shape[0], A.shape[1], -1, A.shape[-1])[:,:,:sequence_length,:].transpose(1,2)

def recompute_w_u_fwd_torch(
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    beta: torch.Tensor,  # [B, T, H]
    g: torch.Tensor,  # [B, T, H]
    A: torch.Tensor,  # [B, H, T, T]
):
    """
    最简单版本：假设等长序列，key和value头数相同
    """
    chunk_size = 64
    num_v_heads, num_k_heads = v.shape[2], k.shape[2]
    k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)
    k, v, beta, g, A = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (k, v, beta, g, A)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = k.shape
    v_head_dim = v.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    k = F.pad(k, (0, 0, 0, pad_size))
    v = F.pad(v, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    A = F.pad(A, (0, 0, 0, pad_size))
    A = A.reshape(A.shape[0], A.shape[1], -1, chunk_size, A.shape[-1])

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    k, v, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (k, v, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    u = A @ v_beta
    w = A @ (k_beta * g.exp().unsqueeze(-1))
    w = w.reshape(w.shape[0], w.shape[1], -1, w.shape[-1])[:,:,:sequence_length,:].transpose(1,2).contiguous()
    u = u.reshape(u.shape[0], u.shape[1], -1, u.shape[-1])[:,:,:sequence_length,:].transpose(1,2).contiguous()

    return w, u


def split_by_value(tensor, chunk_size=64):
    indices = tensor.tolist()
    result = set(indices)  # 使用集合避免重复

    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i + 1]

        # 计算第一个对齐边界
        # 我们要找的是 start + n*chunk_size，其中n是使结果大于start的最小整数
        first_boundary = start + chunk_size

        # 在(start, end)范围内插入所有对齐边界
        boundary = first_boundary
        while boundary < end:
            result.add(boundary)
            boundary += chunk_size

    return torch.tensor(sorted(result), dtype=tensor.dtype, device=tensor.device)

def chunk_gated_delta_rule_fwd(q: torch.Tensor,
                               k: torch.Tensor,
                               v: torch.Tensor,
                               g: torch.Tensor,
                               beta: torch.Tensor,
                               scale: float,
                               initial_state: torch.Tensor,
                               output_final_state: bool,
                               cu_seqlens: Optional[torch.LongTensor] = None):
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(k=k,
                                 beta=beta,
                                 g_cumsum=g,
                                 cu_seqlens=cu_seqlens,
                                 output_dtype=q.dtype)

    #torch版
    # if get_tensor_model_parallel_rank() == 0:
    #     torch.save(A, "A_in")
    #     torch.save(cu_seqlens, "cu_seqlens")
    tmp_cu_seqlens = split_by_value(cu_seqlens)
    torch.ops.xspeedgate_ops.solve_tril_fwd(A, tmp_cu_seqlens)
    # A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    # for i in range(len(cu_seqlens)-1):
    #     A_i = A[:, cu_seqlens[i]:cu_seqlens[i+1], :, :]
    #     A[:, cu_seqlens[i]:cu_seqlens[i+1], :, :] = torch_solve_tril(A=A_i, cu_seqlens=torch.tensor([0, cu_seqlens[i+1]-cu_seqlens[i]], device=q.device), output_dtype=k.dtype)

    '''
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    for i in range(len(cu_seqlens)-1):
        k_i = k[:, cu_seqlens[i]:cu_seqlens[i+1], :, :]
        v_i = v[:, cu_seqlens[i]:cu_seqlens[i+1], :, :]
        beta_i = beta[:, cu_seqlens[i]:cu_seqlens[i+1], :]
        A_i = A[:, cu_seqlens[i]:cu_seqlens[i+1], :, :]
        g_i = g[:, cu_seqlens[i]:cu_seqlens[i+1], :]

        w_i, u_i = recompute_w_u_fwd_torch(
            k=k_i,
            v=v_i,
            beta=beta_i,
            A=A_i,
            g=g_i,
        )
        w[:, cu_seqlens[i]:cu_seqlens[i+1], :, :] = w_i
        u[:, cu_seqlens[i]:cu_seqlens[i+1], :, :] = u_i
    '''
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                g: torch.Tensor,
                beta: torch.Tensor,
                scale: float,
                initial_state: torch.Tensor,
                output_final_state: bool,
                cu_seqlens: Optional[torch.LongTensor] = None,
                use_qk_l2norm_in_kernel: bool = False):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(q: torch.Tensor,
                           k: torch.Tensor,
                           v: torch.Tensor,
                           g: torch.Tensor,
                           beta: torch.Tensor,
                           scale: float = None,
                           initial_state: torch.Tensor = None,
                           output_final_state: bool = False,
                           cu_seqlens: Optional[torch.LongTensor] = None,
                           head_first: bool = False,
                           use_qk_l2norm_in_kernel: bool = False):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(
        beta.shape
    ) == 3, "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
            stacklevel=2)
        q, k, v, beta, g = map(
            lambda x: rearrange(x, 'b h t ... -> b t h ...'),
            (q, k, v, beta, g))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
            stacklevel=2)
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.")
        if initial_state is not None and initial_state.shape[0] != len(
                cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1]**-0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens,
        use_qk_l2norm_in_kernel)
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o, final_state
