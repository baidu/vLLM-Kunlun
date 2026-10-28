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

import torch
from einops import rearrange

# Q38 XPU PORT: 算子统一由 xspeedgate whl 提供——官方 whl 里除了 xspeedgate_ops 还打进
# 一个二进制 cocopod 包，并在 xspeedgate_ops/__init__.py 里 import 它，把那批算子注册进
# 同一个 xspeedgate_ops 命名空间。所以这里只 import xspeedgate_ops 就够了；直接再 import
# 一份会 def() 同名算子，撞上 torch 的 "Duplicate registration" 并**直接 abort**
# （c10::Error 走 terminate，没有 python traceback，表现为进程静默退出）。
try:
    import xspeedgate_ops  # noqa: F401
except ImportError:  # 旧镜像里只有 cocopod
    import cocopod  # noqa: F401

from .index import prepare_chunk_indices, prepare_chunk_offsets
from .l2norm import l2norm_fwd
from .utils import SUPPRESS_LEVEL, input_guard


def _chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum, cu_seqlens, chunk_indices, chunk_size):
    """挑一个能跑当前入参的 chunk_scaled_dot_kkt_fwd 实现。

    Q38 XPU PORT: 随 whl 分发的 cocopod 版只支持 fp16、且必须给 cu_seqlens；XSpeedGate 里
    补的 chunk_scaled_dot_kkt_fwd_v2 支持 fp16/bf16/fp32 且 cu_seqlens 可为空（名字带 _v2
    是因为同名 schema 在两个 .so 里注册会让 import 阶段 abort）。为了不改既有 fp16 varlen
    调用方的行为，只在 cocopod 版覆盖不到的入参上才走 v2。
    """
    ops = torch.ops.xspeedgate_ops
    if k.dtype is torch.float16 and cu_seqlens is not None:
        return ops.chunk_scaled_dot_kkt_fwd(
            k, beta, g_cumsum, cu_seqlens, chunk_indices, chunk_size
        )
    try:
        return ops.chunk_scaled_dot_kkt_fwd_v2(
            k, beta, g_cumsum, cu_seqlens, chunk_indices, chunk_size
        )
    except (AttributeError, RuntimeError) as exc:
        raise NotImplementedError(
            "cocopod 版 chunk_scaled_dot_kkt_fwd 只支持 fp16 + varlen，当前 "
            f"dtype={k.dtype}、cu_seqlens={'None' if cu_seqlens is None else 'given'}；"
            "需要算子包提供 chunk_scaled_dot_kkt_fwd_v2"
        ) from exc


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    chunk_size = 64
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, 64) if cu_seqlens is not None else None
    )
    chunk_offsets = (
        prepare_chunk_offsets(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )

    g = torch.ops.xspeedgate_ops.chunk_local_cumsum(
        g,
        chunk_size=64,
        reverse=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        head_first=False,
    )

    A = _chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens, chunk_indices, chunk_size)

    torch.ops.xspeedgate_ops.solve_tril_ns(A, cu_seqlens, chunk_indices, chunk_size)

    """
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
    """
    w, u = torch.ops.xspeedgate_ops.recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=64,
    )
    """
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    """
    h, v_new, final_state = torch.ops.xspeedgate_ops.chunk_gated_delta_rule_fwd_h(
        k,
        u,
        w,
        g,
        initial_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets.to(torch.int32),
        chunk_size,
        output_final_state,
        True,
    )

    o = torch.ops.xspeedgate_ops.chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=64,
    )
    """
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    """
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
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


# __Q38_NOSYNC__: 融合 chunk kernel 要主机侧 cu_seqlens。原来每层现场
# `cu_seqlens.cpu()`，在含 prefill 的 step 上队列很深，一次 2~5 个 int 的 D2H
# 要等 ~800us 的 cudaStreamSynchronize（36 层 = mixed step 同步时间的一半）。
# GDNAttentionMetadata.non_spec_query_start_loc_cpu 已经是同一份主机镜像
# （同样 int32、同样 cumsum），直接透传。Q38_CUSEQ_CPU=0 回到旧行为。
import os as _q38_os_ck

_Q38_CUSEQ_CPU = _q38_os_ck.environ.get("Q38_CUSEQ_CPU", "1") == "1"


def _q38_cu_seqlens_host(cu_seqlens, cu_seqlens_cpu):
    if cu_seqlens is None:
        return None
    if (
        _Q38_CUSEQ_CPU
        and cu_seqlens_cpu is not None
        and cu_seqlens_cpu.device.type == "cpu"
        and cu_seqlens_cpu.shape == cu_seqlens.shape
        and cu_seqlens_cpu.dtype == cu_seqlens.dtype
    ):
        return cu_seqlens_cpu
    return cu_seqlens.cpu()


def _q38_fused_cgdr_enabled() -> bool:
    """Q38 XPU PORT: 是否用 kunlun_ops 融合 chunk_gated_delta_rule 替代 fla 路径。

    原因见下面 chunk_gated_delta_rule 里的注释：fla 路径的
    chunk_scaled_dot_kkt_fwd 在本镜像只支持 float16。
    """
    import os

    return os.environ.get("Q38_FUSED_CGDR", "1") == "1"


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,  # __Q38_NOSYNC__
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
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
    assert (
        q.dtype != torch.float32
    ), "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert (
        len(beta.shape) == 3
    ), "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
            stacklevel=2,
        )
        q, k, v, beta, g = map(
            lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g)
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
            stacklevel=2,
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if _q38_fused_cgdr_enabled():
        # Q38 XPU PORT: fla 版 chunk 路径依赖 xspeedgate_ops.chunk_scaled_dot_kkt_fwd，
        # 该算子在本镜像里只注册了 float16（bfloat16 直接报
        # chunk_scaled_dot_kkt_fwd not support dtype），而 Qwen3.8-Flash-Next 按
        # bfloat16 部署。改走 kunlun_ops 的融合 chunk_gated_delta_rule：实测支持
        # q/k/v=bfloat16 + g/beta/initial_state=float32，这里统一入参 dtype，
        # 出参 final_state 再转回调用方的 dtype。Q38_FUSED_CGDR=0 可关掉回到 fla 路径。
        # Q38 FP16: kernel 要求 dtype 全套一致——q=bf16 时 g/beta/state 用 fp32，
        # q=fp16 时全 fp16（混搭一律报 unsupported dtype combination）。
        _aux_dtype = torch.float16 if q.dtype == torch.float16 else torch.float32
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.to(_aux_dtype).contiguous()
        beta = beta.to(_aux_dtype).contiguous()
        if initial_state is None:
            num_seqs = 1 if cu_seqlens is None else len(cu_seqlens) - 1
            state_dtype = _aux_dtype
            initial_state = torch.zeros(
                (num_seqs, v.shape[2], v.shape[3], k.shape[3]),
                dtype=_aux_dtype,
                device=v.device,
            )
        else:
            state_dtype = initial_state.dtype
            initial_state = initial_state.to(_aux_dtype).contiguous()

        o = torch.empty_like(v)
        final_state = torch.empty_like(initial_state)
        import kunlun_ops

        kunlun_ops.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            o,
            final_state,
            _q38_cu_seqlens_host(cu_seqlens, cu_seqlens_cpu),  # __Q38_NOSYNC__
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if final_state.dtype != state_dtype:
            final_state = final_state.to(state_dtype)
    else:
        o, final_state = ChunkGatedDeltaRuleFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            use_qk_l2norm_in_kernel,
        )
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, final_state
