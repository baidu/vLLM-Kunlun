# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import os
from typing import Optional

import kunlun_ops
import torch


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        inplace_final_state: bool = True,
        cu_seqlens: Optional[torch.LongTensor] = None,
        ssm_state_indices: Optional[torch.Tensor] = None,
        num_accepted_tokens: Optional[torch.Tensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        o, ht_output = kunlun_ops.fused_recurrent_gated_delta_rule_fwdv2(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            scale,
            initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            h0_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_h0_transposed=True,
        )
        return o, (initial_state if inplace_final_state else ht_output)


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if (
        num_accepted_tokens is not None
        and os.environ.get("KUNLUN_SPEC_KERNEL_RECURRENT", "0") != "1"
    ):
        # MTP/spec-decode path: the kunlun fwdv2 kernel has a known writeback
        # column-mapping bug (col 0 <- S_{num_accepted-1} instead of S_0; see
        # 单测报告_fwdv2_spec回写列映射bug.md) plus a value-dependent fp16
        # fast-path overflow. Route to the pure-torch fp32 native
        # implementation until the kernel is fixed; set
        # KUNLUN_SPEC_KERNEL_RECURRENT=1 to force the kernel for A/B tests.
        return torch_fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state


@torch.no_grad()
def torch_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch reference of ``fused_recurrent_gated_delta_rule`` for the
    MTP / spec-decode (varlen + continuous-batching) path.

    Faithful re-implementation of the community Triton kernel
    ``fused_recurrent_gated_delta_rule_fwd_kernel`` with the flags used on the
    spec path: IS_VARLEN + IS_CONTINUOUS_BATCHING + IS_SPEC_DECODING +
    INPLACE_FINAL_STATE, IS_KDA=False. Meant as a binary-search substitute for
    the ``kunlun_ops`` fwdv2 kernel to isolate whether the multi-concurrency
    NaN originates in that kernel.

    Shapes (matching the qwen3_next spec call):
      q, k          : (1, T, H,  K)
      v             : (1, T, HV, V)
      g             : (1, T, HV)               per-head decay in log space
      beta          : (1, T, HV) or (1,T,HV,V) scalar / headwise
      initial_state : (num_lines, HV, V, K)    == ssm_state; physically stored
                      with V before K, so it is transposed to [K, V] on read
                      and back to [V, K] on write. Updated in place.
      cu_seqlens    : (N+1,)                   varlen boundaries
      ssm_state_indices : (N, num_spec+1) int  per-position cache-block ids
      num_accepted_tokens : (N,) int           read-slot = num_accepted-1

    State per v-head is kept as S[K, V] during the recurrence. Verified against
    the kunlun fwdv2 kernel (out/state max|Δ|~1e-4 for accepted counts 1..4,
    single- and multi-sequence). The recurrence per token is:
        S     *= exp(g)                       # gated decay
        u      = (v - k @ S) * beta           # delta correction
        S     += outer(k, u)                  # rank-1 update
        o      = q @ S                        # readout

    Rolling-buffer writeback (matches the community Triton kernel):
        column t  <- S_t         for every t, including t == 0
    The next forward reads column ``num_accepted - 1``, so all L candidate
    states must be published. NOTE: the kunlun fwdv2 kernel instead writes
    ``column 0 <- S_{num_accepted-1}``, which is stale by one step and corrupts
    the state (see the comment at the writeback below).

    CUDA-graph-friendly form: this is a fully vectorized rewrite of the earlier
    per-sequence Python loop. It contains no host synchronization (no ``.item()``
    / ``.tolist()`` / ``.cpu()``) and no value-dependent control flow. All
    per-request selection (read slot, candidate blocks, NULL_BLOCK skips) is
    expressed with on-device gather / scatter / masking, and the only Python loop
    runs over the *static* padded speculative length ``L`` (a shape, not a tensor
    value), so the launched work is identical on every graph replay.

    Assumes the padded spec-decode layout, where every request contributes the
    same number of query tokens ``L = ssm_state_indices.shape[-1]`` and therefore
    ``T == N * L``. This holds for the padded MTP verify path that calls it.
    """
    assert cu_seqlens is not None, "torch spec path requires cu_seqlens"
    assert ssm_state_indices is not None, "torch spec path requires ssm_state_indices"
    assert initial_state is not None, "torch spec path requires initial_state"

    out_dtype = v.dtype  # preserve the input dtype for the returned output
    q = q[0].float()
    k = k[0].float()
    v = v[0].float()
    g = g[0].float()
    T, H, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    if scale is None:
        scale = K**-0.5
    if beta is None:
        beta = torch.ones(T, HV, device=v.device)
    else:
        beta = beta[0].float()
    beta_headwise = beta.dim() == 3  # (T, HV, V)

    if use_qk_l2norm_in_kernel:
        q = q / (q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q * scale

    # GVA: v-head hv reads q/k head ``hv // (HV // H)`` (kernel line 64).
    r = HV // H
    head_index = torch.arange(HV, device=q.device) // r
    q_hv_all = q[:, head_index, :]  # (T, HV, K)
    k_hv_all = k[:, head_index, :]  # (T, HV, K)

    is_spec = num_accepted_tokens is not None
    idx2d = ssm_state_indices.dim() == 2
    # N and L come from tensor *shapes* (static across a captured graph), never
    # from tensor values -> no host sync.
    N = cu_seqlens.shape[0] - 1
    L = ssm_state_indices.shape[-1] if idx2d else 1
    assert T == N * L, (
        "CUDA-graph torch path requires the padded spec layout: "
        f"T({T}) must equal N({N}) * L({L})."
    )

    device = q.device
    # Callers may pass padded persistent buffers (CUDA-graph capture / replay);
    # only the first N rows belong to real spec requests.
    idx = ssm_state_indices.long()[:N]
    if not idx2d:
        idx = idx.view(N, 1)  # (N, L=1)

    # ---- Read: parallel gather of every sequence's initial state (before any
    # write), reproducing the kernel's parallel h0 load. Read slot is
    # ``num_accepted - 1`` on the spec path, else column 0. NULL_BLOCK_ID
    # (block id <= 0) -> the sequence contributes nothing (zero state, zero
    # output, no writeback), handled via ``valid_read`` masking. ----
    if is_spec:
        read_col = (num_accepted_tokens.long()[:N] - 1).clamp_(0, L - 1)  # (N,)
    else:
        read_col = torch.zeros(N, dtype=torch.long, device=device)
    read_block = idx.gather(1, read_col.view(N, 1)).squeeze(1)  # (N,)
    valid_read = read_block > 0  # (N,)
    safe_read = read_block.clamp(min=0)
    # ssm_state is physically [.., V, K]; transpose to [K, V] for the math.
    # index_select instead of fancy indexing: the XPU aten wrapper
    # materializes the whole (non-contiguous, ~tens of GB) paged pool for
    # advanced indexing, which OOMs during FULL cudagraph capture.
    from vllm_kunlun.models.qwen4_exp.nvidia.ple_xpu_state import (
        gather_rows as _q38_gather_rows,
    )
    from vllm_kunlun.models.qwen4_exp.nvidia.ple_xpu_state import (
        scatter_rows_ as _q38_scatter_rows_,
    )

    S = (
        _q38_gather_rows(initial_state, safe_read).float().transpose(-2, -1)
    )  # (N, HV, K, V)
    S = torch.where(valid_read.view(N, 1, 1, 1), S, S.new_zeros(()))

    # ---- Recurrence: L (static) steps, all N sequences in parallel. ----
    q2 = q_hv_all.reshape(N, L, HV, K)
    k2 = k_hv_all.reshape(N, L, HV, K)
    v2 = v.reshape(N, L, HV, V)
    g2 = g.reshape(N, L, HV)
    beta2 = beta.reshape(N, L, HV, V) if beta_headwise else beta.reshape(N, L, HV)

    def _commit(state_kv, dest_blocks, wmask):
        """Scatter state (N, HV, K, V) -> physical [V, K] into ``initial_state``
        at ``dest_blocks`` for rows where ``wmask`` & block id > 0. Skipped rows
        are routed to the reserved NULL slot 0 and blended with its current
        value (a no-op), so duplicate NULL indices are harmless."""
        vals = state_kv.transpose(-2, -1).contiguous().to(initial_state.dtype)
        blk = dest_blocks.long()
        m = wmask & (blk > 0)
        safe = torch.where(m, blk, blk.new_zeros(())).clamp_(min=0)
        current = _q38_gather_rows(initial_state, safe)
        _q38_scatter_rows_(
            initial_state, safe, torch.where(m.view(-1, 1, 1, 1), vals, current)
        )

    o = q.new_zeros(N, L, HV, V)
    for t in range(L):
        S = S * torch.exp(g2[:, t])[:, :, None, None]  # (N, HV, K, V)
        k_t = k2[:, t]  # (N, HV, K)
        kv = (S * k_t[:, :, :, None]).sum(2)  # (N, HV, V)
        u = v2[:, t] - kv  # (N, HV, V)
        u = u * (beta2[:, t] if beta_headwise else beta2[:, t][:, :, None])
        S = S + k_t[:, :, :, None] * u[:, :, None, :]  # (N, HV, K, V)
        o[:, t] = (S * q2[:, t][:, :, :, None]).sum(2)  # (N, HV, V)
        if inplace_final_state:
            # Rolling buffer: column t <- state after step t, for EVERY t
            # (including t == 0). The next forward reads column
            # ``num_accepted - 1``, i.e. the state after its last accepted
            # token, so all L candidate states must be published. Writing
            # column 0 with ``S_{num_accepted-1}`` of *this* call instead (what
            # the kunlun fwdv2 kernel does) uses the previous step's acceptance
            # count and leaves column 0 holding a state that is too far ahead
            # whenever this call was entered with ``num_accepted >= 2``; the
            # following step then reads it if it accepts exactly 1 token and the
            # sequence state is corrupted from there on.
            _commit(S, idx[:, t], valid_read)
    # Zero the output of skipped (NULL read) sequences, then flatten back.
    o = (o * valid_read.view(N, 1, 1, 1)).reshape(T, HV, V).to(out_dtype)

    return o.unsqueeze(0), initial_state


# __Q38_FWDV2_XSG__ : spec branch via xspeedgate_ops (native col0-fixed kernel).
# UT (/workspace/ut_xsg_spec_verify.py, 2026-09-07): accepted 1/2 组合与 col0
# 语义均与 torch 参考逐位一致（有效行 out rel<=2.4e-5，状态池 fp32 1ulp）；
# kernel 不跳过 NULL(blockid=0) 填充行，但只污染保留槽 block 0 且 padding 输出
# 下游丢弃，生产无害（vLLM BlockPool 的 block 0 永不分配给真实请求）。
# 回退：Q38_FWDV2_XSG=0 恢复 torch fp32 兜底路径。
_q38_xsg_prev = fused_recurrent_gated_delta_rule
_Q38_FWDV2_XSG = os.environ.get("Q38_FWDV2_XSG", "1") == "1"


def _q38_xsg_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    h0_source,
    inplace_final_state,
    cu_seqlens,
    h0_indices,
    num_accepted_tokens,
    use_qk_l2norm_in_kernel,
    is_h0_transposed=True,
):
    # Mirrors kunlun_ops._xflashinfer_fusion.fused_recurrent_gated_delta_rule_fwdv2
    # but calls the col0-fixed xspeedgate_ops kernel instead.
    import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)

    B, T, _H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    o = torch.zeros_like(v)
    if inplace_final_state and h0_source is not None:
        ht_output = h0_source
    else:
        if cu_seqlens is None:
            ht_output = q.new_empty(B * T, HV, K, V, dtype=torch.float32)
        else:
            ht_output = q.new_empty(T, HV, K, V, dtype=torch.float32)
    is_beta_headwise = beta.ndim == v.ndim
    torch.ops.xspeedgate_ops.fused_recurrent_gated_delta_rule_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        o,
        ht_output,
        is_beta_headwise,
        use_qk_l2norm_in_kernel,
        inplace_final_state,
        is_h0_transposed,
        scale,
        cu_seqlens,
        num_accepted_tokens,
        h0_source,
        h0_indices,
    )
    return o, ht_output


def fused_recurrent_gated_delta_rule(  # noqa: F811
    q,
    k,
    v,
    g,
    beta=None,
    scale=None,
    initial_state=None,
    inplace_final_state=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
):
    # Q38 FP16: fwdv2/xsg kernel 在 q=fp16 时要求 g/beta 也为 fp16
    # （host 检查 not support q/ht_output/g dtype）。bf16 部署不受影响。
    if q.dtype == torch.float16:
        if g.dtype != torch.float16:
            g = g.half()
        if beta is not None and beta.dtype != torch.float16:
            beta = beta.half()
    if (
        _Q38_FWDV2_XSG
        and num_accepted_tokens is not None
        and cu_seqlens is not None
        and ssm_state_indices is not None
    ):
        if scale is None:
            scale = k.shape[-1] ** -0.5
        if beta is None:
            beta = torch.ones_like(q[..., 0])
        return _q38_xsg_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            inplace_final_state,
            cu_seqlens,
            ssm_state_indices,
            num_accepted_tokens,
            use_qk_l2norm_in_kernel,
            is_h0_transposed=True,
        )
    return _q38_xsg_prev(
        q,
        k,
        v,
        g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
