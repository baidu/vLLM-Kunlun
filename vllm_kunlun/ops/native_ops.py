"""
Pure PyTorch native implementations for:
  1. causal_conv1d_update (decode)
  2. causal_conv1d_fn (prefill)
  3. chunk_gated_delta_rule (prefill SSM)
  4. fused_recurrent_gated_delta_rule (decode SSM)

Purpose: bypass ALL XPU kernels to isolate state corruption bug.
"""

import torch
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match XPU l2norm_fwd: normalize along last dim with eps=1e-6."""
    # XPU l2norm uses eps=1e-6, F.normalize default is 1e-12
    return F.normalize(x.float(), p=2, dim=-1, eps=eps).to(x.dtype)


# ============================================================
# 1. causal_conv1d_update — decode 路径 conv1d
# ============================================================
def native_causal_conv1d_update(
    x: torch.Tensor,  # [N, D] (after unsqueeze→[N,1,D])
    conv_state: torch.Tensor,  # [num_slots, W-1, D]
    weight: torch.Tensor,  # [D, W]
    bias: torch.Tensor,  # [D]
    activation: str = "silu",
    conv_state_indices: torch.Tensor = None,  # [N]
    **kwargs,  # absorb extra params
):
    """Single-token conv1d update per sequence."""
    if x.dim() == 3:
        x_2d = x.squeeze(1)  # [N, D]
    else:
        x_2d = x  # [N, D]

    N, D = x_2d.shape
    out = torch.empty_like(x_2d)

    for i in range(N):
        if conv_state_indices is not None:
            si = conv_state_indices[i].item()
        else:
            si = i

        # state: [W-1, D], new token: [D]
        state = conv_state[si]  # [W-1, D]
        token = x_2d[i]  # [D]

        # full window: [W, D]
        full = torch.cat([state, token.unsqueeze(0)], dim=0)  # [W, D]

        # depthwise conv: output[d] = sum_w(full[w,d] * weight[d,w]) + bias[d]
        o = (full.float().T * weight.float()).sum(dim=1)  # [D]
        if bias is not None:
            o = o + bias.float()

        if activation in ("silu", "swish"):
            o = o * torch.sigmoid(o)

        out[i] = o.to(x_2d.dtype)

        # update state: shift left, append new token
        conv_state[si] = torch.cat([state[1:], token.unsqueeze(0)], dim=0)

    return out


# ============================================================
# 2. causal_conv1d_fn — prefill 路径 conv1d
# ============================================================
def native_causal_conv1d_fn(
    x: torch.Tensor,  # [total_tokens, D]
    weight: torch.Tensor,  # [D, W]
    bias: torch.Tensor,  # [D]
    activation: str = "silu",
    conv_states: torch.Tensor = None,  # [num_slots, W-1, D]
    has_initial_state: torch.Tensor = None,  # [N]
    cache_indices: torch.Tensor = None,  # [N]
    query_start_loc: torch.Tensor = None,  # [N+1]
    **kwargs,  # absorb _cpu, metadata, etc.
):
    """Multi-token causal conv1d for prefill."""
    D = x.shape[-1]
    W = weight.shape[1]
    N = len(query_start_loc) - 1
    out = torch.empty_like(x)

    for seq_idx in range(N):
        si = cache_indices[seq_idx].item() if cache_indices is not None else seq_idx
        t_start = query_start_loc[seq_idx].item()
        t_end = query_start_loc[seq_idx + 1].item()

        # initial conv state
        if has_initial_state is not None and has_initial_state[seq_idx]:
            state = conv_states[si].clone()  # [W-1, D]
        else:
            state = torch.zeros(W - 1, D, dtype=x.dtype, device=x.device)

        for t in range(t_start, t_end):
            token = x[t]  # [D]
            full = torch.cat([state, token.unsqueeze(0)], dim=0)  # [W, D]
            o = (full.float().T * weight.float()).sum(dim=1)
            if bias is not None:
                o = o + bias.float()
            if activation in ("silu", "swish"):
                o = o * torch.sigmoid(o)
            out[t] = o.to(x.dtype)
            state = torch.cat([state[1:], token.unsqueeze(0)], dim=0)

        # write back final conv state
        if conv_states is not None:
            conv_states[si] = state

    return out


# ============================================================
# 3. chunk_gated_delta_rule — prefill SSM
# ============================================================
def native_chunk_gated_delta_rule(
    q: torch.Tensor,  # [1, T, H, K]
    k: torch.Tensor,  # [1, T, H, K]
    v: torch.Tensor,  # [1, T, HV, V]
    g: torch.Tensor,  # [1, T, HV]
    beta: torch.Tensor,  # [1, T, HV]
    scale: float = None,
    initial_state: torch.Tensor = None,  # [N, HV, K, V] standard layout
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor = None,  # [N+1]
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
    """Pure torch gated delta rule recurrence for prefill (standard layout)."""
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    GVA = HV // H

    if scale is None:
        scale = K**-0.5

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)

    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
    else:
        N = B

    output = torch.zeros(B, T, HV, V, dtype=v.dtype, device=v.device)
    final_states = []

    for seq_idx in range(N):
        if cu_seqlens is not None:
            t_start = cu_seqlens[seq_idx].item()
            t_end = cu_seqlens[seq_idx + 1].item()
        else:
            t_start, t_end = 0, T

        # initial_state: [N, HV, K, V] standard layout
        state = (
            initial_state[seq_idx].float()
            if initial_state is not None
            else torch.zeros(HV, K, V, dtype=torch.float32, device=v.device)
        )

        for t in range(t_start, t_end):
            k_t = k[0, t].float()  # [H, K]
            q_t = q[0, t].float()  # [H, K]
            v_t = v[0, t].float()  # [HV, V]
            g_t = g[0, t].float()  # [HV]
            b_t = beta[0, t].float()  # [HV]

            # expand key heads → value heads
            k_exp = k_t.repeat_interleave(GVA, dim=0)  # [HV, K]
            q_exp = q_t.repeat_interleave(GVA, dim=0)  # [HV, K]

            decay = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # [HV, 1, 1]

            # outer(k, v): [HV, K, 1] * [HV, 1, V] = [HV, K, V]
            outer = k_exp.unsqueeze(-1) * v_t.unsqueeze(-2)

            # S = decay * S + beta * outer(k, v)
            state = decay * state + b_t.unsqueeze(-1).unsqueeze(-1) * outer

            # o = scale * q^T @ S: [HV, 1, K] @ [HV, K, V] = [HV, 1, V] → [HV, V]
            o_t = torch.bmm(q_exp.unsqueeze(1), state).squeeze(1)
            output[0, t] = (scale * o_t).to(output.dtype)

        if output_final_state:
            final_states.append(state.to(initial_state.dtype))

    final_state = (
        torch.stack(final_states) if output_final_state and final_states else None
    )
    return output, final_state


# ============================================================
# 4. fused_recurrent_gated_delta_rule — decode SSM
# ============================================================
def native_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,  # [1, N, H, K]
    k: torch.Tensor,  # [1, N, H, K]
    v: torch.Tensor,  # [1, N, HV, V]
    g: torch.Tensor,  # [1, N, HV]
    beta: torch.Tensor,  # [1, N, HV]
    scale: float = None,
    initial_state: torch.Tensor = None,  # [num_slots, HV, V, K] TRANSPOSED layout
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor = None,
    ssm_state_indices: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
    """Pure torch gated delta rule for decode (transposed state layout [HV, V, K])."""
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    GVA = HV // H

    if scale is None:
        scale = K**-0.5

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)

    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
    else:
        N = T

    output = torch.zeros(B, T, HV, V, dtype=v.dtype, device=v.device)

    for seq_idx in range(N):
        if ssm_state_indices is not None:
            si = ssm_state_indices[seq_idx].item()
        else:
            si = seq_idx

        if cu_seqlens is not None:
            t_start = cu_seqlens[seq_idx].item()
            t_end = cu_seqlens[seq_idx + 1].item()
        else:
            t_start, t_end = seq_idx, seq_idx + 1

        # state: [HV, V, K] transposed layout
        state = initial_state[si].float()

        for t in range(t_start, t_end):
            k_t = k[0, t].float()
            q_t = q[0, t].float()
            v_t = v[0, t].float()
            g_t = g[0, t].float()
            b_t = beta[0, t].float()

            k_exp = k_t.repeat_interleave(GVA, dim=0)  # [HV, K]
            q_exp = q_t.repeat_interleave(GVA, dim=0)  # [HV, K]

            decay = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # [HV, 1, 1]

            # transposed layout: outer(v, k) = [HV, V, 1] * [HV, 1, K] = [HV, V, K]
            outer = v_t.unsqueeze(-1) * k_exp.unsqueeze(-2)

            state = decay * state + b_t.unsqueeze(-1).unsqueeze(-1) * outer

            # o = scale * S @ q: [HV, V, K] @ [HV, K, 1] = [HV, V]
            o_t = torch.bmm(state, q_exp.unsqueeze(-1)).squeeze(-1)
            output[0, t] = (scale * o_t).to(output.dtype)

        if inplace_final_state:
            initial_state[si] = state.to(initial_state.dtype)

    return output, initial_state if inplace_final_state else None
