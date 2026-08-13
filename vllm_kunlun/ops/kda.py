"""Torch replacements for the triton-only KDA kernels on Kunlun XPU.

Triton cannot load its binaries on P800 (``Triton Error [CUDA]:
CUDA_ERROR_NOT_SUPPORTED`` from ``load_binary``), and every KDA kernel
(``causal_conv1d_*``, ``chunk_kda_*``, ``fused_recurrent_kda*``,
``gather_initial_states``, ``rms_norm_gated``) is triton-only.

The kunlun gated-delta-rule kernels are not a usable substitute:
``fused_recurrent_gated_delta_rule_fwd`` and ``...fwdv2`` both reject a
per-channel gate (``RuntimeError: g size must equal to B * T * HV``), i.e. they
only support one scalar decay per head, while KDA decays the recurrent state per
channel (``g`` is ``[B, T, H, head_dim]``).

Each function below replaces exactly one kernel entry point and keeps its
signature, so ``KimiK3DeltaAttention._forward`` and its prefill/decode split,
cache bookkeeping and spec-decode handling all run unchanged. Where a native XPU
kernel exists the replacement forwards to it (``kunlun_ops.l2norm``,
``kunlun_ops.chunk_gla_fwd_o_gk``, ``xspeedgate_ops.layer_norm_gated_fwd``,
``xspeedgate_ops.fused_recurrent_kda_packed_decode``) instead of using torch.

Recurrence ported from
``vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py``:

    gate  = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))   if lower_bound
            -exp(A_log) * softplus(raw_g + dt_bias)                 otherwise
    q, k  = l2norm(q), l2norm(k);  q *= head_dim ** -0.5
    S     = S * exp(gate)                     # decay along the K axis
    v     = (v - S @ k) * sigmoid(raw_beta)
    S     = S + v (x) k
    out   = S @ q
"""

import kunlun_ops
import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_SOFTPLUS_THRESHOLD = 20.0

FLA_CHUNK_SIZE = 64
_SUB_CHUNK_SIZE = 16
RCP_LN2 = 1.4426950216


def kda_gate(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    """Per-channel decay gate in negative log space, shape ``[B, T, H, D]``."""
    num_heads, head_dim = raw_g.shape[-2:]
    g = raw_g.float()
    if dt_bias is not None:
        g = g + dt_bias.float().view(1, 1, num_heads, head_dim)
    a = A_log.float().exp().view(1, 1, num_heads, 1)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a * g)
    softplus = torch.where(g > _SOFTPLUS_THRESHOLD, g, torch.log1p(g.exp()))
    return -a * softplus


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise ``[B, T, H, D]`` along the last dim, same as upstream.

    ``kunlun_ops.l2norm`` writes into a caller-allocated ``out`` and requires it
    to carry ``x``'s dtype, so the output follows the input. ``eps`` defaults to
    upstream's 1e-6, not the kernel's 1e-5.
    """
    x = x.contiguous()
    out = torch.empty_like(x)
    kunlun_ops.l2norm(x, out, eps)
    return out


def _delta_rule_scan(
    qf: torch.Tensor,
    kf: torch.Tensor,
    vf: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    begin: int,
    end: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run the gated delta rule over ``[begin, end)``, batched over heads.

    ``qf``/``kf``/``vf``/``decay`` are ``[T, H, D]`` fp32, ``beta`` is ``[T, H]``
    fp32 (already sigmoid-ed), ``state`` is ``[H, V, K]`` fp32. Writes ``out[t]``
    and returns the updated state.
    """
    for t in range(begin, end):
        state = state * decay[t].unsqueeze(-2)
        kt = kf[t]
        delta = (vf[t] - (state @ kt.unsqueeze(-1)).squeeze(-1)) * beta[t].unsqueeze(-1)
        state = state + delta.unsqueeze(-1) * kt.unsqueeze(-2)
        out[t] = (state @ qf[t].unsqueeze(-1)).squeeze(-1)
    return state


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    **kwargs,
) -> torch.Tensor:
    """Varlen causal depthwise conv, ``x`` is ``[dim, num_tokens]``.

    ``conv_states`` (``[num_slots, dim, state_len]``) is updated in place with
    the trailing ``state_len`` inputs of every sequence.
    """
    assert activation in ("silu", "swish", None)
    dim = x.shape[0]
    state_len = conv_states.shape[-1]
    starts = query_start_loc.tolist()
    num_seqs = len(starts) - 1
    slots = (
        list(range(num_seqs)) if cache_indices is None else cache_indices.tolist()
    )
    init_flags = (
        [False] * num_seqs if has_initial_state is None else has_initial_state.tolist()
    )

    out = torch.empty_like(x)
    w = weight.float().unsqueeze(1)
    b = None if bias is None else bias.float()
    for i in range(num_seqs):
        begin, end = starts[i], starts[i + 1]
        slot = slots[i]
        if begin == end or slot < 0:
            continue
        seq = x[:, begin:end].float()
        if init_flags[i]:
            seq = torch.cat([conv_states[slot].float(), seq], dim=-1)
        else:
            seq = F.pad(seq, (state_len, 0))
        y = F.conv1d(seq.unsqueeze(0), w, b, groups=dim)[0]
        if activation is not None:
            y = F.silu(y)
        out[:, begin:end] = y.to(out.dtype)
        conv_states[slot].copy_(seq[:, -state_len:])
    return out


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    out: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Single-token causal conv, ``x`` is ``[num_tokens, dim]``.

    ``conv_state`` is ``[num_slots, dim, state_len]`` and is updated in place.
    """
    if num_accepted_tokens is not None:
        raise NotImplementedError(
            "KDA speculative decode conv update is not supported on Kunlun XPU"
        )
    if x.dim() != 2:
        raise NotImplementedError(f"expected x of shape [tokens, dim], got {x.shape}")
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    # Vectorised on purpose: `.tolist()` plus python-int row indexing bakes the
    # capture-time slots into a cuda graph, so every replay would read and write
    # the wrong conv-state rows (and since the capture dummy run passes all-`-1`
    # indices the loop body would not be recorded at all, leaving `y`
    # uninitialised). `index_select`/`index_copy_` keep it a gather/scatter over
    # a device tensor, which replay re-executes against the refreshed indices.
    # Padded lanes carry -1; send them to slot 0, vLLM's reserved null block
    # (block_pool.py reserves the first block, so no request owns slot 0).
    slots = conv_state_indices[: x.shape[0]].to(torch.long).clamp_min(0)
    w = weight.float()
    b = None if bias is None else bias.float()
    state = conv_state.index_select(0, slots).float()  # [tokens, dim, state_len]
    window = torch.cat([state, x.unsqueeze(-1).float()], dim=-1)  # [tokens, dim, width]
    y = (window * w).sum(-1)
    if b is not None:
        y = y + b
    conv_state.index_copy_(0, slots, window[:, :, 1:].to(conv_state.dtype))
    if activation is not None:
        y = F.silu(y)

    y = y.to(x.dtype)
    if out is not None:
        out.copy_(y)
        return out
    return y


def gather_initial_states(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Gather dense state rows, zeroing rows without an initial state.

    Per-slot basic indexing: ``state.index_select`` copies the whole paged cache
    on XPU (see fused_recurrent_kda_packed_decode).
    """
    slots = indices.tolist()
    flags = has_initial_state.tolist()
    out = state.new_zeros((len(slots), *state.shape[1:]))
    for i, slot in enumerate(slots):
        if flags[i] and slot >= 0:
            out[i] = state[slot]
    return out


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """``[NT, 2]`` table of ``(sequence index, chunk index inside the sequence)``.

    Same layout and order as upstream's triton-side helper, so a kernel taking
    ``chunk_indices`` can be dropped in unchanged. The sequence column is built
    explicitly instead of upstream's ``indices.eq(0).cumsum(0) - 1``: that form
    skips a zero-length sequence and shifts every later sequence index by one.
    """
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = ((lens + chunk_size - 1) // chunk_size).tolist()
    seq = torch.cat([torch.full((n,), i) for i, n in enumerate(num_chunks)])
    indices = torch.cat([torch.arange(n) for n in num_chunks])
    return torch.stack([seq, indices], 1).to(cu_seqlens)


def _chunk_token_span(
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``([NT] first token of every chunk, [NT] end of its sequence)``."""
    seq = chunk_indices[:, 0].long()
    starts = cu_seqlens[seq].long() + chunk_indices[:, 1].long() * chunk_size
    return starts, cu_seqlens[seq + 1].long()


def _chunk_tiles(
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token index and validity of every chunk row, both ``[NT, BT]``.

    The rows of a chunk that fall past the end of its sequence (only the last
    chunk can have any) are marked invalid and index-clamped, which keeps every
    stage on static shapes instead of ragged per-sequence slices.
    """
    starts, seq_ends = _chunk_token_span(cu_seqlens, chunk_indices, chunk_size)
    pos = starts.unsqueeze(1) + torch.arange(chunk_size, device=cu_seqlens.device)
    valid = pos < seq_ends.unsqueeze(1)
    return pos.clamp_max(num_tokens - 1), valid


def _tile(x: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``[1, T, H, D]`` -> fp32 ``[NT, H, BT, D]`` tiles, invalid rows zeroed."""
    tiles = x[0][pos].float().masked_fill(~valid[:, :, None, None], 0.0)
    return tiles.transpose(1, 2).contiguous()


def _untile(
    tiles: torch.Tensor,
    pos: torch.Tensor,
    valid: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Scatter ``[NT, H, BT, D]`` tiles back into the ``[1, T, H, D]`` ``out``."""
    out[0][pos[valid]] = tiles.transpose(1, 2)[valid].to(out.dtype)
    return out


def _chunk_last_gate(g: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Cumulative gate of each chunk's last valid token, ``[NT, H, 1, D]``."""
    last = valid.sum(1) - 1
    rows = torch.arange(g.shape[0], device=g.device)
    return g[rows, :, last].unsqueeze(-2)


def _inv_unit_lower_panel(L: torch.Tensor) -> torch.Tensor:
    """``(I + L)^-1`` by exact row substitution, ``L`` strictly lower ``[..., n, n]``.

    Row ``t`` of the inverse only needs the rows above it, so one pass down the
    panel is exact; ``n`` is ``_SUB_CHUNK_SIZE``, small enough for the loop to
    stay cheap.
    """
    n = L.shape[-1]
    inv = torch.eye(n, dtype=L.dtype, device=L.device).expand_as(L).contiguous()
    for t in range(1, n):
        inv[..., t : t + 1, :] -= L[..., t : t + 1, :t] @ inv[..., :t, :]
    return inv


def _inv_unit_lower(L: torch.Tensor) -> torch.Tensor:
    """``(I + L)^-1`` for strictly lower-triangular ``L``, ``[..., BT, BT]``.

    Blocked forward substitution: each ``_SUB_CHUNK_SIZE`` diagonal panel is
    inverted exactly, then the strictly lower blocks follow from
    ``A_ij = -A_ii @ sum_{j <= k < i} L_ik @ A_kj``.

    ``L`` is nilpotent, so ``sum_j (-L)^j`` would also be exact and needs only
    ``log2(BT)`` matmuls, but it is unusable in fp32: with correlated keys under
    a weak gate the intermediate powers of ``L`` reach 1e8 while the inverse
    itself stays O(1), and the cancellation leaves no correct digit -- which
    showed up as pure garbage tokens for any prompt past one chunk.
    """
    bt, bc = L.shape[-1], _SUB_CHUNK_SIZE
    nb = bt // bc
    blk = [
        [L[..., i * bc : (i + 1) * bc, j * bc : (j + 1) * bc] for j in range(nb)]
        for i in range(nb)
    ]
    diag = _inv_unit_lower_panel(torch.stack([blk[i][i] for i in range(nb)]))
    inv = [[torch.zeros_like(diag[0])] * nb for _ in range(nb)]
    for i in range(nb):
        inv[i][i] = diag[i]
        for j in range(i):
            acc = blk[i][j] @ inv[j][j]
            for k in range(j + 1, i):
                acc = acc + blk[i][k] @ inv[k][j]
            inv[i][j] = -(diag[i] @ acc)
    return torch.cat([torch.cat(row, -1) for row in inv], -2)


def fused_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype | None = torch.float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate stage, same contract as the upstream fused kernel.

    Returns ``g`` (``[1, T, H, D]``: the *chunk-local* cumulative sum of the
    per-token gate, scaled by ``RCP_LN2`` so consumers rebuild ``exp(gate)`` with
    ``exp2``) and ``beta`` (``[1, T, H]`` fp32 ``sigmoid(raw_beta)``).

    ``xspeedgate_ops.fused_kda_gate_chunk_cumsum`` implements this stage with the
    same contract; ``beta``/``threshold`` are the softplus parameters of the
    ``lower_bound is None`` branch, which upstream leaves at their defaults.
    """
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    return torch.ops.xspeedgate_ops.fused_kda_gate_chunk_cumsum(
        raw_g.contiguous(),
        raw_beta,
        A_log.float().contiguous(),
        None if g_bias is None else g_bias.float().contiguous(),
        1.0,
        _SOFTPLUS_THRESHOLD,
        lower_bound,
        cu_seqlens.to(torch.int32).contiguous(),
        chunk_indices.to(torch.int32).contiguous(),
        chunk_size,
        output_dtype or raw_g.dtype,
    )


def _fused_kda_gate_chunk_cumsum_torch(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype | None = torch.float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for the gate stage, kept for numerical comparison."""
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    gate = kda_gate(raw_g, A_log, g_bias, lower_bound)  # [1, T, H, D] fp32
    pos, valid = _chunk_tiles(cu_seqlens, chunk_indices, chunk_size, raw_g.shape[1])
    # Tiling is what makes the cumulative sum chunk-local: it cannot leak across
    # a chunk (nor a sequence) boundary.
    tiles = _tile(gate, pos, valid).cumsum(-2) * RCP_LN2
    g = torch.zeros_like(gate, dtype=output_dtype or raw_g.dtype)
    _untile(tiles, pos, valid, g)
    return g, torch.sigmoid(raw_beta.float())


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    chunk_indices: torch.Tensor | None = None,
    safe_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intra-chunk attention matrices ``Aqk`` and ``A``, both ``[1, T, H, BT]``.

    With ``d`` the gate channel and ``s``, ``t`` two tokens of one chunk::

        Aqk[t, s] = scale * sum_d q[t, d] * k[s, d] * exp2(g[t, d] - g[s, d])
        Akk[t, s] = beta[t] * sum_d k[t, d] * k[s, d] * exp2(g[t, d] - g[s, d])
        A         = (I + tril(Akk, -1))^-1

    ``A`` is the WY transform that turns the delta rule's sequential correction
    into one triangular solve, and row ``t`` of both matrices holds the ``BT``
    columns of ``t``'s own chunk -- upstream's layout exactly. ``safe_gate`` is
    accepted for signature parity: the gate difference is always evaluated
    exactly (and clamped at 0 before ``exp2``) inside a sub-block, so no
    intermediate can overflow, which is what upstream's safe path buys.
    """
    num_tokens, num_heads = q.shape[1], q.shape[2]
    bt, bc = chunk_size, _SUB_CHUNK_SIZE
    pos, valid = _chunk_tiles(cu_seqlens, chunk_indices, bt, num_tokens)
    q_t = _tile(q, pos, valid) * scale
    k_t = _tile(k, pos, valid)
    g_t = _tile(gk, pos, valid)
    b_t = _tile(beta.unsqueeze(-1), pos, valid)

    Aqk_t = q_t.new_zeros(pos.shape[0], num_heads, bt, bt)
    Akk_t = torch.zeros_like(Aqk_t)
    for i_c in range(bt // bc):
        rows = slice(i_c * bc, (i_c + 1) * bc)
        q_i, k_i, g_i = q_t[:, :, rows], k_t[:, :, rows], g_t[:, :, rows]
        # Diagonal sub-block: subtract the gates before exponentiating. The upper
        # triangle is masked off below, so clamping it at 0 only keeps a long
        # decay run from overflowing on its way to being discarded.
        decay = torch.exp2((g_i.unsqueeze(-2) - g_i.unsqueeze(-3)).clamp_max(0.0))
        Aqk_t[:, :, rows, rows] = (
            q_i.unsqueeze(-2) * decay * k_i.unsqueeze(-3)
        ).sum(-1)
        Akk_t[:, :, rows, rows] = (
            k_i.unsqueeze(-2) * decay * k_i.unsqueeze(-3)
        ).sum(-1)
        if i_c:
            # Earlier sub-blocks: factor the decay through this sub-block's first
            # gate, which turns the pairwise term into a matmul. For a valid row
            # both exponents are already <= 0 (the cumulative gate never
            # increases); the clamp only bounds padding rows, whose gate reads as
            # 0 and would otherwise overflow to inf and poison the matmul.
            cols = slice(0, i_c * bc)
            g_ref = g_i[:, :, :1]
            lhs_decay = torch.exp2((g_i - g_ref).clamp_max(0.0))
            rhs = k_t[:, :, cols] * torch.exp2(
                (g_ref - g_t[:, :, cols]).clamp_max(0.0)
            )
            rhs = rhs.transpose(-1, -2)
            Aqk_t[:, :, rows, cols] = (q_i * lhs_decay) @ rhs
            Akk_t[:, :, rows, cols] = (k_i * lhs_decay) @ rhs

    o_i = torch.arange(bt, device=q.device)
    Akk_t = (Akk_t * b_t).masked_fill_(o_i.unsqueeze(-1) <= o_i, 0.0)
    Aqk = torch.zeros(
        1, num_tokens, num_heads, bt, dtype=torch.float32, device=q.device
    )
    A = torch.zeros_like(Aqk)
    _untile(Aqk_t, pos, valid, Aqk)
    _untile(_inv_unit_lower(Akk_t), pos, valid, A)
    return Aqk, A


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
    """WY-transformed keys and values, plus the chunk-end scaled keys::

        u  = A @ (beta * v)
        w  = A @ (beta * k * exp2(g))
        kg = k * exp2(g_last - g)

    ``g_last`` is the cumulative gate of the chunk's last valid token, so ``kg``
    carries every key forward to the chunk boundary. The third slot of the tuple
    keeps upstream's ``w, u, _, kg`` unpacking; ``q`` is unused for the same
    reason (upstream can fuse the ``q`` gating here, this port does it in
    ``chunk_gla_fwd_o_gk``).
    """
    chunk_size = A.shape[-1]
    pos, valid = _chunk_tiles(cu_seqlens, chunk_indices, chunk_size, k.shape[1])
    k_t = _tile(k, pos, valid)
    g_t = _tile(gk, pos, valid)
    b_t = _tile(beta.unsqueeze(-1), pos, valid)
    A_t = _tile(A, pos, valid)

    u_t = A_t @ (b_t * _tile(v, pos, valid))
    w_t = A_t @ (b_t * k_t * torch.exp2(g_t))
    kg_t = k_t * torch.exp2(_chunk_last_gate(g_t, valid) - g_t)

    # fp32 everywhere: unlike the triton kernels there is no tensor-core reason
    # to round these intermediates down to the input dtype.
    w = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    u = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
    kg = torch.zeros_like(w)
    _untile(w_t, pos, valid, w)
    _untile(u_t, pos, valid, u)
    _untile(kg_t, pos, valid, kg)
    return w, u, None, kg


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = FLA_CHUNK_SIZE,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    use_exp2: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Chunk-level state recurrence over the WY-transformed chunks::

        v_new = u - w @ h^T
        h_next = h * exp2(g_last) + v_new^T @ kg

    ``k`` is the ``kg`` of ``recompute_w_u_fwd``. Returns the state ``h``
    *entering* each chunk (``[1, NT, H, V, K]``), the corrected values ``v_new``
    and, if asked, the fp32 per-sequence ``final_state``. Only the chunk axis is
    sequential: chunks that sit at the same position in different sequences are
    stepped together.
    """
    num_heads, head_dim, value_dim = k.shape[2], k.shape[3], u.shape[-1]
    pos, valid = _chunk_tiles(cu_seqlens, chunk_indices, chunk_size, k.shape[1])
    kg_t = _tile(k, pos, valid)
    w_t = _tile(w, pos, valid)
    u_t = _tile(u, pos, valid)
    decay_last = torch.exp2(_chunk_last_gate(_tile(gk, pos, valid), valid))

    if initial_state is None:
        state = torch.zeros(
            cu_seqlens.numel() - 1,
            num_heads,
            value_dim,
            head_dim,
            dtype=torch.float32,
            device=k.device,
        )
    else:
        state = initial_state.float().clone()
    h = state.new_zeros(1, pos.shape[0], num_heads, value_dim, head_dim)
    v_new_t = torch.zeros_like(u_t)

    seq, step = chunk_indices[:, 0].long(), chunk_indices[:, 1].long()
    for i_t in range(int(step.max().item()) + 1 if pos.shape[0] else 0):
        rows = (step == i_t).nonzero().flatten()
        i_n = seq[rows]
        h_t = state[i_n]
        h[0, rows] = h_t
        v_new_t[rows] = u_t[rows] - w_t[rows] @ h_t.transpose(-1, -2)
        state[i_n] = (
            h_t * decay_last[rows] + v_new_t[rows].transpose(-1, -2) @ kg_t[rows]
        )

    v_new = torch.zeros_like(u)
    _untile(v_new_t, pos, valid, v_new)
    return h, v_new, state if output_final_state else None


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
) -> torch.Tensor:
    """Chunk output on XPU: inter-chunk state read plus intra-chunk attention::

        o[t] = (scale * q[t] * exp(g[t])) @ h_chunk^T + sum_{s <= t} A[t, s] * v[s]

    ``v`` is the ``v_new`` of the state recurrence, ``A`` is ``Aqk`` (which
    already carries ``scale``; the kernel applies the causal mask itself) and
    ``h`` the state entering ``t``'s chunk, kept as ``[1, NT, H, V, K]``.

    Two conventions were pinned against the torch stage on device: the kernel
    exponentiates ``g`` with ``exp``, not ``exp2``, so the ``RCP_LN2`` factor
    that ``fused_kda_gate_chunk_cumsum`` bakes in is divided out here, and ``h``
    must not be transposed (``[1, NT, H, K, V]`` gives a ~1.3 relative L2 error).
    ``chunk_indices`` is only validated by the wrapper, never consumed.
    """
    return kunlun_ops.chunk_gla_fwd_o_gk(
        q,
        v,
        g / RCP_LN2,
        A,
        h,
        o,
        scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )


def _chunk_kda_fwd_with_cumulative_g(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    safe_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked delta rule, one torch stage per upstream kernel.

    ``g`` must already be chunk-local cumulatively-summed AND scaled by
    ``RCP_LN2`` -- exactly what ``fused_kda_gate_chunk_cumsum`` returns -- so
    every stage boundary keeps upstream's contract and can be swapped for a
    kernel one at a time.

    ``initial_state`` is the dense per-request state ``[N, H, V, K]`` produced by
    ``gather_initial_states``; the per-request final states are returned rather
    than written into the paged cache (the caller does that).
    """
    Aqk, A = chunk_kda_fwd_intra(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
    )
    w, u, _, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gk=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    del A
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        use_exp2=True,
    )
    del w, u, kg
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        o=v,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    del Aqk, v_new, h
    if final_state is not None:
        # The paged cache keeps the state in its own dtype; upstream hands back
        # fp32 and lets the caller round.
        final_state = final_state.to(
            v.dtype if initial_state is None else initial_state.dtype
        )
    return o, final_state


def chunk_kda_with_fused_gate_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked KDA prefill, split like upstream: chunk table, gate, delta rule."""
    chunk_size = FLA_CHUNK_SIZE
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    g, beta = fused_kda_gate_chunk_cumsum(
        raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    )
    return _chunk_kda_fwd_with_cumulative_g(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=lower_bound is not None,
    )


def chunk_kda_with_fused_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run chunk KDA from raw gate and beta projections."""
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    o, final_state = chunk_kda_with_fused_gate_fwd(
        q=q,
        k=k,
        v=v.contiguous(),
        raw_g=raw_g.contiguous(),
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        scale=scale,
        initial_state=initial_state.contiguous()
        if initial_state is not None
        else None,
        output_final_state=output_final_state,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state


def fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA single-token decode from packed post-conv QKV.

    ``mixed_qkv`` is ``[B, 2 * H * K + H * V]``, ``initial_state`` is the paged
    ``[num_slots, H, V, K]`` cache and is updated in place at ``state_indices``.
    """
    out = torch.ops.xspeedgate_ops.fused_recurrent_kda_packed_decode(
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        state_indices,
        scale,
    )
    return out, initial_state


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    fuse_gate: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA multi-query (spec decode) recurrence over the paged state cache."""
    scale = k.shape[-1] ** -0.5
    qf = l2norm_fwd(q.float())[0] * scale
    kf = l2norm_fwd(k.float())[0]
    vf = v[0].float()
    decay = kda_gate(raw_g, A_log, dt_bias, lower_bound)[0].exp()
    beta = torch.sigmoid(raw_beta[0].float())

    result = torch.empty_like(vf)
    starts = cu_seqlens.tolist()
    indices = ssm_state_indices
    if indices.dim() == 1:
        indices = indices.unsqueeze(-1)
    index_rows = indices.tolist()
    accepted = None if num_accepted_tokens is None else num_accepted_tokens.tolist()

    for i in range(len(starts) - 1):
        begin, end = starts[i], starts[i + 1]
        first = 0 if accepted is None else accepted[i] - 1
        state = initial_state[index_rows[i][first]].float()
        for t in range(begin, end):
            state = _delta_rule_scan(
                qf, kf, vf, decay, beta, state, t, t + 1, result
            )
            slot = index_rows[i][t - begin]
            if slot > 0:
                initial_state[slot] = state.to(initial_state.dtype)
    if out is not None:
        out[0] = result.to(out.dtype)
        return out, initial_state
    return result.unsqueeze(0).to(v.dtype), initial_state


def patch_kda_model(mod) -> None:
    """Swap the conv / state-gather kernels used by ``KimiK3DeltaAttention``.

    NOTE: no imports here. Importing ``vllm.models.kimi_k3.nvidia.ops.*`` from a
    post-import hook makes that package's *relative* ``from .attn_res import ...``
    run before the plugin's module mapping can redirect it, which silently pulls
    in the upstream triton ``attn_res``.
    """
    if not hasattr(mod, "KimiK3DeltaAttention"):
        # Module body still executing: its own `from ... import causal_conv1d_*`
        # would overwrite the patch. Retry on a later import event.
        return
    mod.causal_conv1d_fn = causal_conv1d_fn
    mod.causal_conv1d_update = causal_conv1d_update
    mod.gather_initial_states = gather_initial_states
    mod._kunlun_kda_patched = True
    logger.info("[KunlunPlugin] KDA conv / gather kernels -> torch")


def patch_kda_ops(mod) -> None:
    """Swap the KDA delta-rule kernels (prefill chunk + recurrent decode)."""
    if not all(
        hasattr(mod, name)
        for name in (
            "chunk_kda_with_fused_gate",
            "fused_recurrent_kda",
            "fused_recurrent_kda_packed_decode",
        )
    ):
        return
    mod.chunk_kda_with_fused_gate = chunk_kda_with_fused_gate
    mod.fused_recurrent_kda = fused_recurrent_kda
    mod.fused_recurrent_kda_packed_decode = fused_recurrent_kda_packed_decode
    mod._kunlun_kda_patched = True
    logger.info("[KunlunPlugin] KDA delta-rule kernels -> torch")
    # The replacement above already calls the xspeedgate gate stage, so this only
    # matters if the upstream chunk path runs: patch it on ``chunk`` as well as on
    # the package, since chunk_kda_with_fused_gate_fwd resolves the name in its
    # own module.
    chunk_mod = getattr(mod, "chunk", None)
    if chunk_mod is None or not hasattr(chunk_mod, "fused_kda_gate_chunk_cumsum"):
        return
    mod.fused_kda_gate_chunk_cumsum = fused_kda_gate_chunk_cumsum
    chunk_mod.fused_kda_gate_chunk_cumsum = fused_kda_gate_chunk_cumsum
    logger.info("[KunlunPlugin] fused_kda_gate_chunk_cumsum -> xspeedgate_ops")


def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    is_rms_norm: bool = False,
    H: int = 1,
    g_stride_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Gated (RMS) norm, ``x`` is ``[T, D]`` and ``g`` is ``[T, H, D]``.

    ``xspeedgate_ops.layer_norm_gated_fwd`` takes upstream's arguments in the
    same order and supports the ``sigmoid`` gate K3's ``o_norm`` uses (the
    separate ``rms_norm_gated_fwd`` op hardcodes the swish gate). Returns
    upstream's ``(y, mean, rstd, residual_out)``.
    """
    return torch.ops.xspeedgate_ops.layer_norm_gated_fwd(
        x,
        g,
        weight,
        bias,
        activation,
        eps,
        residual,
        out_dtype,
        residual_dtype,
        is_rms_norm,
        H,
        g_stride_n,
    )


def patch_rms_norm_gated(mod) -> None:
    """``o_norm``'s forward_cuda goes through the triton layer_norm_gated_fwd.

    Only the kernel entry point is swapped, so ``rms_norm_gated``'s reshaping
    (``H``, ``g_stride_n``, residual dtype) stays upstream's.
    """
    if not hasattr(mod, "FusedRMSNormGated"):
        return
    mod.layer_norm_gated_fwd = layer_norm_gated_fwd
    mod._kunlun_kda_patched = True
    logger.info("[KunlunPlugin] layer_norm_gated_fwd -> xspeedgate_ops")
