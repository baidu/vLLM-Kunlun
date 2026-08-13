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
kernel exists the replacement forwards to it (``xspeedgate_ops.l2norm_fwd``,
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

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_SOFTPLUS_THRESHOLD = 20.0

# Upstream chunk length of the KDA prefill kernels.
FLA_CHUNK_SIZE = 64


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


def l2norm_fwd(x: torch.Tensor) -> torch.Tensor:
    """L2-normalise ``[B, T, H, D]`` along the last dim, same as upstream.

    ``xspeedgate_ops.l2norm_fwd`` computes ``x / sqrt(sum(x^2) + 1e-6)`` (eps is
    fixed in the wrapper) and requires a contiguous 4-D fp32 input; it flattens
    the leading dims into a row count.
    """
    return torch.ops.xspeedgate_ops.l2norm_fwd(x.float().contiguous())


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
    """Sequential scan standing in for the chunked KDA prefill kernels.

    ``initial_state`` is the dense per-request state ``[N, H, V, K]`` produced by
    ``gather_initial_states``; the per-request final states are returned rather
    than written into the paged cache (the caller does that).
    """
    qf = q[0].float() * scale
    kf = k[0].float()
    vf = v[0].float()
    decay = kda_gate(raw_g, A_log, g_bias, lower_bound)[0].exp()
    beta = torch.sigmoid(raw_beta[0].float())

    out = torch.empty_like(vf)
    starts = cu_seqlens.tolist()
    if initial_state is None:
        states = torch.zeros(
            len(starts) - 1,
            vf.shape[1],
            vf.shape[2],
            kf.shape[2],
            dtype=torch.float32,
            device=vf.device,
        )
        state_dtype = v.dtype
    else:
        states = initial_state.float().clone()
        state_dtype = initial_state.dtype
    for i in range(len(starts) - 1):
        states[i] = _delta_rule_scan(
            qf, kf, vf, decay, beta, states[i], starts[i], starts[i + 1], out
        )
    final_state = states.to(state_dtype) if output_final_state else None
    return out.unsqueeze(0).to(v.dtype), final_state


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
    qf = l2norm_fwd(q)[0] * scale
    kf = l2norm_fwd(k)[0]
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
    # The gate stage is only reached if the upstream chunk path runs (the
    # replacement above bypasses it), so patch it on ``chunk`` as well as on the
    # package: chunk_kda_with_fused_gate_fwd resolves the name in its own module.
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
