# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""torch-native stand-ins for the V2 model-runner Triton kernels.

Deliberately depends on ``torch`` only -- no ``vllm``, no ``kunlun_ops`` -- so
these can be unit-tested on CPU without XPU hardware. Each function documents
the upstream kernel it mirrors and any intentional semantic divergence.
"""

import torch


class TorchKernel:
    """Stand-in for a Triton kernel object: ``obj[grid](*args, **kw)`` -> fn.

    Lets a replacement swap a module-private kernel *object* rather than
    rewriting the method that launches it, which keeps the surrounding upstream
    logic intact.
    """

    def __init__(self, fn):
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", "kunlun_torch_kernel")

    def __getitem__(self, grid):
        return self._fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def segment_ids(cu_seqlens, num_segments: int, length: int):
    """Map each position in ``[0, length)`` to the segment containing it.

    ``cu_seqlens`` is a cumulative offset array of ``num_segments + 1`` entries,
    as produced by ``query_start_loc`` / ``cu_num_logits``. Equivalent to
    ``searchsorted(cu_seqlens[1 : num_segments + 1], arange(length),
    right=True)``, including the handling of zero-length segments, which are
    skipped rather than claiming a position.

    Built from ``index_add_`` + ``cumsum`` instead of ``torch.searchsorted``
    because searchsorted has no Kunlun XPU kernel and silently falls back to
    CPU, which costs a host sync plus a D2H/H2D round trip of the whole
    position array on every step (see vllm_kunlun/v1/worker/block_table.py:8-11
    for the same problem on the V1 path).

    The marker array is one longer than ``length`` so that a boundary sitting
    exactly at ``length`` -- a trailing zero-length segment -- lands in the
    spare slot instead of being clamped onto the last real position.
    """
    boundaries = cu_seqlens[1:num_segments].to(torch.long).clamp(0, length)
    marks = torch.zeros(length + 1, dtype=torch.long, device=cu_seqlens.device)
    marks.index_add_(0, boundaries, torch.ones_like(boundaries))
    return marks.cumsum(0)[:length]


def scatter_num_accepted(idx_mapping, num_sampled, num_accepted) -> None:
    """Mirror ``mamba_hybrid._scatter_num_accepted_kernel``.

    Upstream kernel (model_states/mamba_hybrid.py:337-348), grid
    ``(idx_mapping.shape[0],)``, one program per batch row::

        req = idx_mapping[row]
        if req < 0: return
        num_accepted[req] = max(num_sampled[row], 1)

    The ``-1`` sentinels appear only under pipeline parallelism, where
    ``PPHandler.get_prev_sampled_outputs`` (upstream pp_utils.py:115) masks out
    rows whose request was freed or needs no sampled output. V2 with PP does
    run on Kunlun, but a sentinel needs a request to be freed or to stop
    needing sampled output mid-flight, so the branch is covered by the unit
    tests rather than by anything exercised on hardware so far. It costs
    nothing here -- one extra term in a ``torch.where`` over a mask that is
    computed either way -- and matching upstream is what keeps a filtered row
    from silently corrupting another request's state.

    Written to perform **no device-to-host transfer**, for the same reason as
    ``post_update``: this runs in ``GPUModelRunner.postprocess_sampled``, one
    statement after it, so a sync here stalls the host exactly as badly. That
    rules out ``idx_mapping[valid]`` and any ``.numel()`` test on its result --
    a boolean-mask selection has a data-dependent shape and synchronises.
    Invalid rows are instead folded onto destination 0 with a zero contribution
    and every write is an ``index_add_`` accumulation. ``idx_mapping`` is
    injective over its valid rows, so a valid row landing on 0 still ends up at
    ``cur + (new - cur) == new``.
    """
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    zero_long = torch.zeros((), dtype=torch.long, device=idx_mapping.device)
    valid = idx_mapping >= 0
    req = torch.where(valid, idx_mapping.to(torch.long), zero_long)
    new = num_sampled[:num_reqs].clamp(min=1).to(num_accepted.dtype)
    cur = num_accepted[req]
    num_accepted.index_add_(
        0, req, torch.where(valid, new - cur, torch.zeros_like(cur))
    )


def post_update(
    idx_mapping,  # [num_reqs] batch_idx -> req_state_idx; negative means skip
    num_computed_tokens,  # [max_num_reqs]
    last_sampled_tokens,  # [max_num_reqs]
    output_bin_counts,  # [max_num_reqs, vocab_size] or None
    sampled_tokens,  # [num_reqs, num_speculative_steps + 1]
    num_sampled,  # [max_num_reqs]
    num_rejected,  # [max_num_reqs]
    query_start_loc,  # [num_reqs + 1] or None
    all_token_ids,  # [max_num_reqs, max_model_len]
    total_len,  # [max_num_reqs]
) -> None:
    """Mirror ``input_batch._post_update_kernel``.

    Upstream kernel (input_batch.py:459-517), grid ``(num_reqs,)``, one program
    per batch row ``b`` with ``s = idx_mapping[b]``::

        if s < 0: return
        base = total_len[s]
        if num_sampled[b] > 0:
            last_sampled_tokens[s] = sampled_tokens[b, num_sampled[b] - 1]
            total_len[s] = base + num_sampled[b]
        for i in range(num_sampled[b]):
            all_token_ids[s, base + i] = sampled_tokens[b, i]
            if output_bin_counts is not None:
                output_bin_counts[s, sampled_tokens[b, i]] += 1
        query_len = query_start_loc[b + 1] - query_start_loc[b]  (0 if None)
        num_computed_tokens[s] += query_len - num_rejected[b]

    Written to perform **no device-to-host transfer**. It runs inside
    ``GPUModelRunner.postprocess_sampled``, right after the runner hands the
    sampled tokens to ``AsyncOutput`` (model_runner.py:1425), which issues its
    D2H copies on a separate stream and records an event the scheduler waits on
    later; upstream even orders those two statements so the copy is recorded
    first (model_runner.py:1444-1447). One ``.tolist()`` here would block the
    host on the main stream until this step's forward and sampling finish,
    defeating that arrangement entirely.

    Two consequences shape the code:

    * No boolean-mask *selection* (``x[mask]``, ``masked_select``, ``nonzero``):
      those produce a data-dependent shape and therefore synchronise. Masking
      is done with ``torch.where`` over fixed-shape tensors instead.
    * Filtered rows (``idx_mapping < 0``, only produced under pipeline
      parallelism) are folded onto destination 0 with a zero contribution
      rather than skipped. Several of them can then share a destination, and
      duplicate destinations in an indexed *assignment* have no defined
      ordering -- so every write is an ``index_add_`` accumulation, which is
      well defined either way. ``idx_mapping`` is injective over its valid rows,
      so the real writes never collide with each other.
    """
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    assert all_token_ids.is_contiguous()

    zero_long = torch.zeros((), dtype=torch.long, device=idx_mapping.device)
    valid = idx_mapping >= 0
    req = torch.where(valid, idx_mapping.to(torch.long), zero_long)
    # Folding invalid rows to a zero count makes ``valid`` implicit below:
    # nothing is written for a row with no sampled tokens anyway.
    ns = torch.where(valid, num_sampled[:num_reqs].to(torch.long), zero_long)
    # Write offset, read before total_len is advanced -- the upstream kernel
    # likewise loads total_len once up front.
    base = total_len[req].to(torch.long)

    max_len = all_token_ids.shape[1]
    max_sampled = sampled_tokens.shape[1]
    col = torch.arange(max_sampled, device=idx_mapping.device)
    written = ns.unsqueeze(1) > col.unsqueeze(0)  # [num_reqs, max_sampled]
    # The clamp only guards rows that would run past max_model_len, which the
    # scheduler should never produce; those entries are masked off anyway.
    pos = (base.unsqueeze(1) + col.unsqueeze(0)).clamp(0, max_len - 1)

    # all_token_ids[req, base + i] = sampled_tokens[b, i]  for i < ns
    flat_tokens = all_token_ids.view(-1)
    dst = (req.unsqueeze(1) * max_len + pos).reshape(-1)
    src = sampled_tokens[:num_reqs].to(flat_tokens.dtype).reshape(-1)
    cur = flat_tokens[dst]
    flat_tokens.index_add_(
        0, dst, torch.where(written.reshape(-1), src - cur, torch.zeros_like(cur))
    )

    # last_sampled_tokens[req] = sampled_tokens[b, ns - 1]  when ns > 0
    last = (
        sampled_tokens[:num_reqs]
        .gather(1, (ns - 1).clamp(min=0).unsqueeze(1))
        .squeeze(1)
    )
    cur_last = last_sampled_tokens[req]
    last_sampled_tokens.index_add_(
        0,
        req,
        torch.where(
            ns > 0,
            last.to(last_sampled_tokens.dtype) - cur_last,
            torch.zeros_like(cur_last),
        ),
    )

    # total_len[req] = base + ns. total_len[req] already holds base, so the
    # accumulation form is exact rather than a rewrite of the same value, and
    # it is a no-op for ns == 0 just like the upstream branch.
    total_len.index_add_(0, req, ns.to(total_len.dtype))

    if output_bin_counts is not None:
        assert output_bin_counts.is_contiguous()
        vocab_size = output_bin_counts.shape[1]
        # clamp(), not clamp_(): ``.to(torch.long)`` returns the caller's tensor
        # unchanged when sampled_tokens is already int64.
        tok = sampled_tokens[:num_reqs].to(torch.long).clamp(0, vocab_size - 1)
        output_bin_counts.view(-1).index_add_(
            0,
            (req.unsqueeze(1) * vocab_size + tok).reshape(-1),
            written.reshape(-1).to(output_bin_counts.dtype),
        )

    # num_computed_tokens[req] += query_len - num_rejected
    rejected = num_rejected[:num_reqs].to(torch.long)
    if query_start_loc is None:
        delta = -rejected
    else:
        query_len = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
        delta = query_len.to(torch.long) - rejected
    num_computed_tokens.index_add_(
        0, req, torch.where(valid, delta, zero_long).to(num_computed_tokens.dtype)
    )


def prepare_rope_positions(
    positions,  # [num_dims, max_num_tokens + 1] int64, written in place
    prefill_positions,  # [max_num_reqs * num_dims, max_model_len] int32
    prefill_delta,  # [max_num_reqs] int32 (all zeros for XD-RoPE)
    idx_mapping,  # [num_reqs] batch_idx -> req_state_idx
    query_start_loc,  # [num_reqs + 1] cumulative token offsets
    prefill_lens,  # [max_num_reqs]
    num_computed_tokens,  # [max_num_reqs]
    num_dims: int,
    max_model_len: int,
) -> None:
    """Mirror ``mm/rope.py::_prepare_rope_positions_kernel``.

    Upstream kernel (mm/rope.py:166-214), grid ``(num_reqs,)``, per batch row
    ``i`` with ``s = idx_mapping[i]``::

        is_prefill = num_computed_tokens[s] < prefill_lens[s]
        for t in [0, query_len_i):
            orig = num_computed_tokens[s] + t
            pos[j] = prefill_positions[s * num_dims + j, orig] if is_prefill
                     else orig + prefill_delta[s]
            positions[j, query_start_loc[i] + t] = pos[j]

    The launch site (rope.py:118-131) passes ``stride0 = num_dims *
    max_model_len`` (per request) and ``stride1 = max_model_len`` (per dim), so
    the kernel's pointer arithmetic is equivalent to the 2-D index
    ``prefill_positions[s * num_dims + j, orig]``; the store side is equivalent
    to ``positions[j, query_start_loc[i] + t]``.

    Because ``query_start_loc`` is a cumsum, ``query_start_loc[i] + t`` is the
    global token index, so the whole batch is one contiguous write.

    Two intentional divergences from the kernel:

    * ``orig_pos`` is clamped to ``max_model_len - 1`` before the gather. The
      kernel masks only on ``t < query_len`` and would read past the row
      (undefined) if a prefill request were ever scheduled beyond
      ``prefill_len``.
    * negative ``idx_mapping`` entries are clamped to 0. Note this kernel --
      unlike the mamba scatter kernel -- does *not* filter ``-1``, so upstream
      would read garbage there too; the clamp only avoids a hard torch
      indexing error.
    """
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    device = positions.device
    qsl = query_start_loc[: num_reqs + 1].to(torch.long)
    # TODO: this .item() is the last host sync left in V2 input preparation.
    # Removing it means sizing the work off ``positions.shape[1] - 1`` and
    # masking on ``tok < qsl[num_reqs]`` (a device scalar) instead, which is
    # only worth doing together with BlockTables.compute_slot_mappings -- one
    # remaining sync stalls the whole phase, so the two have to go at once.
    num_tokens = int(qsl[num_reqs].item())
    if num_tokens == 0:
        return

    tok = torch.arange(num_tokens, device=device)
    batch = segment_ids(qsl, num_reqs, num_tokens)  # token -> batch row
    req = idx_mapping.to(torch.long).clamp(min=0)[batch]
    offset = tok - qsl[:num_reqs][batch]

    num_computed = num_computed_tokens.to(torch.long)[req]
    orig_pos = num_computed + offset
    is_prefill = num_computed < prefill_lens.to(torch.long)[req]
    delta = prefill_delta.to(torch.long)[req]

    rows = (req * num_dims).unsqueeze(0) + torch.arange(
        num_dims, device=device
    ).unsqueeze(
        1
    )  # [D, T]
    cols = orig_pos.clamp(max=max_model_len - 1).unsqueeze(0).expand(num_dims, -1)
    gathered = prefill_positions[rows, cols].to(torch.long)  # [D, T]

    positions[:num_dims, :num_tokens] = torch.where(
        is_prefill.unsqueeze(0), gathered, (orig_pos + delta).unsqueeze(0)
    )
