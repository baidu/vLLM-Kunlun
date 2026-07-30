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


def scatter_num_accepted(idx_mapping, num_sampled, num_accepted) -> None:
    """Mirror ``mamba_hybrid._scatter_num_accepted_kernel``.

    Upstream kernel (model_states/mamba_hybrid.py:337-348), grid
    ``(idx_mapping.shape[0],)``, one program per batch row::

        req = idx_mapping[row]
        if req < 0: return
        num_accepted[req] = max(num_sampled[row], 1)

    ``idx_mapping`` is injective over its valid rows, so this scatter has no
    duplicate-index race. The ``-1`` sentinels only appear under pipeline
    parallelism (rows filtered out on non-last PP ranks).

    Cost note: the boolean mask needs the selected-row count, i.e. one small
    D2H sync -- the same order of cost as the ``.item()`` already accepted in
    ``BlockTables.compute_slot_mappings``.
    """
    valid = idx_mapping >= 0
    rows = idx_mapping[valid].to(torch.long)
    if rows.numel() == 0:
        return
    num_accepted[rows] = num_sampled[valid].clamp(min=1).to(num_accepted.dtype)


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
    # One small D2H sync; the V2 runner offers no sync-free way to size this.
    num_tokens = int(qsl[num_reqs].item())
    if num_tokens == 0:
        return

    tok = torch.arange(num_tokens, device=device)
    # token -> batch row. searchsorted on the per-request *ends* handles
    # zero-length requests correctly and needs no extra sync.
    batch = torch.searchsorted(qsl[1:].contiguous(), tok, right=True)
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
