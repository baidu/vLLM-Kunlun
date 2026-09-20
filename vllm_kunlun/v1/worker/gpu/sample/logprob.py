# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op override for ``vllm.v1.worker.gpu.sample.logprob``.

Reimplements the two Triton launchers on this module's live path -- the
log-softmax gather ``compute_token_logprobs`` (``_topk_log_softmax_kernel``) and
the selected-token rank count (``_ranks_kernel``) -- on the Kunlun native ops
``torch.ops.xspeedgate_ops.compute_token_logprobs`` / ``.ranks_kernel``.

Both native ops require float32 logits; ``compute_token_logprobs`` requires an
int64 ``token_ids`` matrix and ``ranks_kernel`` a 1-D int64 ``token_ids``
vector, so the wrappers cast to those before dispatching (the casts are no-ops
on the dtypes upstream actually passes).

The third upstream Triton launcher, ``_fill_logprob_token_ids_kernel`` (used
only when some request set ``SamplingParams.logprob_token_ids``), has no single
native op. The previous torch-native stand-in built its output with a per-row
``.tolist()`` loop, which host-syncs once per step. It is replaced here with a
fully vectorised, sync-free ``torch.where`` over fixed-shape tensors that
reproduces the kernel's per-row branch (custom token ids override the topk
columns when ``num_custom > 0``, else topk fills them). ``LogprobTokenIdsState``
is upstream's and is reused as-is.
"""

import logging

import torch
import vllm.v1.worker.gpu.sample.logprob as _up
from vllm.v1.outputs import LogprobsTensors

logger = logging.getLogger("vllm_kunlun")


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """Log-softmax gather at ``token_ids`` (native ``compute_token_logprobs``).

    The native op emits only the logprobs at ``token_ids`` (never the full
    ``[batch, vocab]`` matrix), matching the upstream kernel's memory
    behaviour. It requires float32 logits and an int64 index matrix.
    """
    lf = logits if logits.dtype == torch.float32 else logits.to(torch.float32)
    return torch.ops.xspeedgate_ops.compute_token_logprobs(
        lf, token_ids.to(torch.int64)
    )


def _selected_token_ranks(
    logits: torch.Tensor, sampled_token_ids: torch.Tensor
) -> torch.Tensor:
    """Rank of each sampled token = count of logits >= its logit (native op)."""
    lf = logits if logits.dtype == torch.float32 else logits.to(torch.float32)
    return torch.ops.xspeedgate_ops.ranks_kernel(
        lf, sampled_token_ids.reshape(-1).to(torch.int64)
    )


def compute_topk_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits=None,
    logprob_token_ids_state=None,
    expanded_idx_mapping=None,
    max_per_req_token_ids: int = 0,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape

    if max_per_req_token_ids == 0:
        # Fast path: no request asked for custom logprob_token_ids.
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
        logprobs = compute_token_logprobs(logits, logprob_token_ids)
    else:
        # Some requests specified logprob_token_ids. Build the
        # ``[batch_size, 1 + num_cols]`` token-id matrix and its validity mask
        # the way ``_fill_logprob_token_ids_kernel`` does, but vectorised: no
        # per-row Python loop and no ``.tolist()`` host sync (this runs on the
        # sampler's critical path).
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None
        device = logits.device
        num_cols = max(num_logprobs, max_per_req_token_ids)

        idx = expanded_idx_mapping.to(torch.long)  # [B] -> req_state_idx
        # ``num_token_ids``/``token_ids`` are the state's UVA/staged buffers;
        # the Kunlun patch may back them with plain device tensors -- ``.gpu``
        # is valid either way.
        num_custom = logprob_token_ids_state.num_token_ids.gpu[idx].to(torch.long)
        per_req = logprob_token_ids_state.token_ids.gpu  # [max_num_reqs, MAX]
        per_req_rows = per_req[idx]  # [B, MAX_LOGPROB_TOKEN_IDS]

        col = torch.arange(num_cols, device=device)  # [num_cols]
        col_b = col.unsqueeze(0)  # [1, num_cols]
        use_custom = (num_custom > 0).unsqueeze(1)  # [B, 1]

        # Custom source: gather columns, clamping the index so rows narrower
        # than ``num_cols`` never index out of bounds (those columns are
        # invalid anyway).
        custom_valid = col_b < num_custom.unsqueeze(1)  # [B, num_cols]
        cwidth = per_req_rows.shape[1]
        custom_tokens = per_req_rows[:, col.clamp(max=cwidth - 1)]  # [B, num_cols]

        # Topk source (no-op columns when num_logprobs == 0).
        if num_logprobs > 0:
            topk_ids = torch.topk(logits, num_logprobs, dim=-1).indices
            topk_tokens = topk_ids[:, col.clamp(max=num_logprobs - 1)]
            topk_valid = col_b < num_logprobs
        else:
            topk_tokens = torch.zeros(
                (batch_size, num_cols), dtype=torch.long, device=device
            )
            topk_valid = torch.zeros(
                (batch_size, num_cols), dtype=torch.bool, device=device
            )

        tokens = torch.where(use_custom, custom_tokens.to(torch.long), topk_tokens)
        valid = torch.where(use_custom, custom_valid, topk_valid)
        # Invalid columns must stay 0, matching upstream's ``new_zeros`` +
        # masked ``store`` (``_fill_logprob_token_ids_kernel``). ``tokens``
        # holds clamped duplicates / stale staged-buffer slots in those columns,
        # so emitting it verbatim would leak a real (but -inf) token id that can
        # clobber that token's true logprob when a row is consumed in full.
        tokens = torch.where(valid, tokens, tokens.new_zeros(()))

        logprob_token_ids = sampled_token_ids.new_zeros((batch_size, 1 + num_cols))
        logprob_token_ids[:, 0] = sampled_token_ids
        logprob_token_ids[:, 1:] = tokens.to(logprob_token_ids.dtype)

        valid_mask = torch.zeros_like(logprob_token_ids, dtype=torch.bool)
        valid_mask[:, 0] = True
        valid_mask[:, 1:] = valid

        logprobs = compute_token_logprobs(logits, logprob_token_ids)
        logprobs = logprobs.masked_fill(~valid_mask, float("-inf"))

    token_ranks = _selected_token_ranks(logits, sampled_token_ids)
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )


# Install into the upstream module's globals, so both its own code and the
# consumers that bind these names on import (sample/sampler.py,
# sample/prompt_logprob.py, spec_decode/rejection_sampler.py) get them.
_up.compute_token_logprobs = compute_token_logprobs
_up.compute_topk_logprobs = compute_topk_logprobs
logger.info("[KunlunPlugin] V2 logprob patched (xspeedgate_ops native)")
