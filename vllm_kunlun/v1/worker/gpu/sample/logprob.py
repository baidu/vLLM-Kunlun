# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacements for ``vllm.v1.worker.gpu.sample.logprob``.

``compute_token_logprobs`` and ``compute_topk_scores`` launch the Triton
``_topk_log_softmax_kernel`` / ``_fill_logprob_token_ids_kernel`` /
``_ranks_kernel`` upstream. Kunlun XPU cannot JIT-compile Triton kernels, so
each launch is replaced by its xspeedgate_ops equivalent.

Preconditions the ops enforce (adapted here):
  * ``compute_token_logprobs`` requires fp32 logits (the fp16/bf16 widening is
    exact -- the native path widened internally anyway) and int64 2-D
    ``token_ids``; the op allocates and returns its output itself.
  * ``ranks_kernel`` requires fp32 logits plus 1-D int32/int64 token ids.
    Widening fp16/bf16 -> fp32 is exact and order-preserving, so the
    ``logits >= x`` counts are bit-identical to the native fp16 comparison.
    This only runs when logprobs are requested, never on the sampling hot path.

``_fill_logprob_token_ids_torch`` is carried verbatim from upstream and stays
**live**: the ``fill_logprob_token_ids`` op hard-caps ``num_cols`` at
``_FILL_LOGPROB_MAX_COLS`` (128), so the wider ``num_cols`` path (e.g.
logprobs > 128, or logprobs=-1 together with custom logprob_token_ids) keeps
the native loop.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)
from vllm.v1.outputs import LogprobsTensors

logger = logging.getLogger("vllm_kunlun")

# `fill_logprob_token_ids` hard-caps `num_cols` at 128: its wrapper checks
# `num_cols <= FILL_LOGPROB_TOKEN_IDS_MAX_COLS` and the kernel's LM budget is
# sized for exactly 128. The op mirrors vLLM's `MAX_LOGPROB_TOKEN_IDS`, which
# is what sizes `LogprobTokenIdsState.token_ids` -- so today the two agree, but
# the table width can be raised on the vLLM side independently of the op's cap.
_FILL_LOGPROB_MAX_COLS = 128


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernel computes
    # max + logsumexp per row and only emits logprobs at `token_ids`.
    token_ids = token_ids.to(torch.int64)
    # The op allocates and returns the output itself and derives its own
    # blocking. It requires fp32 logits (fp16/bf16 widening is exact).
    return torch.ops.xspeedgate_ops.compute_token_logprobs(
        logits if logits.dtype == torch.float32 else logits.to(torch.float32),
        token_ids,
    )


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | None = None,
    logprob_token_ids_state: "LogprobTokenIdsState | None" = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape

    if max_per_req_token_ids == 0:
        # Fast path: no request asked for custom logprob_token_ids.
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
    else:
        # Some requests specified logprob_token_ids. Build the [batch_size,
        # 1 + max_cols] token_ids matrix and validity mask on the GPU, overriding
        # the topk columns with per-request tokens where applicable.
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None

        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
            topk_token_ids = topk_token_ids.to(torch.int32)
        else:
            # This tensor just used as an int32 pointer, data not accessed.
            topk_token_ids = logprob_token_ids_state.token_ids.gpu

        num_cols = max(num_logprobs, max_per_req_token_ids)
        num_per_req_token_ids = logprob_token_ids_state.num_token_ids.gpu
        per_req_token_ids = logprob_token_ids_state.token_ids.gpu
        if num_cols <= min(per_req_token_ids.shape[1], _FILL_LOGPROB_MAX_COLS):
            # The op allocates and returns both outputs itself; it requires
            # `per_req_token_ids` to be at least `num_cols` wide and caps
            # `num_cols` at _FILL_LOGPROB_MAX_COLS (see the constant).
            logprob_token_ids, valid_mask = (
                torch.ops.xspeedgate_ops.fill_logprob_token_ids(
                    sampled_token_ids.reshape(-1),
                    topk_token_ids,
                    expanded_idx_mapping,
                    num_per_req_token_ids,
                    per_req_token_ids,
                    num_logprobs,
                    num_cols,
                )
            )
        else:
            # `num_cols` exceeds what the op can express (e.g. logprobs > 128 or
            # logprobs=-1 together with custom logprob_token_ids): keep native.
            logprob_token_ids = sampled_token_ids.new_zeros((batch_size, 1 + num_cols))
            valid_mask = torch.zeros_like(logprob_token_ids, dtype=torch.bool)
            _fill_logprob_token_ids_torch(
                logprob_token_ids,
                valid_mask,
                sampled_token_ids,
                topk_token_ids,
                expanded_idx_mapping,
                num_per_req_token_ids,
                per_req_token_ids,
                num_logprobs,
            )
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
        scores = scores.masked_fill(~valid_mask, float("-inf"))

    # The op returns the ranks and requires fp32 logits plus 1-D int32/int64
    # token ids.
    token_ranks = torch.ops.xspeedgate_ops.ranks_kernel(
        logits if logits.dtype == torch.float32 else logits.to(torch.float32),
        sampled_token_ids.reshape(-1),
    )
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=scores,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )


def _fill_logprob_token_ids_torch(
    out_token_ids: torch.Tensor,  # [B, 1 + num_cols]
    valid_mask: torch.Tensor,  # [B, 1 + num_cols] bool
    sampled_token_ids: torch.Tensor,  # [B]
    topk_token_ids: torch.Tensor,  # [B, num_topk] (dummy when num_topk == 0)
    expanded_idx_mapping: torch.Tensor,  # [B] -> req_state_idx
    num_per_req_token_ids: torch.Tensor,  # [max_num_reqs]
    per_req_token_ids: torch.Tensor,  # [max_num_reqs, max_logprob_token_ids]
    num_topk: int,
) -> None:
    """Pure-torch replacement for `_fill_logprob_token_ids_kernel`.

    Column 0 is always the sampled token (valid). The remaining columns come
    from each request's custom `per_req_token_ids[:num_custom]` when it set any,
    otherwise from `topk_token_ids[:num_topk]`; only the valid prefix per row is
    written, leaving the rest zero/invalid (matching the kernel).

    Kept live for the ``num_cols > _FILL_LOGPROB_MAX_COLS`` path.
    """
    batch_size = out_token_ids.shape[0]
    num_cols = out_token_ids.shape[1] - 1
    device = out_token_ids.device

    out_token_ids[:, 0] = sampled_token_ids
    valid_mask[:, 0] = True
    if num_cols == 0 or batch_size == 0:
        return

    req = expanded_idx_mapping.to(torch.int64)
    num_custom = num_per_req_token_ids[req].to(torch.int64)  # [B]
    use_custom = num_custom > 0
    cols = torch.arange(num_cols, device=device)
    limit = torch.where(
        use_custom, num_custom, num_custom.new_full((), num_topk).expand_as(num_custom)
    )
    valid = cols.unsqueeze(0) < limit.unsqueeze(1)  # [B, num_cols]

    src = out_token_ids.new_zeros((batch_size, num_cols))
    if per_req_token_ids.shape[1] > 0:
        w = min(num_cols, per_req_token_ids.shape[1])
        cust = per_req_token_ids[req][:, :w].to(out_token_ids.dtype)
        src[:, :w] = torch.where(use_custom.unsqueeze(1), cust, src[:, :w])
    if num_topk > 0:
        w = min(num_cols, topk_token_ids.shape[1])
        tk = topk_token_ids[:, :w].to(out_token_ids.dtype)
        src[:, :w] = torch.where((~use_custom).unsqueeze(1), tk, src[:, :w])

    tgt_tokens = out_token_ids[:, 1:]
    tgt_mask = valid_mask[:, 1:]
    tgt_tokens[valid] = src[valid]
    tgt_mask[valid] = True


compute_token_logprobs._kunlun_patched = True
compute_topk_scores._kunlun_patched = True
