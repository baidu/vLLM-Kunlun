# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op overrides for ``vllm.v1.worker.gpu.sample.logprob``.

Reimplements selected-token log-softmax and rank calculation on
``torch.ops.xspeedgate_ops.compute_token_logprobs`` / ``.ranks_kernel``. Both
native ops require float32 logits, so ``compute_topk_logprobs`` creates one FP32
view and shares it across top-k, logprob and rank work.

For requests with custom ``logprob_token_ids``, the installed wheel provides
``fill_logprob_token_ids``. Using it keeps the wrapper thin and avoids the
per-row ``.tolist()`` synchronization in the old torch stand-in. Empty batches
are handled before native dispatch, and invalid padded columns are masked to
``-inf`` exactly as upstream requires.
"""

import logging

import torch
import vllm.v1.worker.gpu.sample.logprob as _up
from vllm.v1.outputs import LogprobsTensors

logger = logging.getLogger("vllm_kunlun")


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    if logits.shape[0] == 0:
        return token_ids.new_empty(token_ids.shape, dtype=torch.float32)
    return torch.ops.xspeedgate_ops.compute_token_logprobs(
        logits.to(torch.float32), token_ids.to(torch.int64).contiguous()
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
    # Raw model logits may be FP16/BF16; both native logprobs and ranks
    # require FP32. Share the conversion without changing the caller's logits.
    lf = logits.to(torch.float32)
    batch_size = logits.shape[0]
    if max_per_req_token_ids == 0:
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = torch.topk(lf, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
        logprobs = compute_token_logprobs(lf, logprob_token_ids)
    else:
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None
        num_cols = max(num_logprobs, max_per_req_token_ids)
        topk_token_ids = torch.topk(lf, num_logprobs, dim=-1).indices.to(torch.int32)
        if batch_size == 0:
            logprob_token_ids = sampled_token_ids.new_empty((0, 1 + num_cols))
            logprobs = logits.new_empty((0, 1 + num_cols), dtype=torch.float32)
        else:
            (
                logprob_token_ids,
                valid_mask,
            ) = torch.ops.xspeedgate_ops.fill_logprob_token_ids(
                sampled_token_ids,
                topk_token_ids,
                expanded_idx_mapping,
                logprob_token_ids_state.num_token_ids.gpu,
                logprob_token_ids_state.token_ids.gpu,
                num_logprobs,
                num_cols,
            )
            logprobs = compute_token_logprobs(lf, logprob_token_ids)
            logprobs.masked_fill_(~valid_mask, float("-inf"))
    if batch_size == 0:
        token_ranks = sampled_token_ids.new_empty((0,))
    else:
        token_ranks = torch.ops.xspeedgate_ops.ranks_kernel(lf, sampled_token_ids)
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )


# Install into the upstream module's globals before the sampler, prompt-logprob
# worker and rejection sampler bind these functions.
_up.compute_token_logprobs = compute_token_logprobs
_up.compute_topk_logprobs = compute_topk_logprobs
logger.info("[KunlunPlugin] V2 logprob patched (xspeedgate_ops native)")
