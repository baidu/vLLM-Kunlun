# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun torch-native overrides for ``vllm.v1.worker.gpu.sample.logprob``.

Leaves the upstream ``LogprobTokenIdsState`` alone and reimplements the logprob
computation (log-softmax gather + selected-token ranks) with torch ops. The
custom per-request ``logprob_token_ids`` path is reimplemented with a small
per-row loop.
"""

import torch
import vllm.v1.worker.gpu.sample.logprob as _up
from vllm.v1.outputs import LogprobsTensors


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    token_ids = token_ids.to(torch.int64)
    lf = logits.to(torch.float32)
    log_probs = lf - torch.logsumexp(lf, dim=-1, keepdim=True)
    return torch.gather(log_probs, 1, token_ids)


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
    lf = logits.to(torch.float32)

    if max_per_req_token_ids == 0:
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
        logprobs = compute_token_logprobs(logits, logprob_token_ids)
    else:
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None
        num_cols = max(num_logprobs, max_per_req_token_ids)
        logprob_token_ids = sampled_token_ids.new_zeros((batch_size, 1 + num_cols))
        valid_mask = torch.zeros_like(logprob_token_ids, dtype=torch.bool)
        logprob_token_ids[:, 0] = sampled_token_ids
        valid_mask[:, 0] = True

        topk_token_ids = None
        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices

        idx = expanded_idx_mapping.to(torch.long)
        num_custom = logprob_token_ids_state.num_token_ids.gpu[idx].tolist()
        per_req = logprob_token_ids_state.token_ids.gpu
        for b in range(batch_size):
            nc = num_custom[b]
            if nc > 0:
                logprob_token_ids[b, 1 : 1 + nc] = per_req[idx[b], :nc].to(
                    logprob_token_ids.dtype
                )
                valid_mask[b, 1 : 1 + nc] = True
            elif num_logprobs > 0:
                logprob_token_ids[b, 1 : 1 + num_logprobs] = topk_token_ids[b].to(
                    logprob_token_ids.dtype
                )
                valid_mask[b, 1 : 1 + num_logprobs] = True
        logprobs = compute_token_logprobs(logits, logprob_token_ids)
        logprobs = logprobs.masked_fill(~valid_mask, float("-inf"))

    # Selected-token ranks: count logits >= the sampled token's logit.
    x = torch.gather(lf, 1, sampled_token_ids.view(-1, 1).to(torch.int64))
    token_ranks = (lf >= x).sum(dim=-1).to(torch.int64)

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
