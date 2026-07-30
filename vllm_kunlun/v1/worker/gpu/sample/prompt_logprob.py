# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun torch-native replacement for ``vllm.v1.worker.gpu.sample.prompt_logprob``.

Reuses the upstream ``PromptLogprobsWorker`` (and its chunked-logits helper,
which already routes through the swapped ``logprob`` module) and only
reimplements the single Triton function ``get_prompt_logprobs_token_ids`` that
gathers the shifted next-token ids for each prompt position.
"""

import torch

from vllm_kunlun.v1.worker.gpu._upstream import load_upstream

_up = load_upstream("vllm.v1.worker.gpu.sample.prompt_logprob")

PromptLogprobsWorker = _up.PromptLogprobsWorker
compute_prompt_logprobs_with_chunking = _up.compute_prompt_logprobs_with_chunking


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    device = idx_mapping.device
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=device)
    num_reqs = idx_mapping.shape[0]
    idx = idx_mapping.tolist()
    qsl = query_start_loc.tolist()
    nct = num_computed_tokens.tolist()
    for b in range(num_reqs):
        rs = idx[b]
        qs = qsl[b]
        qe = qsl[b + 1]
        query_len = qe - qs
        if query_len <= 0:
            continue
        # Shift by one: the logprob at each position targets the next token.
        base = nct[rs] + 1
        token_ids[qs:qe] = all_token_ids[rs, base : base + query_len].to(torch.int64)
    return token_ids


# PromptLogprobsWorker (reused from the genuine module) resolves
# ``get_prompt_logprobs_token_ids`` from that module's globals.
_up.get_prompt_logprobs_token_ids = get_prompt_logprobs_token_ids
