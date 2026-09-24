# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op override for ``vllm.v1.worker.gpu.sample.prompt_logprob``.

Reimplements shifted next-token gathering on
``torch.ops.xspeedgate_ops.get_prompt_logprobs_token_ids``. Upstream requires
int64 output, so the native op's optional int32-output flag remains disabled.
The empty case is returned directly because it has no kernel work.
"""

import logging

import torch
import vllm.v1.worker.gpu.sample.prompt_logprob as _up

logger = logging.getLogger("vllm_kunlun")


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    if num_tokens == 0:
        return idx_mapping.new_empty((0,), dtype=torch.int64)
    return torch.ops.xspeedgate_ops.get_prompt_logprobs_token_ids(
        num_tokens, query_start_loc, idx_mapping, num_computed_tokens, all_token_ids
    )


# ``PromptLogprobsWorker`` resolves this function from the upstream module's
# globals; install the native-op version there.
_up.get_prompt_logprobs_token_ids = get_prompt_logprobs_token_ids
logger.info("[KunlunPlugin] V2 prompt_logprob patched (xspeedgate_ops native)")
