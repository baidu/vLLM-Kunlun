# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op override for ``vllm.v1.worker.gpu.sample.prompt_logprob``.

Reimplements the single Triton function ``get_prompt_logprobs_token_ids`` (which
gathers the shifted next-token ids for each prompt position) on the Kunlun
native op ``torch.ops.xspeedgate_ops.get_prompt_logprobs_token_ids``.

The native op gains an ``out_int32`` flag; upstream never passes it and returns
int64, so the wrapper leaves it at its ``False`` default to match.
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
    return torch.ops.xspeedgate_ops.get_prompt_logprobs_token_ids(
        num_tokens,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
    )


# ``PromptLogprobsWorker`` resolves ``get_prompt_logprobs_token_ids`` from the
# upstream module's globals; install the native-op version there.
_up.get_prompt_logprobs_token_ids = get_prompt_logprobs_token_ids
logger.info("[KunlunPlugin] V2 prompt_logprob patched (xspeedgate_ops native)")
