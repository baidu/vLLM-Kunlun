# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacement for ``vllm.v1.worker.gpu.sample.prompt_logprob``.

``get_prompt_logprobs_token_ids`` launches the Triton
``_prompt_logprobs_token_ids_kernel`` upstream. Kunlun XPU cannot JIT-compile
Triton kernels, so it is replaced by the xspeedgate_ops equivalent.

``num_tokens`` is a host scalar and the op derives its own grid. The op also
requires ``query_start_loc`` to be exactly ``num_reqs + 1`` long, while the
runner hands over the CUDA-graph padded view (``[:num_reqs_padded + 1]``);
slice it so the padded tail cannot trip the length check.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)

logger = logging.getLogger("vllm_kunlun")


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    num_reqs = idx_mapping.shape[0]
    return torch.ops.xspeedgate_ops.get_prompt_logprobs_token_ids(
        num_tokens,
        query_start_loc[: num_reqs + 1],
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
    )


get_prompt_logprobs_token_ids._kunlun_patched = True
