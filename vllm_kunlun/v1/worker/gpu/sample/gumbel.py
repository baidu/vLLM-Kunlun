# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op overrides for ``vllm.v1.worker.gpu.sample.gumbel``.

Replaces the two Triton Gumbel-max entry points with the Kunlun native ops
``torch.ops.xspeedgate_ops.apply_temperature`` and ``.gumbel_sample``. The
native ``gumbel_sample`` uses a seeded Philox stream, preserving per-request
seed reproducibility, and implements the spec-decode processed-logits outputs
and ``use_fp64`` reduction.

The native sample result is ``[n, 1]`` while upstream callers require ``[n]``.
The wrapper restores that contract, returns argmax for negative request IDs as
upstream does, and preserves existing ``-inf`` masks around the current
temperature kernel so excluded tokens cannot become NaN.
"""

import logging

import torch
import vllm.v1.worker.gpu.sample.gumbel as _up

logger = logging.getLogger("vllm_kunlun")


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    if logits.shape[0] == 0:
        return
    # The current wheel's vectorized division turns -inf masks into NaN.
    # Preserve tokens excluded by grammar, bad words or logit processors.
    masked = torch.isneginf(logits)
    torch.ops.xspeedgate_ops.apply_temperature(
        logits, expanded_idx_mapping, temperature
    )
    logits.masked_fill_(masked, float("-inf"))


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    if logits.shape[0] == 0:
        return logits.new_empty((0,), dtype=torch.int64)
    sampled = torch.ops.xspeedgate_ops.gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature,
        output_processed_logits,
        output_processed_logits_col,
        use_fp64,
    ).squeeze(-1)
    # The wheel returns 0 for negative request IDs; upstream uses argmax.
    # Match upstream without a device-to-host synchronization.
    return torch.where(expanded_idx_mapping < 0, logits.argmax(dim=-1), sampled)


# ``SamplingStates`` and the samplers bind these functions at import time, so
# install the native-op versions before those modules execute.
_up.apply_temperature = apply_temperature
_up.gumbel_sample = gumbel_sample
logger.info("[KunlunPlugin] V2 gumbel patched (xspeedgate_ops native)")
