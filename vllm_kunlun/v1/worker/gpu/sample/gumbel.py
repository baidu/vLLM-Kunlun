# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op overrides for ``vllm.v1.worker.gpu.sample.gumbel``.

Replaces the two Triton Gumbel-max entry points with the Kunlun native ops
``torch.ops.xspeedgate_ops.apply_temperature`` and ``.gumbel_sample``. The
native ``gumbel_sample`` uses a seeded Philox stream, so per-request seed
reproducibility is preserved (unlike the torch-native stand-in this replaces),
and it also implements the spec-decode ``output_processed_logits`` /
``output_processed_logits_col`` outputs and the ``use_fp64`` reduction, so those
arguments are forwarded rather than rejected.

``gumbel_sample`` is only ever called on the sampled-position logits
(``num_tokens`` ~= number of requests). The native op returns an ``[n, 1]``
int64 tensor; upstream's Triton kernel returns a 1-D ``[n]`` tensor and callers
(``sample/sampler.py:138`` does ``sampled.view(-1, 1)``, ``compute_topk_logprobs``
does ``sampled_token_ids.unsqueeze(-1)``) rely on the 1-D shape, so the wrapper
flattens the native output back to 1-D to keep that contract.
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
    torch.ops.xspeedgate_ops.apply_temperature(
        logits, expanded_idx_mapping, temperature
    )


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
    )
    # Native op returns [n, 1]; upstream Triton returns 1-D [n]. Match the 1-D
    # contract the samplers depend on.
    return sampled.reshape(-1)


# ``SamplingStates`` binds ``apply_temperature`` at sample/states.py:9 and the
# samplers bind ``gumbel_sample`` at sample/sampler.py:18 and
# spec_decode/speculator.py:26, so these must be installed upstream before those
# modules execute; the post-import dispatcher guarantees the ordering.
_up.apply_temperature = apply_temperature
_up.gumbel_sample = gumbel_sample
logger.info("[KunlunPlugin] V2 gumbel patched (xspeedgate_ops native)")
