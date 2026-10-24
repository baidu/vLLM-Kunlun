# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun overrides for temperature scaling and Gumbel sampling."""

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
    valid = expanded_idx_mapping >= 0
    mapping = torch.where(valid, expanded_idx_mapping, temperature.numel())
    extended_temperature = torch.cat((temperature, temperature.new_zeros(1)))
    extended_seed = torch.cat((seed, seed.new_zeros(1)))
    processed = logits.float()
    if apply_temperature:
        t = extended_temperature[mapping.long()]
        processed = processed / torch.where(t != 0, t, 1)[:, None]
    sampled = torch.ops.xspeedgate_ops.gumbel_sample(
        processed.contiguous(),
        mapping,
        extended_temperature,
        extended_seed,
        pos,
        False,
        None,
        None,
        use_fp64,
    ).squeeze(-1)
    if output_processed_logits is not None:
        vocab = processed.shape[1]
        cache = output_processed_logits
        columns = output_processed_logits_col
        starts = torch.zeros_like(expanded_idx_mapping)
        if columns is not None:
            starts = starts + columns.to(torch.int32) * vocab
        # MTP request padding occupies the tail; zero-length writes skip it.
        ends = torch.cumsum(valid.to(torch.int32) * vocab, dim=0, dtype=torch.int32)
        torch.ops.xspeedgate_ops.apply_write(
            cache.view(cache.shape[0], -1),
            expanded_idx_mapping.clamp_min(0),
            starts,
            processed.contiguous().view(-1),
            ends,
        )
    return sampled


# ``SamplingStates`` and the samplers bind these functions at import time, so
# install the native-op versions before those modules execute.
_up.apply_temperature = apply_temperature
_up.gumbel_sample = gumbel_sample
logger.info("[KunlunPlugin] V2 gumbel patched (xspeedgate_ops native)")
