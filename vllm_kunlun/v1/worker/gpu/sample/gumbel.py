# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun torch-native overrides for ``vllm.v1.worker.gpu.sample.gumbel``.

Replaces the two Triton Gumbel-max entry points. Greedy requests
(temperature == 0) reduce to a plain argmax and are bit-exact vs upstream.
Random requests use torch-native Gumbel-max noise, which is distributionally
equivalent but NOT bit-reproducible against the upstream Philox kernel;
per-request seed reproducibility is therefore not preserved in this milestone
(documented limitation -- a seeded ``kunlun_ops`` gumbel kernel is the planned
follow-up).

``gumbel_sample`` is only ever called on the sampled-position logits
(``num_tokens`` ~= number of requests), so the tensors here are small.

The Triton device functions ``tl_rand32`` / ``gumbel_block_argmax``, which
``spec_decode/rejection_sampler_utils.py:6`` imports at module scope, are left
in place: they are only referenced from inside ``@triton.jit`` kernel bodies,
which are never executed on Kunlun XPU, and decorating them costs nothing.
"""

import torch
import vllm.v1.worker.gpu.sample.gumbel as _up

# fp32-safe clamp bounds so log/log1p never see 0 or 1.
_TINY = 1.0e-20
_ONE_MINUS = 1.0 - 1.0e-7


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """In-place divide each row's logits by its request temperature.

    Rows whose temperature is 0 (greedy) or 1 are left unchanged, matching the
    upstream kernel's early-return.
    """
    idx = expanded_idx_mapping.to(torch.long)
    temp = temperature[idx].to(torch.float32)  # [num_tokens]
    # temp == 0 -> greedy (skip); use 1.0 as a no-op divisor.
    safe = torch.where(temp == 0.0, torch.ones_like(temp), temp)
    logits.div_(safe.unsqueeze(1).to(logits.dtype))


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
    assert output_processed_logits is None, (
        "gumbel_sample: output_processed_logits (spec-decode path) is not "
        "supported on the Kunlun V2 milestone-1 sampler."
    )
    idx = expanded_idx_mapping.to(torch.long)
    temp = temperature[idx].to(torch.float32).unsqueeze(1)  # [num_tokens, 1]

    lf = logits.to(torch.float32)
    if apply_temperature:
        safe = torch.where(temp == 0.0, torch.ones_like(temp), temp)
        lf = lf / safe

    # Gumbel-max: argmax(logits + gumbel_noise). Draw the winning tail from
    # u -> 0 via log1p(-u), matching the upstream numerics choice.
    u = torch.rand_like(lf).clamp_(_TINY, _ONE_MINUS)
    gumbel = -torch.log(-torch.log1p(-u))

    # Greedy rows (temp == 0): no noise, plain argmax.
    noised = torch.where(temp == 0.0, lf, lf + gumbel)
    return noised.argmax(dim=-1).to(torch.int64)


# ``SamplingStates`` binds ``apply_temperature`` at sample/states.py:9 and the
# samplers bind ``gumbel_sample`` at sample/sampler.py:18 and
# spec_decode/speculator.py:26, so these must be installed upstream before those
# modules execute; the post-import dispatcher guarantees the ordering.
_up.apply_temperature = apply_temperature
_up.gumbel_sample = gumbel_sample
