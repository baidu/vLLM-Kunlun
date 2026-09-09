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

# splitmix64 constants as signed two's-complement int64: torch rejects the
# unsigned literals, they are out of the int64 range.
_M1 = -7046029254386353131  # 0x9E3779B97F4A7C15
_M2 = -4658895280553007687  # 0xBF58476D1CE4E5B9
_M3 = -7723592293110705685  # 0x94D049BB133111EB

# Smallest positive value produced by Triton's fp32 ``tl.rand``; the same clamp
# bound as upstream's ``_TL_RAND_MIN``, so the extreme tail behaves alike.
_TINY = 4.6566127342e-10
_ONE_MINUS = 1.0 - _TINY
_TWO_POW_M24 = 1.0 / (1 << 24)

# Vocab tile width. Bounds the fp32 temporaries to [num_tokens, _BLOCK].
_BLOCK = 8192


def _lshr(x: torch.Tensor, n: int) -> torch.Tensor:
    """Logical right shift: torch's ``>>`` on signed int64 propagates the sign."""
    return (x >> n) & ((1 << (64 - n)) - 1)


def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    z = x + _M1
    z = (z ^ _lshr(z, 30)) * _M2
    z = (z ^ _lshr(z, 27)) * _M3
    return z ^ _lshr(z, 31)


def _uniform(counter: torch.Tensor) -> torch.Tensor:
    """Map an int64 counter to fp32 uniforms in (0, 1).

    Takes the top 24 bits, which is exactly the precision upstream's fp32
    ``tl.rand`` delivers, and keeps fp64 out of the picture entirely.
    """
    bits = _lshr(_splitmix64(counter), 40)
    return (bits.to(torch.float32) * _TWO_POW_M24).clamp_(_TINY, _ONE_MINUS)


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
    # Only the spec-decode speculators pass these
    # (gpu/spec_decode/speculator.py:282, gpu/spec_decode/dspark/speculator.py:141),
    # and KunlunPlatform reports speculative decoding as unsupported for Model
    # Runner V2. Lifting that gate requires implementing this branch first.
    assert output_processed_logits is None, (
        "gumbel_sample: output_processed_logits (spec-decode path) is not "
        "supported on the Kunlun V2 milestone-1 sampler."
    )
    assert output_processed_logits_col is None, (
        "gumbel_sample: output_processed_logits_col (spec-decode path) is not "
        "supported on the Kunlun V2 milestone-1 sampler."
    )
    # Upstream draws fp64 noise and reduces in fp64 when this is set. Rejected
    # by the same capability gate, via model_config.use_fp64_gumbel.
    assert not use_fp64, (
        "gumbel_sample: use_fp64 is not supported on the Kunlun V2 "
        "milestone-1 sampler."
    )

    num_tokens, vocab_size = logits.shape
    # Upstream masks its temperature and seed loads on req_state_idx >= 0; plain
    # torch indexing would wrap -1 around to the last request instead.
    valid = (expanded_idx_mapping >= 0).unsqueeze(1)
    idx = expanded_idx_mapping.to(torch.long).clamp(min=0)

    temp = temperature[idx].to(torch.float32).unsqueeze(1)  # [num_tokens, 1]
    # Invalid rows degrade to a plain argmax, as they do upstream.
    greedy = (temp == 0.0) | ~valid

    # Mixing seed and pos into a per-row counter base is what ties the stream to
    # the request instead of to the batch it is scheduled in.
    row = _splitmix64(seed[idx].to(torch.int64) ^ _splitmix64(pos.to(torch.int64)))
    row = row.unsqueeze(1)  # [num_tokens, 1]

    best_val = torch.full((num_tokens,), float("-inf"), device=logits.device)
    best_idx = torch.zeros(num_tokens, dtype=torch.int64, device=logits.device)

    for start in range(0, vocab_size, _BLOCK):
        stop = min(start + _BLOCK, vocab_size)
        col = torch.arange(start, stop, device=logits.device, dtype=torch.int64)

        lf = logits[:, start:stop].to(torch.float32)
        if apply_temperature:
            lf = lf / torch.where(greedy, torch.ones_like(temp), temp)

        u = _uniform(row ^ col)
        # Gumbel-max: argmax(logits + gumbel_noise). log1p(-u) draws the winning
        # tail from u -> 0, where fp32 resolves finely; the naive -log(-log(u))
        # puts that tail at u -> 1, where the spacing is ~2**-24, capping the
        # noise near 16.6 and quantising it coarsely. Upstream's numerics choice.
        noised = torch.where(greedy, lf, lf - torch.log(-torch.log1p(-u)))

        val, arg = noised.max(dim=-1)
        # Strict `>` keeps the earlier tile on ties, matching the argmax upstream
        # runs over its per-block maxima.
        better = val > best_val
        best_idx = torch.where(better, arg + start, best_idx)
        best_val = torch.where(better, val, best_val)

    return best_idx


# ``SamplingStates`` binds ``apply_temperature`` at sample/states.py:9 and the
# samplers bind ``gumbel_sample`` at sample/sampler.py:18 and
# spec_decode/speculator.py:26, so these must be installed upstream before those
# modules execute; the post-import dispatcher guarantees the ordering.
_up.apply_temperature = apply_temperature
_up.gumbel_sample = gumbel_sample
