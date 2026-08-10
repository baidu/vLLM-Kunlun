# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kunlun XPU kernel patch for the current vLLM rejection sampler.

The sampler API, logits processors and logprob handling remain owned by the
vendored vLLM implementation.  This module only replaces CUDA Triton kernels
with Kunlun ops, keeping slow PyTorch loops solely for synthetic acceptance
mode, which the native kernels do not implement.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import kunlun_ops
import torch
import xspeedgate_ops  # noqa: F401  (register torch.ops.xspeedgate_ops)


class _KernelAdapter:
    _kunlun_patched = True

    def __init__(self, launcher: Callable[..., None]) -> None:
        self._launcher = launcher

    def __getitem__(self, _grid: Any) -> Callable[..., None]:
        return self._launcher


def _flatten_bonus_token_ids(
    bonus_token_ids: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Adapt Sampler's ``[B, 1]`` output to the XPU op's ``[B]`` ABI."""

    if bonus_token_ids.numel() != batch_size:
        raise ValueError(
            "bonus_token_ids must contain one token per request: "
            f"shape={tuple(bonus_token_ids.shape)}, batch_size={batch_size}"
        )
    if not bonus_token_ids.is_contiguous():
        bonus_token_ids = bonus_token_ids.contiguous()
    return bonus_token_ids.view(batch_size)


def _expand_kernel(
    output: torch.Tensor,
    values: torch.Tensor,
    cu_num_tokens: torch.Tensor,
    replace_from: int,
    replace_to: int,
    **_kwargs: Any,
) -> None:
    if output.numel() == 0:
        return
    if values.dtype in (torch.int32, torch.float32):
        kunlun_ops.expand_tokens(
            output, values, cu_num_tokens, replace_from, replace_to
        )
        return

    # Non-production dtype fallback.  Normal sampling metadata uses int32 or
    # float32 and takes the native path above.
    starts = torch.cat((cu_num_tokens.new_zeros(1), cu_num_tokens[:-1]))
    counts = cu_num_tokens - starts
    replacement = values.new_full((), replace_to)
    expanded_values = torch.where(values == replace_from, replacement, values)
    output.copy_(torch.repeat_interleave(expanded_values, counts))


def _request_bounds(cu_num_draft_tokens: torch.Tensor, req_idx: int) -> tuple[int, int]:
    start = 0 if req_idx == 0 else int(cu_num_draft_tokens[req_idx - 1].item())
    end = int(cu_num_draft_tokens[req_idx].item())
    return start, end


def _synthetic_greedy(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_argmax: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    is_greedy: torch.Tensor | None,
    uniform_probs: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor,
) -> None:
    for req_idx in range(cu_num_draft_tokens.shape[0]):
        if is_greedy is not None and not bool(is_greedy[req_idx].item()):
            continue
        start, end = _request_bounds(cu_num_draft_tokens, req_idx)
        rejected = False
        for pos, token_idx in enumerate(range(start, end)):
            if rejected:
                break
            accepted = bool(
                (
                    uniform_probs[token_idx]
                    < synthetic_conditional_rates[pos]
                ).item()
            )
            output_token_ids[req_idx, pos] = (
                draft_token_ids[token_idx]
                if accepted
                else target_argmax[token_idx]
            )
            rejected = not accepted
        if not rejected:
            output_token_ids[req_idx, end - start] = bonus_token_ids[req_idx]


def _greedy_kernel(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_argmax: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    is_greedy: torch.Tensor | None,
    max_spec_len: int,
    uniform_probs: torch.Tensor | None,
    synthetic_conditional_rates: torch.Tensor | None,
    *,
    SYNTHETIC_MODE: bool,
    **_kwargs: Any,
) -> None:
    bonus_token_ids = _flatten_bonus_token_ids(
        bonus_token_ids, cu_num_draft_tokens.shape[0]
    )
    if not SYNTHETIC_MODE:
        kunlun_ops.rejection_greedy_sample(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        return
    assert uniform_probs is not None
    assert synthetic_conditional_rates is not None
    _synthetic_greedy(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        uniform_probs,
        synthetic_conditional_rates,
    )


def _synthetic_random(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    recovered_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor,
    is_greedy: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor,
) -> None:
    for req_idx in range(cu_num_draft_tokens.shape[0]):
        if bool(is_greedy[req_idx].item()):
            continue
        start, end = _request_bounds(cu_num_draft_tokens, req_idx)
        rejected = False
        for pos, token_idx in enumerate(range(start, end)):
            if rejected:
                break
            accepted = bool(
                (
                    uniform_probs[token_idx]
                    < synthetic_conditional_rates[pos]
                ).item()
            )
            output_token_ids[req_idx, pos] = (
                draft_token_ids[token_idx]
                if accepted
                else recovered_token_ids[token_idx]
            )
            rejected = not accepted
        if not rejected:
            output_token_ids[req_idx, end - start] = bonus_token_ids[req_idx]


def _random_kernel(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    recovered_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor,
    is_greedy: torch.Tensor,
    max_spec_len: int,
    vocab_size: int,
    synthetic_conditional_rates: torch.Tensor | None,
    *,
    NO_DRAFT_PROBS: bool,
    SYNTHETIC_MODE: bool,
    **_kwargs: Any,
) -> None:
    bonus_token_ids = _flatten_bonus_token_ids(
        bonus_token_ids, cu_num_draft_tokens.shape[0]
    )
    if not SYNTHETIC_MODE:
        kunlun_ops.rejection_random_sample(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            no_draft_probs=NO_DRAFT_PROBS,
        )
        return
    assert synthetic_conditional_rates is not None
    _synthetic_random(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        synthetic_conditional_rates,
    )


def _sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    sampling_metadata: Any,
    device: torch.device,
) -> torch.Tensor:
    del max_spec_len
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size), dtype=torch.float32, device=device
    )
    torch.ops.xspeedgate_ops.inplace_exponential(q)
    for req_idx, generator in sampling_metadata.generators.items():
        if num_draft_tokens[req_idx] > 0:
            torch.ops.xspeedgate_ops.inplace_exponential(
                q[req_idx], generator=generator
            )

    recovered_token_ids = torch.empty_like(draft_token_ids)
    kunlun_ops.sample_recovered_tokens(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        no_draft_probs=draft_probs is None,
    )
    return recovered_token_ids


def patch_rejection_sampler(module: Any) -> None:
    """Apply Kunlun XPU kernel patches to the rejection sampler module."""
    module.expand_kernel = _KernelAdapter(_expand_kernel)
    module.rejection_greedy_sample_kernel = _KernelAdapter(_greedy_kernel)
    module.rejection_random_sample_kernel = _KernelAdapter(_random_kernel)
    module.sample_recovered_tokens = _sample_recovered_tokens
    module._kunlun_xpu_patched = True
