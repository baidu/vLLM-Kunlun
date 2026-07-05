# SPDX-License-Identifier: Apache-2.0
"""PyTorch launch shims for the Triton kernels used in
``vllm.v1.sample.rejection_sampler``.

The upstream rejection sampler unconditionally launches Triton/CUDA kernels
(``expand_kernel``, ``rejection_greedy_sample_kernel``,
``rejection_random_sample_kernel``, ``sample_recovered_tokens_kernel``).
On Kunlun XPU the CUDA runtime is unavailable, so every launch ends in
``AssertionError: libcuda.so cannot found!``.

This module provides shims that mimic the ``kernel[grid](args...)`` launch
protocol while running a PyTorch implementation instead. The behavior is kept
in sync with the upstream kernels at vLLM 0.15.1.
"""

from __future__ import annotations

from typing import Callable

import torch


class _KernelLauncher:
    """Shim that mimics a ``triton.JITFunction`` launcher."""

    def __init__(self, fn: Callable):
        self._fn = fn

    def __getitem__(self, grid):
        if not isinstance(grid, tuple):
            grid = (grid,)
        fn = self._fn

        def _launch(*args, **kwargs):
            return fn(grid, *args, **kwargs)

        return _launch


def _expand_impl(
    grid,
    output_ptr,
    input_ptr,
    cu_num_tokens_ptr,
    replace_from,
    replace_to,
    MAX_NUM_TOKENS=None,
    **_,
):
    batch_size = int(grid[0])
    if batch_size == 0:
        return
    cu = cu_num_tokens_ptr.tolist()
    inp = input_ptr.tolist()
    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu[req_idx - 1]
        end_idx = cu[req_idx]
        if end_idx <= start_idx:
            continue
        src_val = inp[req_idx]
        if src_val == replace_from:
            src_val = replace_to
        output_ptr[start_idx:end_idx] = src_val


def _greedy_impl(
    grid,
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    target_argmax_ptr,
    bonus_token_ids_ptr,
    is_greedy_ptr,
    max_spec_len,
    **_,
):
    batch_size = int(grid[0])
    if batch_size == 0:
        return
    cu = cu_num_draft_tokens_ptr.tolist()
    draft = draft_token_ids_ptr.tolist()
    argmax = target_argmax_ptr.tolist()
    bonus = bonus_token_ids_ptr.reshape(-1).tolist()
    if is_greedy_ptr is None:
        is_greedy_list = [True] * batch_size
    else:
        is_greedy_list = [bool(x) for x in is_greedy_ptr.tolist()]

    out_cpu = output_token_ids_ptr.cpu()
    out_list = out_cpu.tolist()

    for req_idx in range(batch_size):
        if not is_greedy_list[req_idx]:
            continue
        start_idx = 0 if req_idx == 0 else cu[req_idx - 1]
        end_idx = cu[req_idx]
        num_draft = end_idx - start_idx
        rejected = False
        for pos in range(num_draft):
            if rejected:
                break
            d = draft[start_idx + pos]
            a = argmax[start_idx + pos]
            out_list[req_idx][pos] = a
            if d != a:
                rejected = True
        if not rejected:
            out_list[req_idx][num_draft] = bonus[req_idx]

    output_token_ids_ptr.copy_(
        torch.tensor(
            out_list,
            dtype=output_token_ids_ptr.dtype,
            device=output_token_ids_ptr.device,
        )
    )


def _random_impl(
    grid,
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    bonus_token_ids_ptr,
    recovered_token_ids_ptr,
    uniform_probs_ptr,
    is_greedy_ptr,
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS=False,
    **_,
):
    batch_size = int(grid[0])
    if batch_size == 0:
        return
    cu = cu_num_draft_tokens_ptr.tolist()
    draft = draft_token_ids_ptr.tolist()
    bonus = bonus_token_ids_ptr.reshape(-1).tolist()
    recovered = recovered_token_ids_ptr.tolist()
    uniform = uniform_probs_ptr.tolist()
    is_greedy_list = [bool(x) for x in is_greedy_ptr.tolist()]

    out_cpu = output_token_ids_ptr.cpu()
    out_list = out_cpu.tolist()
    target_cpu = target_probs_ptr.detach().cpu()
    draft_probs_cpu = None if NO_DRAFT_PROBS else draft_probs_ptr.detach().cpu()

    for req_idx in range(batch_size):
        if is_greedy_list[req_idx]:
            continue
        start_idx = 0 if req_idx == 0 else cu[req_idx - 1]
        end_idx = cu[req_idx]
        num_draft = end_idx - start_idx
        rejected = False
        for pos in range(num_draft):
            if rejected:
                break
            tok = start_idx + pos
            d = draft[tok]
            if NO_DRAFT_PROBS:
                draft_prob = 1.0
            else:
                draft_prob = float(draft_probs_cpu[tok, d].item())
            target_prob = float(target_cpu[tok, d].item())
            u = float(uniform[tok])
            if draft_prob > 0 and target_prob / draft_prob >= u:
                token_id = d
            else:
                rejected = True
                token_id = recovered[tok]
            out_list[req_idx][pos] = token_id
        if not rejected:
            out_list[req_idx][num_draft] = bonus[req_idx]

    output_token_ids_ptr.copy_(
        torch.tensor(
            out_list,
            dtype=output_token_ids_ptr.dtype,
            device=output_token_ids_ptr.device,
        )
    )


def _sample_recovered_impl(
    grid,
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    q_ptr,
    vocab_size,
    PADDED_VOCAB_SIZE=None,
    NO_DRAFT_PROBS=False,
    **_,
):
    batch_size = int(grid[0])
    if batch_size == 0:
        return
    cu = cu_num_draft_tokens_ptr.tolist()

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu[req_idx - 1]
        end_idx = cu[req_idx]
        num_draft = end_idx - start_idx
        if num_draft <= 0:
            continue
        q = q_ptr[req_idx, :vocab_size]
        for pos in range(num_draft):
            tok = start_idx + pos
            if NO_DRAFT_PROBS:
                d = int(draft_token_ids_ptr[tok].item())
                prob = target_probs_ptr[tok, :vocab_size].clone()
                prob[d] = 0
            else:
                draft_prob = draft_probs_ptr[tok, :vocab_size]
                target_prob = target_probs_ptr[tok, :vocab_size]
                prob = torch.clamp(target_prob - draft_prob, min=0)
            recovered_id = int(torch.argmax(prob / q).item())
            output_token_ids_ptr[tok] = recovered_id
