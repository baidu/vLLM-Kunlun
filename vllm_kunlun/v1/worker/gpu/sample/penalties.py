# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun torch-native overrides for ``vllm.v1.worker.gpu.sample.penalties``.

Leaves the upstream ``PenaltiesState`` / ``use_penalty`` alone and reimplements
the two Triton functions ``bincount`` (prompt/output token statistics, built
once per new penalty request) and ``apply_penalties`` (per-step repetition /
frequency / presence penalties).

NOTE: the spec-decode draft-token accumulation (the ``expanded_local_pos``
loop in the upstream kernel) is omitted; in the milestone-1 non-spec path
``expanded_local_pos`` is all zeros, so ``output_bin_counts`` as maintained by
``post_update`` is already correct.
"""

import torch
import vllm.v1.worker.gpu.sample.penalties as _up


def bincount(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int,
) -> None:
    idx_long = expanded_idx_mapping.long()
    # Reset stats for the affected requests (index_fill_ avoids a host sync).
    prompt_bin_mask.index_fill_(0, idx_long, 0)
    output_bin_counts.index_fill_(0, idx_long, 0)

    device = prompt_bin_mask.device
    vocab_size = output_bin_counts.shape[1]
    packed_cols = prompt_bin_mask.shape[1]
    padded = packed_cols * 32
    shifts = torch.arange(32, device=device, dtype=torch.int64)

    reqs = idx_long.tolist()
    pls = prompt_len.tolist()
    pfl = prefill_len.tolist()
    for rs in reqs:
        pl = pls[rs]
        pf = pfl[rs]
        tokens = all_token_ids[rs, :pf].to(torch.long)
        if pl > 0:
            # Pack prompt-token presence into 32-bit words (word=tok//32,
            # bit=tok%32) to match the layout apply_penalties unpacks.
            present = torch.zeros(padded, dtype=torch.int64, device=device)
            present[tokens[:pl]] = 1
            words = (present.view(packed_cols, 32) << shifts).sum(dim=1)
            prompt_bin_mask[rs] = words.to(torch.int32)
        if pf > pl:
            counts = torch.bincount(tokens[pl:pf], minlength=vocab_size)[:vocab_size]
            output_bin_counts[rs] = counts.to(torch.int32)


def apply_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    device = logits.device
    idx = expanded_idx_mapping.to(torch.long)  # [num_tokens]
    rep = repetition_penalty[idx].to(torch.float32)  # [num_tokens]
    freq = frequency_penalty[idx].to(torch.float32)
    pres = presence_penalty[idx].to(torch.float32)

    lf = logits.to(torch.float32)

    # Per-position output token counts (spec-decode draft accumulation omitted).
    out_counts = output_bin_counts[idx].to(torch.float32)  # [num_tokens, vocab]
    output_mask = out_counts > 0

    # Unpack prompt-token bitmask to a [num_tokens, vocab] boolean.
    packed = prompt_bin_mask[idx].to(torch.int64) & 0xFFFFFFFF  # [num_tokens, packed]
    shifts = torch.arange(32, device=device, dtype=torch.int64)
    bits = (packed.unsqueeze(-1) >> shifts) & 1  # [num_tokens, packed, 32]
    prompt_mask = bits.reshape(num_tokens, -1)[:, :vocab_size].to(torch.bool)

    # Repetition penalty: divide positive logits, multiply negative ones.
    ones = torch.ones_like(rep).unsqueeze(1)
    scale = torch.where(prompt_mask | output_mask, rep.unsqueeze(1), ones)
    lf = lf * torch.where(lf > 0, 1.0 / scale, scale)

    # Frequency and presence penalties.
    lf = lf - freq.unsqueeze(1) * out_counts
    lf = lf - pres.unsqueeze(1) * output_mask.to(torch.float32)

    logits.copy_(lf.to(logits.dtype))


# ``PenaltiesState`` resolves ``apply_penalties`` / ``bincount`` from the
# upstream module's globals, so the torch-native versions are installed there.
_up.apply_penalties = apply_penalties
_up.bincount = bincount
