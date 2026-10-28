# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun torch-native override for ``vllm.v1.worker.gpu.sample.bad_words``.

Leaves the upstream ``BadWordsState`` / constants alone and reimplements
``apply_bad_words``. For each sampled position it checks, per bad-word, whether
the trailing generated tokens match the bad-word prefix and, if so, bans the
final token by setting its logit to -inf.

NOTE: the spec-decode path (reading candidate tokens from ``input_ids`` when
``expanded_local_pos`` > 0) is not implemented; milestone-1 has
``expanded_local_pos == 0`` so all comparison tokens come from the committed
output sequence.
"""

import torch
import vllm.v1.worker.gpu.sample.bad_words as _up

_NEG_INF = float("-inf")


def apply_bad_words(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    bad_word_token_ids: torch.Tensor,
    bad_word_offsets: torch.Tensor,
    num_bad_words: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    max_num_bad_words: int,
) -> None:
    num_tokens = logits.shape[0]
    idx = expanded_idx_mapping.tolist()
    nbw_list = num_bad_words.tolist()
    pl_list = prompt_len.tolist()
    tl_list = total_len.tolist()
    pos_list = expanded_local_pos.tolist()

    for t in range(num_tokens):
        rs = idx[t]
        nbw = nbw_list[rs]
        if nbw == 0:
            continue
        prompt = pl_list[rs]
        output_len = tl_list[rs] - prompt
        effective_len = output_len + pos_list[t]
        offsets_row = bad_word_offsets[rs, : nbw + 1].tolist()
        for bw in range(nbw):
            start = offsets_row[bw]
            end = offsets_row[bw + 1]
            prefix_len = end - start - 1
            if prefix_len > effective_len:
                continue
            last_token = bad_word_token_ids[rs, end - 1 : end].to(torch.long)
            if prefix_len == 0:
                logits[t, last_token] = _NEG_INF
                continue
            expected = bad_word_token_ids[rs, start : end - 1].to(torch.long)
            a_start = prompt + (effective_len - prefix_len)
            actual = all_token_ids[rs, a_start : a_start + prefix_len].to(torch.long)
            if bool((expected == actual).all()):
                logits[t, last_token] = _NEG_INF


# ``BadWordsState`` resolves ``apply_bad_words`` from the upstream module's
# globals; install the torch-native version there.
_up.apply_bad_words = apply_bad_words
