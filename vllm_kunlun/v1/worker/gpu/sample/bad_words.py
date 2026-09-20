# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op override for ``vllm.v1.worker.gpu.sample.bad_words``.

Reimplements ``apply_bad_words`` on the Kunlun native op
``torch.ops.xspeedgate_ops.bad_words``. Unlike the previous torch-native
stand-in, the native op implements the spec-decode path (reading candidate
tokens from ``input_ids`` when ``expanded_local_pos`` > 0), so it is correct for
both the milestone-1 non-spec path and the spec-decode path.
"""

import logging

import torch
import vllm.v1.worker.gpu.sample.bad_words as _up

logger = logging.getLogger("vllm_kunlun")


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
    torch.ops.xspeedgate_ops.bad_words(
        logits,
        expanded_idx_mapping,
        bad_word_token_ids,
        bad_word_offsets,
        num_bad_words,
        all_token_ids,
        prompt_len,
        total_len,
        input_ids,
        expanded_local_pos,
        max_num_bad_words,
    )


# ``BadWordsState`` resolves ``apply_bad_words`` from the upstream module's
# globals; install the native-op version there.
_up.apply_bad_words = apply_bad_words
logger.info("[KunlunPlugin] V2 bad_words patched (xspeedgate_ops native)")
