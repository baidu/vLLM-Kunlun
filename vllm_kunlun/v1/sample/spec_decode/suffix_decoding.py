# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun override for the Suffix Decoding proposer.

Suffix decoding proposes a *dynamic* number of draft tokens per request, so the
verify step gets a non-uniform query length. Two consequences on Kunlun XPU:

* ``gpu_model_runner._is_uniform_decode`` only dispatches the FULL cudagraph
  when every request contributes exactly ``1 + num_speculative_tokens`` tokens,
  so a variable draft length keeps falling back to the PIECEWISE path. Measured
  on Qwen3.6-35B-A3B / 1 request: ~45 ms per step with MTP (uniform, FULL graph)
  vs ~70-77 ms with suffix (non-uniform).
* The GDN spec kernels are written for that same padded layout (see
  ``gdn_attn.build_spec_pad_indices`` for the fallback that unpads it).

Padding every draft up to the speculation budget trades a couple of wasted
draft tokens for graph replay. The trade is favourable because a single-request
decode step is bandwidth-bound: verifying 3 tokens costs about as much as 2.
At high concurrency the padded tokens stop being free (128 requests x 3 tokens
instead of the actual draft count), but measured graph replay savings still
outweigh that cost, so padding is enabled by default.

The rejection sampler still validates every padded draft token against the
target model. A repeated filler can be accepted when it matches the target, so
padding is a Kunlun performance policy rather than a bit-identical transform.
Set ``VLLM_KUNLUN_SUFFIX_PAD_DRAFT=0`` when validating upstream variable-length
suffix semantics.

Enabled by default on Kunlun for every ``num_speculative_tokens``. Setting
``VLLM_KUNLUN_SUFFIX_PAD_DRAFT=0`` restores the upstream variable-length
behaviour, which is equally correct but measured **2-4x slower per step** on
Qwen3.6-35B-A3B (k=16: copy 19.3 ms/step padded vs 60.5 ms/step variable;
fresh 22 vs 90 ms/step) -- losing the FULL cudagraph costs far more than the
wasted draft tokens, at conc=1 and conc=8 alike.
"""

import os

from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

_orig_propose = SuffixDecodingProposer.propose


def _env_enabled(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "off")


def _pad_draft_enabled() -> bool:
    return _env_enabled("VLLM_KUNLUN_SUFFIX_PAD_DRAFT", "1")


def propose(self, input_batch, sampled_token_ids, slot_mappings=None):
    draft_token_ids = _orig_propose(
        self, input_batch, sampled_token_ids, slot_mappings
    )
    pad_enabled = _pad_draft_enabled()

    for i, sampled_ids in enumerate(sampled_token_ids):
        if not sampled_ids:
            # Partial prefill: upstream deliberately proposes nothing, and the
            # request must not get speculative tokens before it is decoding.
            continue
        num_tokens = int(input_batch.num_tokens_no_spec[i])
        # Same budget upstream uses for ``max_spec_tokens``: never speculate
        # past ``max_model_len``. Near the end of the context this is smaller
        # than ``num_speculative_tokens``, so the batch can still be
        # non-uniform -- the GDN varlen fallback stays necessary.
        budget = min(
            self.num_speculative_tokens, self.max_model_len - num_tokens - 1
        )
        if budget <= 0:
            continue
        draft = [int(t) for t in draft_token_ids[i]]
        if pad_enabled and len(draft) < budget:
            # Repeat the most recent token to keep the verify width uniform.
            # Rejection sampling still evaluates these fillers normally.
            filler = draft[-1] if draft else int(sampled_ids[-1])
            draft.extend([filler] * (budget - len(draft)))
        draft_token_ids[i] = draft
    return draft_token_ids


SuffixDecodingProposer.propose = propose
