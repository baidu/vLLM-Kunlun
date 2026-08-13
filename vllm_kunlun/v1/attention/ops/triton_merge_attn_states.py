# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def mask_empty_context(
    lse: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    context_start_loc: torch.Tensor,
) -> None:
    """Neutralize context chunks that cover no keys before merging.

    A prefill query whose context chunk is empty attended to no keys, so its
    partial attention is undefined: the backend leaves the output rows as
    uninitialized scratch (which may hold NaN/Inf) even when it reports an LSE
    of -inf. Sanitize both here so ``merge_attn_states`` can stay generic:
    force the LSE to -inf (zero softmax weight) and zero the undefined output
    rows (so a zero weight cannot combine with NaN/Inf). Emptiness is derived
    from the context offsets, not from the -inf LSE, so no merge kernel has to
    reason about undefined partials.

    Args:
        lse: Chunk log-sum-exp, shape [num_heads, num_tokens].
        output: Chunk attention output, shape [num_tokens, num_heads, ...].
        query_start_loc: Prefill query cumulative offsets, shape [num_reqs + 1].
        context_start_loc: Chunk context cumulative offsets,
            shape [num_reqs + 1]; an empty chunk has a zero-length span.
    """
    num_reqs = query_start_loc.shape[0] - 1
    if num_reqs <= 0:
        return

    # Identify requests with empty context chunks.
    # An empty chunk has context_start_loc[i] == context_start_loc[i+1].
    empty_req = context_start_loc[1:] == context_start_loc[:-1]  # [num_reqs]

    if not empty_req.any():
        return

    # Compute query lengths for each request.
    q_lens = query_start_loc[1:] - query_start_loc[:-1]  # [num_reqs]

    # Expand per-request empty mask to per-token mask.
    # token_empty[token_idx] is True iff token belongs to an empty request.
    token_empty = torch.repeat_interleave(empty_req, q_lens)  # [num_tokens]

    # Set LSE to -inf for empty tokens (all heads).
    lse[:, token_empty] = float("-inf")

    # Zero the output for empty tokens.
    output[token_empty] = 0
