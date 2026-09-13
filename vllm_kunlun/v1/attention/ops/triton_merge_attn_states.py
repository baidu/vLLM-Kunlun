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
    torch.ops.xspeedgate_ops.mask_empty_context(lse, output, query_start_loc, context_start_loc)
