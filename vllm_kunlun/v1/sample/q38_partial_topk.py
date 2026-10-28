# SPDX-License-Identifier: Apache-2.0
"""Experimental partial-sort mask, preserving the full-vocabulary reductions.

Not a model-graph operation: admission performs one device-to-host scalar read.
Call only from the graph-external rejection sampler. Disabled by the caller by
default. A rejected batch must use the untouched original logits and native path.

Unlike a top-20/small-softmax approximation, this keeps all kth-threshold ties,
the original vocabulary length and suffix positions for softmax and cumsum.
If top-p splits an equal-value group, reject rather than assume sort stability.
"""

import functools
import json
import os

import torch


def try_partial_mask(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    *,
    capacity: int = 128,
) -> torch.Tensor | None:
    """Build a candidate mask without mutating the input.

    Args:
        logits: Contiguous float32 batch-by-vocabulary logits.
        k: Per-row top-k counts on the logits device.
        p: Per-row float32 top-p thresholds on the logits device.
        capacity: Maximum sorted candidate count, strictly larger than each k.

    Returns:
        Masked logits, or None when equivalence to the native mask is unproven.
    """
    if (
        k is None
        or p is None
        or logits.ndim != 2
        or logits.dtype != torch.float32
        or not logits.is_contiguous()
        or logits.shape[0] == 0
        or logits.shape[1] <= capacity
        or k.ndim != 1
        or p.ndim != 1
        or k.shape[0] != logits.shape[0]
        or p.shape[0] != logits.shape[0]
        or k.device != logits.device
        or p.device != logits.device
        or k.dtype not in (torch.int32, torch.int64)
        or p.dtype != torch.float32
        or capacity < 2
    ):
        return None

    # Clamping only protects the speculative gather; invalid k rejects the batch.
    safe_k = k.to(torch.long).clamp(1, capacity)
    descending, ids = logits.topk(capacity, dim=-1, largest=True, sorted=True)
    threshold = descending.gather(1, (safe_k - 1).unsqueeze(1))
    # Strict inequality certifies that no omitted logit can meet kth threshold.
    bad = (descending[:, -1:] >= threshold).any()
    bad = bad | (~torch.isfinite(logits)).any()
    bad = bad | ((k < 1) | (k >= capacity)).any()
    bad = bad | ((p < 0) | (p > 1) | ~torch.isfinite(p)).any()

    values = descending.flip(-1)
    indices = ids.flip(-1)
    values = values.masked_fill(values < threshold, -float("inf"))
    # Match the complete native sorted, top-k-masked vector, including zeros'
    # positions in the softmax reduction and cumsum. Never shorten the reduction.
    sorted_full = torch.full_like(logits, -float("inf"))
    sorted_full[:, -capacity:] = values
    probs = sorted_full.softmax(dim=-1)
    sums = torch.cumsum(probs, dim=-1, out=probs)
    # Prefix mask entries are never consumed. Keep the full-V arithmetic above,
    # but compare only the suffix that is used by the tie guard and scatter.
    tail_mask = sums[:, -capacity:] <= 1 - p.unsqueeze(1)
    tail_mask[:, -1] = False
    split_tie = (
        (values[:, 1:] == values[:, :-1])
        & (tail_mask[:, 1:] != tail_mask[:, :-1])
        & torch.isfinite(values[:, 1:])
    )
    bad = bad | split_tie.any()
    if bad.item():
        return None
    values = values.masked_fill(tail_mask, -float("inf"))
    output = torch.full_like(logits, -float("inf"))
    output.scatter_(dim=-1, index=indices, src=values)
    return output


def wrap_rejection_mask(native):
    """Wrap only the graph-external rejection mask, with optional shadow checks."""
    mode = os.environ.get("Q38_PARTIAL_TOPK", "off")
    if mode == "off":
        return native
    if mode not in ("shadow", "on"):
        raise ValueError(("invalid Q38_PARTIAL_TOPK", mode))
    counts = {"calls": 0, "fast": 0, "fallback": 0, "exact": 0}

    @functools.wraps(native)
    def apply(logits, k, p):
        candidate = try_partial_mask(logits, k, p)
        counts["calls"] += 1
        if candidate is None:
            counts["fallback"] += 1
            output = native(logits, k, p)
        else:
            counts["fast"] += 1
            output = candidate
            if mode == "shadow":
                reference = native(logits, k, p)
                # Compare integer views to detect signed-zero differences too.
                mask_exact = torch.equal(
                    reference.view(torch.int32), candidate.view(torch.int32)
                )
                probs_exact = torch.equal(
                    reference.softmax(-1).view(torch.int32),
                    candidate.softmax(-1).view(torch.int32),
                )
                if not mask_exact or not probs_exact:
                    raise RuntimeError(
                        (
                            "partial-sort shadow mismatch",
                            logits.shape,
                            counts,
                            mask_exact,
                            probs_exact,
                        )
                    )
                counts["exact"] += 1
                # Shadow mode always returns the baseline value and consumes no RNG.
                output = reference
        if counts["calls"] == 1 or counts["calls"] % 100 == 0:
            print(
                "[Q38-PARTIAL-TOPK]",
                json.dumps(
                    {
                        "mode": mode,
                        "pid": os.getpid(),
                        "shape": list(logits.shape),
                        **counts,
                    }
                ),
                flush=True,
            )
        return output

    return apply
