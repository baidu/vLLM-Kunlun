# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacement for ``vllm.v1.worker.gpu.sample.min_p``.

``apply_min_p`` launches the Triton ``_min_p_kernel`` upstream. Kunlun XPU
cannot JIT-compile Triton kernels, so it is replaced by the xspeedgate_ops
equivalent, which skips ``min_p == 0`` rows and keeps ``x == threshold``
exactly like the kernel.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)

logger = logging.getLogger("vllm_kunlun")


def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    torch.ops.xspeedgate_ops.min_p_inplace(logits, expanded_idx_mapping, min_p)


apply_min_p._kunlun_patched = True
