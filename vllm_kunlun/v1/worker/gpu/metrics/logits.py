# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacement for ``vllm.v1.worker.gpu.metrics.logits``.

``get_num_nans`` launches the Triton ``_num_nans_kernel`` upstream. Kunlun XPU
cannot JIT-compile Triton kernels, so it is replaced by the xspeedgate_ops
equivalent, which infers ``num_reqs``/``vocab_size`` from the tensor and does
the whole reduction on-device in one launch.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)

logger = logging.getLogger("vllm_kunlun")


def get_num_nans(logits: torch.Tensor) -> torch.Tensor:
    return torch.ops.xspeedgate_ops.get_num_nans(logits)


get_num_nans._kunlun_patched = True
