# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XSpeedGate NaN counting for MRV2 logits metrics."""

import torch
import vllm.v1.worker.gpu.metrics.logits as _up


def get_num_nans(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[0] == 0:
        return logits.new_empty((0,), dtype=torch.int32)
    return torch.ops.xspeedgate_ops.get_num_nans(logits)


_up.get_num_nans = get_num_nans
