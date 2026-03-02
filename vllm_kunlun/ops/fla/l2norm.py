# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import kunlun_ops
import torch


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    dtype = output_dtype if output_dtype is not None else x.dtype
    out = torch.empty_like(x, dtype=dtype)
    kunlun_ops.l2norm(x, out, eps)
    return out
