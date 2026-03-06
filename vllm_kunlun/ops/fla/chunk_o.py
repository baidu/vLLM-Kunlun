# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501

from typing import Optional

import torch
from vllm.triton_utils import triton

from .index import prepare_chunk_indices
from .utils import FLA_GDN_FIX_BT


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,  # cumsum of log decay
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    _, T, _, _, _ = *q.shape, v.shape[-1]
    if FLA_GDN_FIX_BT:
        BT = 64
    else:
        BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)

    o = torch.ops.xspeedgate_ops.chunk_fwd_o(
        q, k, v, h, g, scale, cu_seqlens, chunk_indices, chunk_size
    )
    return o
