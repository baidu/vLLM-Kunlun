# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch implementations of MiniMax M3 sparse-attention kernels, for Kunlun XPU."""

from .index_topk import (
    minimax_m3_index_decode,
    minimax_m3_index_decode_score,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from .sparse_attn import minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode

__all__ = [
    "minimax_m3_index_decode",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
]
