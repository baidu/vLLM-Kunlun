# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 for Kunlun XPU.

Vendored from upstream ``vllm.models.minimax_m3`` and adapted, because upstream
selects its variant by platform predicate and Kunlun -- device_type "cuda" with
is_cuda_alike/is_cuda/is_rocm/is_xpu/is_cpu all False and is_out_of_tree() True --
is handed the nvidia variant, which needs flashinfer, fmha_sm100 and triton kernels
that do not run here. Registered through ModelRegistry in
``vllm_kunlun/models/__init__.py``.

Upstream's entry point chose between nvidia/ and amd/; there is nothing left to
choose here.
"""

from .nvidia.model import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
from .nvidia.mtp import MiniMaxM3MTP

__all__ = [
    "MiniMaxM3MTP",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
]
