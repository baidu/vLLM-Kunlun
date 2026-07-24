# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun replacement for upstream unquantized Linear apply."""

import os

import torch

from vllm import envs
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

_FORCE_TORCH_UNQUANTIZED_LINEAR = os.environ.get(
    "VLLM_KUNLUN_FORCE_TORCH_UNQUANTIZED_LINEAR", "0"
) == "1"


def _get_original_apply():
    original = getattr(UnquantizedLinearMethod, "_kunlun_original_apply", None)
    if original is None:
        original = UnquantizedLinearMethod.apply
        UnquantizedLinearMethod._kunlun_original_apply = original
    return original


def apply_unquantized_weights(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply unquantized linear weights using the Kunlun matmul op when eligible."""
    original_apply = _get_original_apply()
    if (
        _FORCE_TORCH_UNQUANTIZED_LINEAR
        or envs.VLLM_BATCH_INVARIANT
        or not x.is_cuda
        or x.dim() != 2
        or layer.weight.dim() != 2
        or x.dtype != layer.weight.dtype
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return original_apply(self, layer, x, bias)

    return torch.ops._C.matmul(
        x=x,
        w=layer.weight,
        out_dtype=x.dtype,
        bias=bias.to(torch.float32).contiguous() if bias is not None else None,
    )


apply_unquantized_weights._kunlun_patched = True
_get_original_apply()
UnquantizedLinearMethod.apply = apply_unquantized_weights
