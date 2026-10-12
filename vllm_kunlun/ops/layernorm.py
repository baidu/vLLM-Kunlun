#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-kunlun project.
#
"""
Kunlun-optimized LayerNorm implementations using vLLM's CustomOp.register_oot mechanism.

Design:
- Uses @CustomOp.register_oot to register Kunlun-optimized RMSNorm/GemmaRMSNorm
- These classes automatically replace the default implementations when instantiated
- Since KunlunPlatform uses _enum=PlatformEnum.OOT, dispatch_forward() selects
  forward_oot, so we implement forward_oot

OOT Mechanism:
- When code calls RMSNorm(...), vLLM's CustomOp.__new__ checks op_registry_oot
- If "RMSNorm" is found in OOT registry, it returns KunlunRMSNorm instance instead
- This is the official vLLM way to replace operators without modifying source code
"""

import logging
import weakref
from typing import Optional, Union

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

logger = logging.getLogger("vllm_kunlun.ops.layernorm")

# Track if OOT classes have logged (for logging once per type)
_oot_rms_norm_init_logged = False
_oot_gemma_rms_norm_init_logged = False


# =============================================================================
# OOT-registered Kunlun LayerNorm classes
# =============================================================================


@CustomOp.register_oot(name="RMSNorm")
class KunlunRMSNorm(RMSNorm):
    """
    Kunlun-optimized RMSNorm registered via OOT mechanism.

    This class replaces the default RMSNorm when instantiated through
    vLLM's CustomOp registry. When code calls RMSNorm(...), vLLM's
    CustomOp.__new__ checks op_registry_oot and returns KunlunRMSNorm instance.

    Since KunlunPlatform uses _enum=PlatformEnum.OOT, dispatch_forward()
    selects forward_oot for execution.
    """

    def __init__(self, *args, **kwargs):
        global _oot_rms_norm_init_logged
        super().__init__(*args, **kwargs)
        if not _oot_rms_norm_init_logged:
            logger.info("[KunlunOOT] KunlunRMSNorm.__init__ called (OOT instantiation)")
            _oot_rms_norm_init_logged = True

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Kunlun-optimized forward_oot using Kunlun RMSNorm kernels."""
        # Kunlun does not support non-contiguous input
        if not x.is_contiguous():
            x = x.contiguous()

        # Fallback to native implementation for variance_size_override
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        if residual is not None:
            # Fused add + RMSNorm: output = RMSNorm(x + residual)
            torch.ops._C.add_rmsnorm(
                x,
                residual,
                residual_output=residual,
                weight=self.weight.data,
                eps=self.variance_epsilon,
                output=x,
            )
            return x, residual

        # Standard RMSNorm
        out = torch.empty_like(x)
        torch.ops._C.rmsnorm(
            x,
            self.weight.data,
            out,
            self.variance_epsilon,
        )
        return out


@CustomOp.register_oot(name="GemmaRMSNorm")
class KunlunGemmaRMSNorm(GemmaRMSNorm):
    """
    Kunlun-optimized GemmaRMSNorm registered via OOT mechanism.

    Similar to KunlunRMSNorm, but implements Gemma's (1 + weight) convention.
    """

    def __init__(self, *args, **kwargs):
        global _oot_gemma_rms_norm_init_logged
        super().__init__(*args, **kwargs)
        if not _oot_gemma_rms_norm_init_logged:
            logger.info(
                "[KunlunOOT] KunlunGemmaRMSNorm.__init__ called (OOT instantiation)"
            )
            _oot_gemma_rms_norm_init_logged = True

    @staticmethod
    def forward_xpu(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            # kunlun does not support uncontiguous input and they do not think it is a bug
            # so we must make it contiguous() manually
            x = x.contiguous()
        if x.dim() == 3:
            x_shape = x.shape
            x = x.view(-1, x.size(-1))
        if residual is not None:
            out = torch.empty_like(x)
            out_residual = torch.empty_like(residual)
            torch.ops._C.gemma_add_rmsnorm(
                x,
                residual,
                residual_output=out_residual,
                weight=weight,
                eps=variance_epsilon,
                output=out,
            )
        else:
            out = torch.empty_like(x)
            torch.ops._C.gemma_rmsnorm(
                x,
                weight,
                out,
                variance_epsilon,
            )

        if x.dim() == 3:
            x = x.view(x_shape)
            if out is not None:
                out = out.view(x_shape)

        if residual is not None:
            return out, out_residual
        else:
            return out

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # """Kunlun-optimized forward_oot for Gemma models."""
        return self.forward_xpu(self.weight.data, self.variance_epsilon, x, residual)


# Log that OOT registration is complete
# NOTE: OOT mechanism uses cls.__name__ (e.g. "RMSNorm") not the op's logical name
# Use print to stderr to ensure visibility in logs
logger.info(
    "[KunlunOOT] Registered KunlunRMSNorm and KunlunGemmaRMSNorm via CustomOp.register_oot"
)


# Each entry keeps only a weak reference to the source parameter. This prevents
# stale hits if a model is unloaded and a later tensor reuses the same address.
_WEIGHT_FP32_CACHE: dict[
    tuple[str, int | None, int],
    tuple[weakref.ReferenceType[torch.Tensor], int, torch.Tensor],
] = {}


def _cached_fp32_weight(weight: torch.Tensor) -> torch.Tensor:
    key = (weight.device.type, weight.device.index, weight.data_ptr())
    version = int(weight._version)
    entry = _WEIGHT_FP32_CACHE.get(key)
    if entry is not None and entry[0]() is weight and entry[1] == version:
        return entry[2]

    converted = weight.float().contiguous()

    def evict(dead_ref, *, cache_key=key):
        cached = _WEIGHT_FP32_CACHE.get(cache_key)
        if cached is not None and cached[0] is dead_ref:
            _WEIGHT_FP32_CACHE.pop(cache_key, None)

    source_ref = weakref.ref(weight, evict)
    _WEIGHT_FP32_CACHE[key] = (source_ref, version, converted)
    return converted


def rms_norm(out, input, weight, epsilon):
    """Adapt the upstream RMSNorm tensor contract to Kunlun kernels."""
    # The DSpark hidden-state norm is a conventional 2-D RMSNorm.
    if (
        input.ndim == 2
        and weight.ndim == 1
        and input.is_contiguous()
        and out.is_contiguous()
        and weight.is_contiguous()
    ):
        import kunlun_ops

        kunlun_ops.rmsnorm(input, _cached_fp32_weight(weight), out, epsilon)
        return

    # DFlash context K uses one independent RMSNorm weight per layer.
    if (
        input.ndim >= 3
        and weight.ndim == 2
        and input.shape[0] == weight.shape[0]
        and input.shape[-1] == weight.shape[-1]
        and input.is_contiguous()
        and out.is_contiguous()
        and weight.is_contiguous()
    ):
        rows_per_layer = input.numel() // (input.shape[0] * input.shape[-1])
        implementation = 1 if rows_per_layer <= 512 else 0
        torch.ops.xspeedgate_ops.dflash_grouped_rmsnorm(
            out, input, weight, epsilon, implementation
        )
        return

    # Reference fallback for shapes outside the optimized DFlash contract.
    x = input.float()
    variance = x.square().mean(dim=-1, keepdim=True)
    w = weight.float()
    if w.ndim == 1:
        w = w.view(*([1] * (input.ndim - 1)), w.shape[0])
    elif w.ndim == 2 and input.ndim >= 3:
        w = w.view(w.shape[0], *([1] * (input.ndim - 3)), 1, w.shape[1])
    out.copy_((x * torch.rsqrt(variance + epsilon) * w).to(dtype=out.dtype))
