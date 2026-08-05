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
from typing import Optional, Union

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

logger = logging.getLogger("vllm_kunlun.ops.layernorm")


def rms_norm_kunlun(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    out = torch.empty_like(x)
    torch.ops._C.rmsnorm(x, weight, out, eps)
    return out


def fused_q_kv_rmsnorm_kunlun(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.shape[:-1] == kv.shape[:-1]
    assert q.shape[-1] == q_weight.shape[0]
    assert kv.shape[-1] == kv_weight.shape[0]
    return (
        rms_norm_kunlun(q, q_weight, eps),
        rms_norm_kunlun(kv, kv_weight, eps),
    )


def fused_add_rms_norm_kunlun(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.ops._C.add_rmsnorm(
        x,
        residual,
        residual_output=residual,
        weight=weight,
        eps=eps,
        output=x,
    )
    return x, residual


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
        super().__init__(*args, **kwargs)

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
        super().__init__(*args, **kwargs)

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


logger.info("[KunlunOOT] Loaded Kunlun RMSNorm implementations")
