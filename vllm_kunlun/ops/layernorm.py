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
# This file is a part of the vllm-ascend project.
#

import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as OriGemmaRMSNorm
from vllm.model_executor.layers import layernorm
from typing import Optional, Union
import kunlun_ops

def vllm_kunlun_forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """forward_cuda"""
        if x.is_contiguous() == False:
            # kunlun does not support uncontiguous input and they do not think it is a bug 
            # so we must make it contiguous() manually
            x = x.contiguous()
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        
        if residual is not None:
            # residual_output = torch.empty_like(residual)
            torch.ops._C.add_rmsnorm(
                x,
                residual,
                residual_output=residual,
                weight=self.weight.data,
                eps=self.variance_epsilon,
                output=x
            )
            return x, residual
        out = torch.empty_like(x)
        torch.ops._C.rmsnorm(
            x,
            self.weight.data,
            out,
            self.variance_epsilon,
        )
        return out

RMSNorm.forward_cuda = vllm_kunlun_forward_cuda
RMSNorm.forward = vllm_kunlun_forward_cuda

class KunlunGemmaRMSNorm(OriGemmaRMSNorm):
    @staticmethod
    def forward_xpu(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.is_contiguous() == False:
            # kunlun does not support uncontiguous input and they do not think it is a bug
            # so we must make it contiguous() manually
            x = x.contiguous()

        if residual is not None:
            torch.ops._C.add_rmsnorm(
                x,
                residual,
                residual_output=residual,
                weight=weight+1,
                eps=variance_epsilon,
                output=x
            )
            return x, residual

        out = torch.empty_like(x)
        torch.ops._C.rmsnorm(
            x,
            weight+1,
            out,
            variance_epsilon,
        )
        return out

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if torch.compiler.is_compiling():
            self.forward_static = self.forward_xpu # only use in cudagraph
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self.forward_static = torch.compile(  # type: ignore
                self.forward_static, backend="aot_eager")
            self._is_compiled = True
        return self.forward_native(x, residual)


RMSNorm.forward_cuda = vllm_kunlun_forward_cuda
RMSNorm.forward = vllm_kunlun_forward_cuda
layernorm.GemmaRMSNorm = KunlunGemmaRMSNorm