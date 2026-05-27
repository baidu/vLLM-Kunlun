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
"""Kunlun-optimized SiluAndMul/GeluAndMul via CustomOp.register_oot."""

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul, GeluAndMul


@CustomOp.register_oot(name="SiluAndMul")
class KunlunSiluAndMul(SiluAndMul):
    """Kunlun-optimized SiluAndMul using XPU kernel."""

    def __init__(self, *, compile_native: bool = True):
        CustomOp.__init__(self, compile_native=compile_native)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(out, x)
        return out


@CustomOp.register_oot(name="GeluAndMul")
class KunlunGeluAndMul(GeluAndMul):
    """Kunlun-optimized GeluAndMul using XPU kernel."""

    def __init__(self, approximate: str = "none"):
        CustomOp.__init__(self)
        self.approximate = approximate

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        if self.approximate == "tanh":
            torch.ops._C.gelu_tanh_and_mul(out, x)
        else:
            torch.ops._C.gelu_and_mul(out, x)
        return out
