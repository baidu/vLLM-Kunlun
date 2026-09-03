#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# Author: Yue Jun
# Email: liwei157@baidu.com, tangshiwen@baidu.com
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

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul


@CustomOp.register_oot(name="SiluAndMul")
class KunlunSiluAndMul(SiluAndMul):
    """Kunlun-optimized SiluAndMul registered through vLLM's OOT mechanism."""

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(out, x)
        return out


def swiglu(x: torch.Tensor, limit: float | None = None) -> torch.Tensor:
    """SwiGLU over a concatenated ``[gate, up]`` tensor, with an optional limit.

    ``limit`` reproduces upstream ``SiluAndMulWithClamp``:

        gate = clamp(x[..., :d], max=limit)
        up   = clamp(x[..., d:], -limit, limit)
        out  = gate * sigmoid(gate) * up

    DeepSeek-V4 sets ``swiglu_limit=10.0``, and dropping it costs far more than
    the clamped element count suggests: the down projection turns a handful of
    saturated intermediates into a large error on the token that produced them.
    Measured on layer 26 against an fp64 CPU golden, four clamped elements out
    of ~74k moved that layer's output by rel_l2 0.6 -- all of it on the
    position-0 sink token, the only one whose activations reach the limit.

    Neither ``torch.ops._C.silu_and_mul`` nor any ``kunlun_ops`` SwiGLU variant
    takes a limit, so the clamped form is elementwise torch. It runs on a
    ``[tokens * topk, intermediate]`` tensor, negligible next to the two
    grouped GEMMs around it.
    """
    d = x.shape[-1] // 2
    if limit is None:
        out = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(out, x)
        return out
    gate = x[..., :d].clamp(max=limit)
    up = x[..., d:].clamp(-limit, limit)
    return gate * torch.sigmoid(gate) * up


