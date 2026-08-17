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

import logging

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul

logger = logging.getLogger("vllm_kunlun")

_oot_silu_and_mul_init_logged = False


@CustomOp.register_oot(name="SiluAndMul")
class KunlunSiluAndMul(SiluAndMul):
    """Kunlun-optimized SiluAndMul registered through vLLM's OOT mechanism."""

    def __init__(self, *args, **kwargs):
        global _oot_silu_and_mul_init_logged
        super().__init__(*args, **kwargs)
        if not _oot_silu_and_mul_init_logged:
            logger.info(
                "[KunlunOOT] KunlunSiluAndMul.__init__ called (OOT instantiation)"
            )
            _oot_silu_and_mul_init_logged = True

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(out, x)
        return out


logger.info("[KunlunOOT] Registered KunlunSiluAndMul via CustomOp.register_oot")
