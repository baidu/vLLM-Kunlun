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
import xspeedgate_ops  # noqa: F401  # register torch.ops.xspeedgate_ops.*
from vllm.model_executor.layers.activation import SiluAndMul as _upstream_silu_cls
from vllm.model_executor.layers.activation import SituAndMul as _upstream_situ_cls

logger = logging.getLogger("vllm_kunlun")

def _silu_forward_native(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


def _situ_forward_native(self, x: torch.Tensor) -> torch.Tensor:
    # xspeedgate_ops::situ_and_mul(input, beta, linear_beta) -> Tensor
    # linear_beta <= 0 tells the kernel to pass up through unchanged.
    linear_beta = -1.0 if self.linear_beta is None else float(self.linear_beta)
    return torch.ops.xspeedgate_ops.situ_and_mul(x, float(self.beta), linear_beta)


# Idempotent monkey-patch: safe under fork() and re-import.
if not getattr(_upstream_silu_cls, "_kunlun_silu_and_mul_patched", False):
    _upstream_silu_cls.forward_native = _silu_forward_native
    _upstream_silu_cls._kunlun_silu_and_mul_patched = True
    logger.info("[KunlunPlugin] SiluAndMul.forward_native patched")

if not getattr(_upstream_situ_cls, "_kunlun_situ_and_mul_patched", False):
    _upstream_situ_cls.forward_native = _situ_forward_native
    _upstream_situ_cls._kunlun_situ_and_mul_patched = True
    logger.info("[KunlunPlugin] SituAndMul.forward_native patched")


# Re-export so `from vllm_kunlun.ops.activation import SiluAndMul/SituAndMul`
# works and pulls in the patch as a side effect.
SiluAndMul = _upstream_silu_cls
SituAndMul = _upstream_situ_cls