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
from vllm.model_executor.layers.activation import SiluAndMul as _upstream_cls

logger = logging.getLogger("vllm_kunlun")


def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


# Idempotent monkey-patch: safe under fork() and re-import.
if not getattr(_upstream_cls, "_kunlun_silu_and_mul_patched", False):
    _upstream_cls.forward_native = _forward_native
    _upstream_cls._kunlun_silu_and_mul_patched = True
    logger.info("[KunlunPlugin] SiluAndMul patched in vllm_kunlun/ops/activation.py")


# Re-export so that ``from vllm_kunlun.ops.activation import SiluAndMul`` keeps
# working for model files written against the pre-patch layout (deepseek_v2,
# seed_oss, mimo_v2_flash). It is the upstream class itself -- patched above --
# not a separate implementation.
SiluAndMul = _upstream_cls
