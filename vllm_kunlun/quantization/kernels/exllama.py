#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# Author: Li Wei
# Email: liwei157@baidu.com
# This file is a part of the vllm-kunlun project.
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

from typing import Optional

import torch
from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
from vllm.model_executor.kernels.linear.mixed_precision import MPLinearLayerConfig
from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
    ExllamaLinearKernel,
)
from vllm.platforms import PlatformEnum, current_platform


class KunlunExllamaLinearKernel(ExllamaLinearKernel):
    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_out_of_tree():
            return super().can_implement(c)

        if c.has_g_idx and c.partition_weight_shape[0] != c.full_weight_shape[0]:
            return (
                False,
                "Act reordering currently not supported by Exllama, "
                "when the input features are partitioned across devices",
            )

        if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:
            return (
                False,
                "Output features must be a multiple of the pack factor "
                "(32 / num_bits) so that we can correctly pack the zero points",
            )

        if c.act_type != torch.float16:
            return False, "Exllama only supports float16 activations"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by Exllama, "
                f"supported types are: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                f"Group size ({c.group_size}) must be positive, "
                "Exllama does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide the number "
                f"of input features ({c.full_weight_shape[0]})",
            )

        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp, w_g_idx = self._get_weight_params(layer)

        assert w_zp is not None, "Zero points are required by Exllama"
        assert w_g_idx is not None, "Group index is required by Exllama"
        output = torch.ops.xspeedgate_ops.gptq_gemm(
            x_2d, w_q, w_zp, w_s, w_g_idx, True, c.weight_type.size_bits
        )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)


_POSSIBLE_KERNELS.setdefault(PlatformEnum.OOT, [])
_POSSIBLE_KERNELS[PlatformEnum.OOT] = [
    kernel
    for kernel in _POSSIBLE_KERNELS[PlatformEnum.OOT]
    if kernel is not ExllamaLinearKernel
]
if KunlunExllamaLinearKernel not in _POSSIBLE_KERNELS[PlatformEnum.OOT]:
    _POSSIBLE_KERNELS[PlatformEnum.OOT].append(KunlunExllamaLinearKernel)
