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
import xspeedgate_ops
from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    ExllamaLinearKernel,
    _POSSIBLE_KERNELS,
)


class KunlunExllamaLinearKernel(ExllamaLinearKernel):

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


# remove ExllamaLinearKernel and add KunlunExllamaLinearKernel
_POSSIBLE_KERNELS.remove(ExllamaLinearKernel)
_POSSIBLE_KERNELS.append(KunlunExllamaLinearKernel)
