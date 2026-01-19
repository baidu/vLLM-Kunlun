#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# Author: Liwei, Tang Shiwen
# Email: liwei157@baidu.com, tangshiwen@baidu.com
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
from vllm.platforms import current_platform, PlatformEnum
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    _POSSIBLE_KERNELS,
    ScaledMMLinearLayerConfig,
    CutlassScaledMMLinearKernel,
)


class KunlunScaledMMLinearKernel(CutlassScaledMMLinearKernel):

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, Optional[str]]:

        if not current_platform.is_kunlun():
            return False, "KunlunScaledMM requires running on XPU."

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # change scale to max for klx ops
        with torch.no_grad():
            getattr(layer, self.w_s_name).mul_(127.0)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w_q, w_s, x_s, x_zp, azp_adj = self._get_weight_params(layer)
        symmetric = azp_adj is None

        # scaled_int8_quant supports both dynamic and static quant
        # Currently, static is per-tensor and dynamic is per-token
        x_q, x_s, x_zp, static = torch.ops._C.scaled_int8_quant(
            x=x.contiguous(),
            scale=x_s,
            azp=x_zp,
            symmetric=symmetric,
        )

        if x_zp is not None:  # asymmetric
            azp = None if static else x_zp
            return torch.ops._C.cutlass_scaled_mm_azp(
                a=x_q,
                b=w_q,
                scale_a=x_s,
                scale_b=(w_s / 127.0).transpose(0, 1),
                out_dtype=x.dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias.to(torch.float32).contiguous() if bias else None,
            )
        else:  # symmetric
            return torch.ops._C.matmul(
                x=x_q,
                w=w_q.transpose(0, 1),
                out_dtype=x.dtype,
                x_pc_max=x_s * 127.0 if static else x_s,
                w_pc_max=w_s,
                bias=bias.to(torch.float32).contiguous() if bias else None,
            )

            # backup option: lower performance
            # return torch.ops._C.cutlass_scaled_mm(
            #     a = x_q,
            #     b = w_q,
            #     scale_a=x_s / 127.0 if not static else x_s,
            #     scale_b=(w_s / 127.0).transpose(0, 1),
            #     out_dtype=x.dtype,
            #     bias=bias.to(torch.float32).contiguous() if bias else None,
            # )


# monkey patch
_POSSIBLE_KERNELS[PlatformEnum.CUDA] = [KunlunScaledMMLinearKernel]
from vllm.model_executor.layers.quantization.kernels.scaled_mm import cutlass

cutlass.CutlassScaledMMLinearKernel = KunlunScaledMMLinearKernel
print(
    "[Monkey Patch Applied] >>> vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass.CutlassScaledMMLinearKernel \
      --> vllm_kunlun.ops.quantization.kernels.kunlun_scale_mm.KunlunScaledMMLinearKernel"
)
