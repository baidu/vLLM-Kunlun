#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# Author: Liwei, Tang Shiwen
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

from typing import Optional

import torch
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    CutlassInt8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.platforms import current_platform


class KunlunScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_out_of_tree():
            return False, "requires OOT platform."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names

        # change scale to max for klx ops
        with torch.no_grad():
            getattr(layer, w_s_name).mul_(127.0)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w_q, w_s, x_s, x_zp, azp_adj = self._get_layer_params(layer)
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
                bias=bias.to(torch.float32).contiguous() if bias is not None else None,
            )
        else:  # symmetric
            return torch.ops._C.matmul(
                x=x_q,
                w=w_q.transpose(0, 1),
                out_dtype=x.dtype,
                x_pc_max=x_s * 127.0 if static else x_s,
                w_pc_max=w_s,
                bias=bias.to(torch.float32).contiguous() if bias is not None else None,
            )

            # backup option: lower performance
            # return torch.ops._C.cutlass_scaled_mm(
            #     a = x_q,
            #     b = w_q,
            #     scale_a=x_s / 127.0 if not static else x_s,
            #     scale_b=(w_s / 127.0).transpose(0, 1),
            #     out_dtype=x.dtype,
            #     bias=bias.to(torch.float32).contiguous() if bias is not None else None,
            # )
