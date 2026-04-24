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
from vllm.model_executor.kernels.linear.scaled_mm import (
    CutlassInt8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.platforms import current_platform


def _torch_scaled_int8_mm_fallback(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    x_max: torch.Tensor,
    w_max: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    orig_shape = x_q.shape[:-1]
    device = x_q.device
    x_2d = x_q.reshape(-1, x_q.shape[-1]).to(torch.float32)

    x_scale = x_max.reshape(-1, 1).to(device=device, dtype=torch.float32) / 127.0
    if x_scale.shape[0] == 1 and x_2d.shape[0] != 1:
        x_scale = x_scale.expand(x_2d.shape[0], 1)
    if x_scale.shape[0] != x_2d.shape[0]:
        raise RuntimeError(
            "scaled int8 fallback expected one activation scale per row, "
            f"got scale rows={x_scale.shape[0]} and input rows={x_2d.shape[0]}"
        )

    w_scale = w_max.reshape(-1).to(device=device, dtype=torch.float32) / 127.0
    if w_scale.numel() == 1 and w_q.shape[1] != 1:
        w_scale = w_scale.expand(w_q.shape[1])
    if w_scale.numel() != w_q.shape[1]:
        raise RuntimeError(
            "scaled int8 fallback expected one weight scale per output column, "
            f"got scales={w_scale.numel()} and output columns={w_q.shape[1]}"
        )

    x_dequant = x_2d * x_scale
    w_dequant = w_q.to(device=device, dtype=torch.float32) * w_scale.reshape(1, -1)
    out = torch.matmul(x_dequant, w_dequant)
    if bias is not None:
        out = out + bias.to(device=device, dtype=torch.float32)
    return out.to(out_dtype).reshape(*orig_shape, w_q.shape[1])


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
            x_max = x_s * 127.0 if static else x_s
            bias_fp32 = (
                bias.to(torch.float32).contiguous() if bias is not None else None
            )
            try:
                return torch.ops._C.matmul(
                    x=x_q,
                    w=w_q.transpose(0, 1),
                    out_dtype=x.dtype,
                    x_pc_max=x_max,
                    w_pc_max=w_s,
                    bias=bias_fp32,
                )
            except (RuntimeError, ValueError):
                return _torch_scaled_int8_mm_fallback(
                    x_q=x_q,
                    w_q=w_q,
                    x_max=x_max,
                    w_max=w_s,
                    out_dtype=x.dtype,
                    bias=bias_fp32,
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
