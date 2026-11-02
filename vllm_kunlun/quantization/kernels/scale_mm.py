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
from torch.library import custom_op
from vllm.model_executor.kernels.linear import (
    CutlassInt8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .quant_ops import convert_packed_int4


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


@custom_op("vllm::kunlun_w4a8_int_linear", mutates_args=())
def w4a8_int_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_max: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # kunlun_ops loads the dependent libraries before importing the binding.
    import kunlun_ops
    import xpu_kunlun_ops

    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty((x2.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    if x2.shape[0] != 0:
        xq = torch.empty_like(x2, dtype=torch.int8)
        xmax = torch.empty((x2.shape[0], 1), dtype=torch.float32, device=x.device)
        kunlun_ops.quant2d(x2, xq, xmax, force_sdnn=True)
        if weight_max.shape[-1] > 1:
            # matmul_v2's packed INT4 path consumes per-channel maxima.
            # Use a single expert with explicit K-group scales for Dense.
            kunlun_ops.moe_fc_v3(
                x=xq,
                weight=weight.unsqueeze(0),
                sorted_tokens_num_lod=torch.arange(
                    2, dtype=torch.int32, device=x.device
                )
                * x2.shape[0],
                sorted_tokens_idx=torch.arange(
                    x2.shape[0], dtype=torch.int32, device=x.device
                ),
                moe_topk=1,
                y=out,
                x_perchannel_max=xmax.expand(-1, weight_max.shape[-1]).contiguous(),
                w_perchannel_max=weight_max.unsqueeze(0),
                bias=None if bias is None else bias.float().contiguous(),
                tgemm_type="int8_wo_t",
                scale_n=1,
                scale_k=2 * weight.shape[-1] // weight_max.shape[-1],
                use_pack_int4=True,
            )
            return out.reshape(*x.shape[:-1], weight.shape[0])
        # Final bool selects packed INT4. Scale arguments are maxima, not steps:
        # activations use 127 * step and signed INT4 weights use 7 * step.
        xpu_kunlun_ops.matmul_v2(
            xq,
            weight,
            out,
            False,
            True,
            1.0,
            0.0,
            None if bias is None else bias.float().contiguous(),
            None,
            None,
            xmax,
            weight_max,
            0,
            True,
        )
    return out.reshape(*x.shape[:-1], weight.shape[0])


@w4a8_int_linear.register_fake
def _fake_w4a8_int_linear(x, weight, weight_max, bias=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


class KunlunW4A8IntLinearKernel(MPLinearKernel):
    """Kunlun W4A8 GEMM for packed INT4 checkpoints."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_out_of_tree():
            return False, "Kunlun W4A8 requires the OOT platform"
        if c.weight_type != scalar_types.int4:
            return False, "Kunlun W4A8 requires signed INT4 weights"
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Kunlun W4A8 requires FP16 or BF16 activations"
        if c.out_type not in (None, c.act_type):
            return False, "Kunlun W4A8 requires output dtype to match activations"
        if c.zero_points or c.has_g_idx:
            return False, "Kunlun W4A8 requires symmetric weights without actorder"
        k = c.partition_weight_shape[0]
        if k % 8:
            return False, "Packed W4A8 requires K partitions divisible by 8"
        if c.group_size != -1 and (
            c.group_size <= 0 or c.group_size % 2 or k % c.group_size
        ):
            return (
                False,
                "Kunlun packed W4A8 requires even group sizes aligned with TP partitions",
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, scale, _, _ = self._get_weight_params(layer)
        packed = torch.nn.Parameter(convert_packed_int4(weight), requires_grad=False)
        setattr(layer, self.w_q_name, packed)
        if hasattr(layer, "weight_shape"):
            del layer.weight_shape
        setattr(
            layer,
            self.w_s_name,
            torch.nn.Parameter(
                scale.float().mul(7.0).contiguous(), requires_grad=False
            ),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight, scale, _, _ = self._get_weight_params(layer)
        return w4a8_int_linear(x, weight, scale, bias)
