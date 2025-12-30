# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import ScaledMMLinearLayerConfig
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import CutlassScaledMMLinearKernel
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise)

def can_implement_kunlun(
            cls, c: ScaledMMLinearLayerConfig=None) -> tuple[bool, Optional[str]]:
        return True, None

def klx_process_weights_after_loading(layer: torch.nn.Module) -> None:
    """modify scale -> abs max"""
    layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        layer.weight_scale.data * 127, requires_grad=False)

def process_weights_after_loading_kunlun(self, layer: torch.nn.Module) -> None:
    # WEIGHT
    # Cutlass kernels need transposed weight.
    weight = getattr(layer, self.w_q_name)
    replace_parameter(
        layer, self.w_q_name,
        torch.nn.Parameter(weight.t().data, requires_grad=False))

    # WEIGHT SCALE
    # Cutlass kernels support only per-tensor and per-channel.
    # If we have a fused module (QKV, MLP) with per tensor scales (thus N
    # scales being passed to the kernel), convert to the per-channel case.
    is_fused_module = len(layer.logical_widths) > 1
    weight_scale = getattr(layer, self.w_s_name)
    if is_fused_module and not self.config.is_channelwise:
        weight_scale = convert_to_channelwise(weight_scale,
                                              layer.logical_widths)
    replace_parameter(
        layer, self.w_s_name,
        torch.nn.Parameter(weight_scale.data, requires_grad=False))

    # INPUT SCALE
    if self.config.is_static_input_scheme:
        input_scale = getattr(layer, self.i_s_name)

        if self.config.input_symmetric:
            replace_parameter(
                layer, self.i_s_name,
                torch.nn.Parameter(input_scale.max(), requires_grad=False))
            setattr(layer, self.i_zp_name, None)
        else:
            input_zero_point = getattr(layer, self.i_zp_name)

            # reconstruct the ranges
            int8_traits = torch.iinfo(torch.int8)
            azps = input_zero_point.to(dtype=torch.int32)
            range_max = (input_scale * (int8_traits.max - azps)).max()
            range_min = (input_scale * (int8_traits.min - azps)).min()

            scale = (range_max - range_min) / (int8_traits.max -
                                               int8_traits.min)
            replace_parameter(
                layer, self.i_s_name,
                torch.nn.Parameter(scale, requires_grad=False))

            # AZP loaded as int8 but used as int32
            azp = (int8_traits.min -
                   range_min / scale).to(dtype=torch.int32)
            replace_parameter(layer, self.i_zp_name,
                              torch.nn.Parameter(azp, requires_grad=False))

    else:
        setattr(layer, self.i_s_name, None)
        setattr(layer, self.i_zp_name, None)

    # azp_adj is the AZP adjustment term, used to account for weights.
    # It does not depend on scales or azp, so it is the same for
    # static and dynamic quantization.
    # For more details, see csrc/quantization/cutlass_w8a8/Epilogues.md
    # https://github.com/vllm-project/vllm/blob/8d59dbb00044a588cab96bcdc028006ed922eb06/csrc/quantization/cutlass_w8a8/Epilogues.md
    if not self.config.input_symmetric:
        weight = getattr(layer, self.w_q_name)
        azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
        if self.config.is_static_input_scheme:
            # cutlass_w8a8 requires azp to be folded into azp_adj
            # in the per-tensor case
            azp_adj = getattr(layer, self.i_zp_name) * azp_adj
        setattr(layer, self.azp_adj_name,
                torch.nn.Parameter(azp_adj, requires_grad=False))
    else:
        setattr(layer, self.azp_adj_name, None)

    klx_process_weights_after_loading(layer)

def apply_weights_kunlun(self,
                layer: torch.nn.Module,
                x: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    x_q, x_scale, out = None, None, None
    w_t_shape = layer.weight.T.shape
    if isinstance(x, tuple):
        x_q, x_scale = x
        out = torch.empty((x_q.shape[0], w_t_shape[0]),
                        dtype=torch.bfloat16,
                        device=x_q.device)
    else:
        x_shape = x.shape
        x_q = torch.empty(x_shape, dtype=torch.int8, device=x.device)
        x_scale = torch.empty((x_shape[0], 1), dtype=torch.float32, device=x.device)
        out = torch.empty((x_shape[0], w_t_shape[0]),
                        dtype=x.dtype,
                        device=x.device)
        torch.ops._C.quant2d(x, x_q, x_scale, force_sdnn=True)
    torch.ops._C.gemm_I8_I8_bf16_nt(x_q, x_scale, layer.weight.T.data, layer.weight_scale.data, out)
    return out

CutlassScaledMMLinearKernel.apply_weights = apply_weights_kunlun
CutlassScaledMMLinearKernel.can_implement = can_implement_kunlun
CutlassScaledMMLinearKernel.process_weights_after_loading = process_weights_after_loading_kunlun