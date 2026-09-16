#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# Author: Li Wei, Tang Shiwen
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

from typing import Optional, Union

import torch
from compressed_tensors import CompressionFormat
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (  # noqa: E501
    CompressedTensorsW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_BITS,
)

from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops
from vllm_kunlun.ops.moe import make_moe_router_for_layer, uninterleave_moe_w13
from vllm_kunlun.quantization.kernels.quant_ops import dequant_int4_native

logger = init_logger(__name__)


class KunlunCompressedTensorsMoEMethod(FusedMoEMethodBase):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> FusedMoEMethodBase:
        # FusedMoE was made by combining multiple Linears so need to
        # make sure quantization config for Linear can target it
        quant_config._add_fused_moe_to_target_scheme_map()

        # Determine projection names from the layer's checkpoint naming
        # (e.g., MiniMax M2 uses "w1", "w2", "w3" instead of "gate_proj", etc.)
        ckpt_gate = getattr(layer, "ckpt_gate_proj_name", "gate_proj")
        ckpt_down = getattr(layer, "ckpt_down_proj_name", "down_proj")
        ckpt_up = getattr(layer, "ckpt_up_proj_name", "up_proj")

        unfused_names = [
            layer_name + f".0.{proj_name}"
            for proj_name in [ckpt_gate, ckpt_up, ckpt_down]
        ]
        # TODO: refactor this to use expert_mapping and check all layer numbers
        all_scheme_dicts = [
            quant_config.get_scheme_dict(layer, name) for name in unfused_names
        ]
        scheme_dict = all_scheme_dicts.pop()

        # multiple schemes found
        if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
            raise ValueError(
                "All MoE projections need to have same "
                "quantization scheme but found multiple"
            )

        if scheme_dict is None:  # ignored layer
            return UnquantizedFusedMoEMethod(layer.moe_config)

        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")
        format = scheme_dict.get("format")

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            valid_format_and_bits = (
                weight_quant.num_bits in WNA16_SUPPORTED_BITS
                and format == CompressionFormat.pack_quantized.value
            )

            if not valid_format_and_bits:
                raise ValueError(
                    "For Fused MoE layers, only format: ",
                    f"{CompressionFormat.pack_quantized.value} ",
                    f" and bits: {WNA16_SUPPORTED_BITS} is supported ",
                    f"but got format: {CompressionFormat.pack_quantized.value} "
                    f" and bits: {weight_quant.num_bits}",
                )

            logger.info_once("Using CompressedTensorsWNA16MoEMethod")
            return KunlunCompressedTensorsWNA16MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return KunlunCompressedTensorsW8A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        # TODO: @liwei support w4a8
        # elif quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
        #     return CompressedTensorsW4A8Int8MoEMethod(
        #         weight_quant, input_quant, layer.moe_config
        #     )
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )


class KunlunCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsW8A8Int8MoEMethod):
    def __init__(
        self,
        weight_quant,
        input_quant,
        moe: "FusedMoEConfig",
        layer_name: str | None = None,
    ):
        # Skip the parent __init__ which calls select_int8_moe_backend
        # (not applicable on Kunlun XPU). Instead, directly init FusedMoEMethodBase.
        from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase

        FusedMoEMethodBase.__init__(self, moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.static_input_scales = not self.input_quant.dynamic
        self.int8_backend = None
        self.experts_cls = None

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # NOTE: kunlun_ops use max as scale
        with torch.no_grad():
            layer.w13_weight_scale.mul_(127.0)
            layer.w2_weight_scale.mul_(127.0)
        # fused_moe_int8 hardcodes silu, so anything else would be computed as
        # silu without a word. Reject it here, next to the router check, rather
        # than returning wrong numbers.
        activation = self.moe.activation.value
        if activation not in ("silu", "swish"):
            raise NotImplementedError(
                f"Kunlun int8 MoE only implements silu, got {activation}"
            )
        # Choosing the routing kernel here rather than per forward pass is what
        # makes a MoE config Kunlun cannot serve fail during setup, with the
        # weights loaded and nothing generated yet, instead of from inside the
        # first forward pass.
        layer.kunlun_moe_router = make_moe_router_for_layer(layer)

    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return ops.fused_moe_int8(
            x,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            router_logits,
            self.moe.experts_per_token,
            layer.kunlun_moe_router,
            e_score_correction_bias=layer.e_score_correction_bias,
            custom_routing_function=layer.custom_routing_function,
        )


class KunlunCompressedTensorsWNA16MoEMethod(CompressedTensorsWNA16MoEMethod):
    """int4 MoE through KunlunOps.fused_moe.

    Monolithic like its int8 sibling, and for the same reason: `fused_moe` runs
    routing itself with Kunlun kernels, and upstream's routers return
    uninitialised buffers on XPU.
    """

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer.kunlun_moe_activation = uninterleave_moe_w13(
            layer, self.moe.activation.value
        )
        layer.kunlun_moe_router = make_moe_router_for_layer(layer)

    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # dequant packed weights to float16
        w13_weight = dequant_int4_native(
            weight_packed_uint8=layer.w13_weight_packed,
            scale=self.moe_quant_config.w1_scale,
        )
        w2_weight = dequant_int4_native(
            weight_packed_uint8=layer.w2_weight_packed,
            scale=self.moe_quant_config.w2_scale,
        )

        if self.moe.use_ep:
            return ops.fused_moe_ep(
                x,
                w13_weight,
                w2_weight,
                router_logits,
                self.moe.ep_rank,
                self.moe.experts_per_token,
                renormalize=layer.renormalize,
                inplace=True,
                use_grouped_topk=layer.use_grouped_topk,
                num_expert_group=layer.num_expert_group,
                topk_group=layer.topk_group,
                scoring_func=layer.scoring_func,
                e_score_correction_bias=layer.e_score_correction_bias,
                w13_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                activation=layer.kunlun_moe_activation,
                custom_routing_function=layer.custom_routing_function,
                routed_scaling_factor=layer.routed_scaling_factor,
                swiglu_alpha=getattr(layer, "swiglu_alpha", None),
                swiglu_beta=getattr(layer, "swiglu_beta", None),
                swiglu_limit=getattr(layer, "swiglu_limit", None),
                expert_map=getattr(layer, "expert_map", None),
                router=layer.kunlun_moe_router,
            )
        else:
            return ops.fused_moe(
                x,
                w13_weight,
                w2_weight,
                router_logits,
                self.moe.experts_per_token,
                renormalize=layer.renormalize,
                use_grouped_topk=layer.use_grouped_topk,
                num_expert_group=layer.num_expert_group,
                topk_group=layer.topk_group,
                scoring_func=layer.scoring_func,
                e_score_correction_bias=layer.e_score_correction_bias,
                w13_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                activation=layer.kunlun_moe_activation,
                custom_routing_function=layer.custom_routing_function,
                routed_scaling_factor=layer.routed_scaling_factor,
                router=layer.kunlun_moe_router,
            )
