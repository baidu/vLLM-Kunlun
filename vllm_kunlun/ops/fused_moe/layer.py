#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
#
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

from typing import Callable, Optional

import torch
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod,
    FusedMoE,
)


class KunlunUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """apply"""
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `UnquantizedFusedMoEMethod` yet."
            )

        """forward_kunlun"""
        from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops

        if self.moe.use_ep:
            return ops.fused_moe_ep(
                x,
                layer.w13_weight,
                layer.w2_weight,
                router_logits,
                self.moe.ep_rank,
                top_k,
                renormalize=renormalize,
                inplace=True,
                use_grouped_topk=use_grouped_topk,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
        else:
            return ops.fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                router_logits,
                self.moe.ep_rank,
                top_k,
                renormalize=renormalize,
                inplace=True,
                use_grouped_topk=use_grouped_topk,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
            )


class KunlunFusedMoE(FusedMoE):
    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = 0,
        topk_group: Optional[int] = 0,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        zero_expert_num: Optional[int] = 0,
        zero_expert_type: Optional[str] = None,
    ):
        super().__init__(
            num_experts=num_experts,  # Global number of experts
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            prefix=prefix,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation=activation,
            enable_eplb=enable_eplb,
            num_redundant_experts=num_redundant_experts,
            has_bias=has_bias,
            is_sequence_parallel=is_sequence_parallel,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
        )
        self.has_bias = has_bias
        self.register_parameter("w13_bias", None)
        self.register_parameter("w2_bias", None)

        if (self.quant_config is None) or (
            should_ignore_layer(
                prefix,
                ignore=self.quant_config.ignore,
                fused_mapping=self.quant_config.packed_modules_mapping,
            )
        ):
            self.quant_method = KunlunUnquantizedFusedMoEMethod(self.moe_config)
            moe_quant_params = {
                "num_experts": self.local_num_experts,
                "hidden_size": hidden_size,
                "intermediate_size_per_partition": self.intermediate_size_per_partition,
                "params_dtype": params_dtype,
                "weight_loader": self.weight_loader,
            }
            self.quant_method.create_weights(layer=self, **moe_quant_params)


# monkey patch
from vllm.model_executor.layers.fused_moe import layer

layer.UnquantizedFusedMoEMethod = KunlunUnquantizedFusedMoEMethod
layer.FusedMoE = KunlunFusedMoE

print(
    "[Monkey Patch Applied] >>> from vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod \
      --> vllm_kunlun.ops.fused_moe.layer.KunlunUnquantizedFusedMoEMethod"
)
print(
    "[Monkey Patch Applied] >>> from vllm.model_executor.layers.fused_moe.layer.FusedMoE \
      --> vllm_kunlun.ops.fused_moe.layer.KunlunFusedMoE"
)
