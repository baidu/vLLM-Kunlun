#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# Author: Tang Shiwen, Li Wei
# Email: tangshiwen@baidu.com, liwei157@baidu.com
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

from typing import Callable, Optional, Union

import torch
import xspeedgate_ops  # noqa: F401
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method

from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops


def dequant_awq_moe(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize AWQ MoE weights in xspeedgate layout.

    Layout:
        qweight: [E, N, K // 2]
        scale:   [E, N, K // group_size]
        zp:      [E, N // 2, K // group_size]
        output:  [E, N, K]
    """
    assert qweight.dim() == 3, f"Expected 3D input for MoE, got {qweight.dim()}D"

    fpweight = []
    for expert_idx in range(qweight.shape[0]):
        expert_weight = torch.empty(
            qweight.shape[1],
            qweight.shape[2] * 2,
            dtype=scale.dtype,
            device=qweight.device,
        )
        torch.ops.xspeedgate_ops.dequant_int4(
            qweight[expert_idx],
            expert_weight,
            scale[expert_idx],
            zp[expert_idx],
            group_size,
        )
        fpweight.append(expert_weight)

    return torch.stack(fpweight, dim=0)


class KunlunMoeWNA16Method(MoeWNA16Method):

    @property
    def is_monolithic(self) -> bool:
        return True

    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        w13_weight = dequant_awq_moe(
            qweight=layer.w13_qweight,
            scale=self.moe_quant_config.w1_scale,
            zp=self.moe_quant_config.w1_zp,
            group_size=layer.group_size,
        )

        w2_weight = dequant_awq_moe(
            qweight=layer.w2_qweight,
            scale=self.moe_quant_config.w2_scale,
            zp=self.moe_quant_config.w2_zp,
            group_size=layer.group_size,
        )

        top_k = layer.moe_config.experts_per_token
        renormalize = layer.renormalize
        use_grouped_topk = layer.use_grouped_topk
        num_expert_group = layer.num_expert_group
        topk_group = layer.topk_group
        scoring_func = layer.scoring_func
        e_score_correction_bias = layer.e_score_correction_bias

        if self.moe.use_ep:
            return ops.fused_moe_ep(
                x,
                w13_weight,
                w2_weight,
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
                w13_weight,
                w2_weight,
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
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        w13_weight = dequant_awq_moe(
            qweight=layer.w13_qweight,
            scale=self.moe_quant_config.w1_scale,
            zp=self.moe_quant_config.w1_zp,
            group_size=layer.group_size,
        )

        w2_weight = dequant_awq_moe(
            qweight=layer.w2_qweight,
            scale=self.moe_quant_config.w2_scale,
            zp=self.moe_quant_config.w2_zp,
            group_size=layer.group_size,
        )

        if self.moe.use_ep:
            return ops.fused_moe_ep(
                x,
                w13_weight,
                w2_weight,
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
                w13_weight,
                w2_weight,
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
