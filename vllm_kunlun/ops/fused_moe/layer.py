"""layer.py"""

from contextlib import nullcontext
from typing import Callable, Optional, Union, get_args

import torch

from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod

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
                "EPLB not supported for `UnquantizedFusedMoEMethod` yet.")
        
        """forward_kunlun"""
        from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops
        if self.moe.use_ep:
            return ops.fused_moe_ep(x,
                             layer.w13_weight,
                             layer.w2_weight,
                             router_logits,
                             self.moe.ep_rank,
                             top_k,
                             renormalize=renormalize,
                             inplace=True,
                             use_grouped_topk=use_grouped_topk,
                             num_expert_group=num_expert_group,
                             topk_group=topk_group)
        else:
            return ops.fused_moe(x,
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
                             w1_bias = layer.w13_bias,
                             w2_bias = layer.w2_bias)

UnquantizedFusedMoEMethod.apply = apply

orig_init = FusedMoE.__init__

def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.register_parameter("w13_bias", None)
    self.register_parameter("w2_bias",  None)

FusedMoE.__init__ = new_init