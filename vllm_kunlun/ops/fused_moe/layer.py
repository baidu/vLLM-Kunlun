"""
Kunlun optimized FusedMoE - replaces UnquantizedFusedMoEMethod
Uses monolithic mode to receive router_logits directly and call KunlunOps.fused_moe
"""

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)


@CustomOp.register_oot(name="UnquantizedFusedMoEMethod")
class KunlunUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """
    Kunlun optimized UnquantizedFusedMoEMethod.

    Key design:
    - is_monolithic = True: FusedMoE calls apply_monolithic(layer, x, router_logits)
      instead of routing first and then calling apply(layer, x, topk_weights, topk_ids).
    - This passes router_logits directly to KunlunOps.fused_moe, which handles
      routing internally with device-optimized kernels.
    """

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Skip _setup_kernel() since Kunlun does not need Triton kernels."""
        FusedMoEMethodBase.process_weights_after_loading(self, layer)

    def apply_monolithic(
        self,
        layer,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Monolithic mode entry point.
        When is_monolithic=True, FusedMoE.forward_impl calls this method
        directly with (layer, hidden_states, router_logits), bypassing
        the default routing logic.
        """
        from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops

        if self.moe.use_ep:
            return ops.fused_moe_ep(
                x,
                layer.w13_weight,
                layer.w2_weight,
                router_logits,
                self.moe.ep_rank,
                self.moe.experts_per_token,
                renormalize=layer.renormalize,
                inplace=True,
                use_grouped_topk=layer.use_grouped_topk,
                num_expert_group=layer.num_expert_group,
                topk_group=layer.topk_group,
            )
        else:
            return ops.fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
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
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
            )
