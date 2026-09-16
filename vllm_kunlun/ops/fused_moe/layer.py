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

from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops
from vllm_kunlun.ops.moe import make_moe_router_for_layer, uninterleave_moe_w13


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

    def _select_monolithic(self):
        """Override parent: parent's __init__ assigns
        ``self.apply_monolithic = self._select_monolithic()`` which would
        otherwise shadow the class-level ``apply_monolithic`` defined below
        with ``forward_monolithic_cuda``. Return the class method instead."""
        return KunlunUnquantizedFusedMoEMethod.apply_monolithic.__get__(self)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Skip _setup_kernel() since Kunlun does not need Triton kernels."""
        FusedMoEMethodBase.process_weights_after_loading(self, layer)

        # `swigluoai` needs W13's rows reordered before a Kunlun kernel can
        # run it; the helper does that and hands back the name of the
        # activation that matches the new layout. Every other activation comes
        # back unchanged. The result is stashed on the layer, not on `self`,
        # because one method instance is shared by all the MoE layers.
        activation = self.moe.activation.value
        layer.kunlun_moe_activation = uninterleave_moe_w13(layer, activation)
        # Choosing the routing kernel here rather than per forward pass is what
        # makes a MoE config Kunlun cannot serve fail during setup, with the
        # weights loaded and nothing generated yet, instead of from inside the
        # first forward pass.
        layer.kunlun_moe_router = make_moe_router_for_layer(layer)

    def apply_monolithic(
        self,
        layer,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Monolithic mode entry point.
        When is_monolithic=True, FusedMoE.forward_impl calls this method
        directly with (layer, hidden_states, router_logits), bypassing
        the default routing logic.
        """
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
                layer.w13_weight,
                layer.w2_weight,
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
                activation=getattr(
                    layer, "kunlun_moe_activation", self.moe.activation.value
                ),
                custom_routing_function=layer.custom_routing_function,
                routed_scaling_factor=layer.routed_scaling_factor,
                swiglu_alpha=getattr(layer, "swiglu_alpha", None),
                swiglu_beta=getattr(layer, "swiglu_beta", None),
                swiglu_limit=getattr(layer, "swiglu_limit", None),
                router=layer.kunlun_moe_router,
            )
