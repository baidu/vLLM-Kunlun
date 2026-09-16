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

    def _select_monolithic(self):
        """Override parent: parent's __init__ assigns
        ``self.apply_monolithic = self._select_monolithic()`` which would
        otherwise shadow the class-level ``apply_monolithic`` defined below
        with ``forward_monolithic_cuda``. Return the class method instead."""
        return KunlunUnquantizedFusedMoEMethod.apply_monolithic.__get__(self)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Skip _setup_kernel() since Kunlun does not need Triton kernels."""
        FusedMoEMethodBase.process_weights_after_loading(self, layer)
        if self.moe.use_ep:
            self._check_linear_expert_placement(layer)

    def _check_linear_expert_placement(self, layer: torch.nn.Module) -> None:
        """The EP sorted path requires experts to be placed linearly
        (rank r owns [r*local_E, (r+1)*local_E)).
        """
        strategy = getattr(layer, "expert_placement_strategy", "linear")
        strategy = getattr(strategy, "value", strategy)
        if strategy != "linear":
            raise NotImplementedError(
                f"The Kunlun EP sorted path only supports linear expert "
                f"placement, but got {strategy!r}. "
                "moe_ep_pre_sorted picks this rank's experts as "
                "ep_rank*local_E inside the kernel and does not honor "
                "expert_map."
            )
        expert_map = getattr(layer, "expert_map", None)
        if expert_map is not None:
            local_e = self.moe.num_local_experts
            base = self.moe.ep_rank * local_e
            expected = torch.full_like(expert_map, -1)
            expected[base : base + local_e] = torch.arange(
                local_e, device=expert_map.device, dtype=expert_map.dtype
            )
            if not torch.equal(expert_map, expected):
                raise NotImplementedError(
                    "The Kunlun EP sorted path only supports linear expert "
                    "placement, but expert_map does not match the linear "
                    f"mapping [{base}, {base + local_e}) -> 0..{local_e - 1} "
                    "(EPLB rebalancing?). moe_ep_pre_sorted does not honor "
                    "expert_map."
                )

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
        from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops

        expert_lora = getattr(layer, "kunlun_expert_lora", None)

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
                ep_size=self.moe.ep_size,
                global_num_experts=self.moe.num_experts,
                scoring_func=layer.scoring_func,
                e_score_correction_bias=layer.e_score_correction_bias,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                expert_lora=expert_lora,
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
                expert_lora=expert_lora,
            )
