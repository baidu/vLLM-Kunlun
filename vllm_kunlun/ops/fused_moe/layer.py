"""layer.py"""
import torch
import os
from typing import Callable, Optional

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.distributed import get_ep_group
from vllm.distributed.eplb.eplb_state import EplbState

from vllm.model_executor.layers.fused_moe import FusedMoE as VllmFusedMoE
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase as VllmFusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod as VllmUnquantizedFusedMoEMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig)

from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform

from vllm_kunlun.ops.quantization.compressed_tensors_moe import CompressedTensorsW8A8Int8MoEMethod


class FusedMoEMethodBase(VllmFusedMoEMethodBase):
    """FusedMoEMethodBase"""
    moe: FusedMoEConfig

@CustomOp.register("vllm_kunlun_unquantized_fused_moe")
class UnquantizedFusedMoEMethod(VllmUnquantizedFusedMoEMethod):
    """UnquantizedFusedMoEMethod"""
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
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
        linear_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """apply"""
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `UnquantizedFusedMoEMethod` yet.")

        return self.forward_kunlun(x=x,
                            layer=layer,
                            router_logits=router_logits,
                            top_k=top_k,
                            renormalize=renormalize,
                            use_grouped_topk=use_grouped_topk,
                            topk_group=topk_group,
                            num_expert_group=num_expert_group,
                            custom_routing_function=custom_routing_function,
                            linear_weights=linear_weights,
                            e_score_correction_bias=e_score_correction_bias)

    def forward_kunlun(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            use_grouped_topk: bool,
            top_k: int,
            router_logits: torch.Tensor,
            linear_weights: torch.Tensor,
            renormalize: bool,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """forward_kunlun"""
        from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops
        if self.moe.use_ep:
            return ops.fused_moe_ep(x,
                             layer.w13_weight,
                             layer.w2_weight,
                             router_logits,
                             linear_weights,
                             self.moe.ep_rank,
                             top_k,
                             renormalize=renormalize,
                             inplace=True,
                             use_grouped_topk=use_grouped_topk,
                             num_expert_group=num_expert_group,
                             topk_group=topk_group
                             )
        else:
            return ops.fused_moe(x,
                             layer.w13_weight,
                             layer.w2_weight,
                             router_logits,
                             linear_weights,
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
                             w2_bias = layer.w2_bias,
                             )

class FusedMoE(VllmFusedMoE):
    """FusedMoE"""
    def __init__(self,
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
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        is_sequence_parallel=False,
        has_bias: bool = False,
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
        e_score_correction_bias=e_score_correction_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
        activation=activation,
        enable_eplb=enable_eplb,
        num_redundant_experts=num_redundant_experts,
    )

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            model_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            model_dtype = params_dtype

        moe = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=model_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=has_bias,
            # quant_config=quant_config,
        )
        self.moe_config = moe
        self.quant_config = quant_config
        self.has_bias=has_bias

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        quant_method: Optional[QuantizeMethodBase] = None
        quant_method = (UnquantizedFusedMoEMethod(moe) if quant_config is None
                        else quant_config.get_quant_method(self, prefix))

        assert quant_method is not None
        # assert isinstance(quant_method, FusedMoEMethodBase)
        self.quant_method = quant_method

        if self.enable_eplb:
            from vllm_kunlun.ops.quantization.fp8 import (
                Fp8MoEMethod)
            if not isinstance(quant_method, Fp8MoEMethod):
                # TODO: Add support for additional quantization methods.
                # The implementation for other quantization methods does not
                # contain essential differences, but the current quant API
                # design causes duplicated work when extending to new
                # quantization methods, so I'm leaving it for now.
                # If you plan to add support for more quantization methods,
                # please refer to the implementation in `Fp8MoEMethod`.
                raise NotImplementedError("EPLB is only supported for FP8 "
                                          "quantization for now.")

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod",
                    "CompressedTensorsWNA16MarlinMoEMethod",
                    "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor = None,
                linear_weights: torch.Tensor = None):
        """forward"""
        # TODO: Once the OOM issue for the TPU backend is resolved, we will
        # switch to using the moe_forward custom op.
        if current_platform.is_tpu():
            return self.forward_impl(hidden_states, router_logits)
        else:
            forward_context: ForwardContext = get_forward_context()
            self = forward_context.no_compile_layers[self.layer_name]
            assert self.quant_method is not None
            return self.forward_impl(hidden_states, router_logits, linear_weights)
            # return torch.ops.vllm.moe_forward(hidden_states, router_logits,
            #                                   self.layer_name)

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor,
                     linear_weights: torch.Tensor = None):
        """forward_impl"""
        assert self.quant_method is not None
        if (self.moe_parallel_config.use_pplx_kernels
                or self.moe_parallel_config.use_deepep_ll_kernels):
            return self.forward_impl_chunked(hidden_states, router_logits)

        do_naive_dispatch_combine: bool = (
            self.dp_size > 1
            and not self.moe_parallel_config.use_deepep_ht_kernels)
        if do_naive_dispatch_combine:
            hidden_states, router_logits = get_ep_group().dispatch(
                hidden_states, router_logits)

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            enable_eplb=self.enable_eplb,
            expert_load_view=self.expert_load_view,
            logical_to_physical_map=self.logical_to_physical_map,
            logical_replica_count=self.logical_replica_count,
            linear_weights=linear_weights
        )

        if do_naive_dispatch_combine:
            final_hidden_states = get_ep_group().combine(final_hidden_states)

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            # Default set to False. (May have to add shared expert outputs.
            final_hidden_states = self.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states)

        return final_hidden_states
    @classmethod
    def make_expert_params_mapping(
            cls,
            ckpt_gate_proj_name: str,
            ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int,
            num_redundant_experts: int = 0) -> list[tuple[str, str, int, str]]:

        num_physical_experts = num_experts + num_redundant_experts

        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        physical_to_logical_map = \
            EplbState.build_initial_global_physical_to_logical_map(
            num_experts, num_redundant_experts)

        return [
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_" if weight_name
             in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.",
             expert_id, shard_id) for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]
