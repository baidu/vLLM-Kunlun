#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# Author: Li Wei, Tang Shiwen
# Email: liwei157@baidu.com, tangshiwen@baidu.com
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

import os
from typing import Callable, Optional, Union

import torch
from compressed_tensors.quantization import ActivationOrdering, QuantizationStrategy
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoEMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsW4A4Mxfp4MoEMethod,
    CompressedTensorsW4A8Int8MoEMethod,
    CompressedTensorsW8A8Fp8MoEMethod,
    CompressedTensorsW8A8Int8MoEMethod,
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
)

from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops
from vllm_kunlun.ops.quantization.kernels.quant_ops import dequant_int4_kunlun

logger = init_logger(__name__)

# Environment variable to control which MoE implementation to use
# USE_MOE_FC=1: use apply_moe_fc (original implementation with dequant)
# USE_MOE_FC=0 or not set: use apply_moe_fc_v3 (optimized implementation)
USE_MOE_FC = os.environ.get("USE_MOE_FC", "0") == "1"


class KunlunCompressedTensorsMoEMethod(FusedMoEMethodBase):

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
    ) -> "KunlunCompressedTensorsMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        # Check if a using "Linear" to select schemes
        if "Linear" in quant_config.target_scheme_map:
            matched_target = "Linear"
        else:
            # May have instead defined the linear layers in the fused model
            fused_layers = ["re:.*down_proj.*", "re:.*gate_proj.*", "re:.*up_proj.*"]
            current_scheme = None
            for fused_layer in fused_layers:
                # Check if one of the fused layers are defined in quant_config
                matched_target = find_matched_target(
                    layer_name=fused_layer,
                    module=layer,
                    targets=quant_config.target_scheme_map.keys(),
                    fused_mapping=quant_config.packed_modules_mapping,
                )

                # Only valid if down_proj, gate_proj, and up_proj
                # are mapped to the same quant scheme in the quant_config
                if current_scheme is None:
                    current_scheme = quant_config.target_scheme_map.get(matched_target)
                else:
                    assert current_scheme == quant_config.target_scheme_map.get(
                        matched_target
                    )

        weight_quant = quant_config.target_scheme_map[matched_target].get("weights")
        input_quant = quant_config.target_scheme_map[matched_target].get(
            "input_activations"
        )
        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            if (
                weight_quant.strategy in QuantizationStrategy.GROUP
                and weight_quant.actorder
                in (ActivationOrdering.GROUP, ActivationOrdering.DYNAMIC)
            ):
                raise ValueError(
                    "WNA16MoE is not supported with actorder=group/dynamic."
                )
            # MarlinMoE kernel is not supported on XPU.
            logger.warning_once("Using KunlunCompressedTensorsWNA16MoEMethod")
            return KunlunCompressedTensorsWNA16MoEMethod(quant_config, layer.moe_config)
        elif quant_config._is_fp4a4_nvfp4(weight_quant, input_quant):
            return CompressedTensorsW4A4Mxfp4MoEMethod(layer.moe_config)
        elif (
            quant_config._is_fp8_w8a8_sm90(weight_quant, input_quant)
            or quant_config._is_fp8_w8a8_sm100(weight_quant, input_quant)
            or quant_config._is_fp8_w8a8(weight_quant, input_quant)
        ):
            return CompressedTensorsW8A8Fp8MoEMethod(quant_config, layer.moe_config)
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return KunlunCompressedTensorsW8A8Int8MoEMethod(
                quant_config, layer.moe_config
            )
        elif quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
            return CompressedTensorsW4A8Int8MoEMethod(quant_config, layer.moe_config)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )


class KunlunCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsW8A8Int8MoEMethod):

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # NOTE: kunlun_ops use max as scale
        with torch.no_grad():
            layer.w13_weight_scale.mul_(127.0)
            layer.w2_weight_scale.mul_(127.0)

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
        hidden_states = x
        global_num_experts, up_gate_size, _ = layer.w13_weight.shape
        M, N = hidden_states.shape
        hidden_dim = layer.w2_weight.shape[1]
        normed_score = torch.empty(
            M, top_k, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, top_k, dtype=torch.int32, device=hidden_states.device)
        num_blocks = 12
        block_statistic = torch.zeros(
            num_blocks,
            global_num_experts,
            dtype=torch.int32,
            device=hidden_states.device,
        )

        router_logits = router_logits.float()
        if scoring_func == "softmax":
            torch.ops._C.moe_softmax_topk_norm(
                x=router_logits,
                normed_score=normed_score,
                topk_index=topk_ids,
                block_statistic=None,
                stable=True,
            )
        elif scoring_func == "sigmoid":
            torch.ops._C.moe_sigmoid_group_topk_norm(
                x=router_logits,
                norm_score=normed_score,
                topk_index=topk_ids,
                block_static=block_statistic,
                bias=e_score_correction_bias,
                n_group=num_expert_group,
                topk_group=topk_group,
                scale=routed_scaling_factor,
            )

        moe_expand = torch.empty(
            (M * top_k, N), dtype=hidden_states.dtype, device=hidden_states.device
        )  # [M, top_k, N], float
        expert_m = torch.zeros(
            global_num_experts, dtype=torch.int32, device=hidden_states.device
        )  # [E]
        sorted_tokens_num_lod = torch.zeros(
            global_num_experts + 1, dtype=torch.int32, device=hidden_states.device
        )  # [E+1]
        sorted_tokens_idx = torch.zeros(
            M * top_k, dtype=torch.int32, device=hidden_states.device
        )

        torch.ops._C.gen_block_statistic(topk_ids, block_statistic)

        torch.ops._C.moe_pre_sorted(
            x=hidden_states,
            topk_index=topk_ids,
            block_statistic=block_statistic,
            moe_expand=moe_expand,
            moe_index=sorted_tokens_idx,
            expert_m=expert_m,
            sorted_tokens_num_lod=sorted_tokens_num_lod,
        )

        y = torch.empty(
            M,
            top_k,
            layer.w13_weight.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        moe_expand = moe_expand.view(M * top_k, hidden_dim)

        x_shape = moe_expand.shape
        x_q = torch.empty(x_shape, dtype=torch.int8, device=moe_expand.device)
        x_scale = torch.empty(
            (x_shape[0], 1), dtype=torch.float32, device=moe_expand.device
        )
        torch.ops._C.quant2d(moe_expand, x_q, x_scale, force_sdnn=True)

        torch.ops._C.moe_fc(
            x=x_q,
            x_perchannel_max=x_scale,
            weight=layer.w13_weight,
            w_perchannel_max=layer.w13_weight_scale,
            sorted_tokens_num_lod=sorted_tokens_num_lod,
            sorted_tokens_idx=sorted_tokens_idx,
            moe_topk=top_k,
            y=y,
            topk_ids=topk_ids,
            # sort_mode=False,
            act=None,
        )

        d = y.shape[-1] // 2
        output_shape = y.shape[:-1] + (d,)
        out1 = torch.empty(output_shape, dtype=y.dtype, device=y.device)
        torch.ops._C.silu_and_mul(out1, y)

        del y

        out1 = out1.reshape(-1, out1.shape[-1])
        x_shape = out1.shape
        x_q = torch.empty(x_shape, dtype=torch.int8, device=moe_expand.device)
        x_scale = torch.empty(
            (x_shape[0], 1), dtype=torch.float32, device=moe_expand.device
        )
        torch.ops._C.quant2d(out1, x_q, x_scale, force_sdnn=True)
        del out1, moe_expand
        out = torch.empty(
            M,
            top_k,
            layer.w2_weight.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops._C.moe_fc(
            x=x_q,
            x_perchannel_max=x_scale,
            weight=layer.w2_weight,
            w_perchannel_max=layer.w2_weight_scale,
            sorted_tokens_num_lod=sorted_tokens_num_lod,
            sorted_tokens_idx=sorted_tokens_idx,
            moe_topk=top_k,
            y=out,
            topk_ids=topk_ids,
            # sort_mode=False,
            act=None,
        )
        del x_q, x_scale, sorted_tokens_num_lod, expert_m

        dequant_scale = torch.ones([M, top_k], dtype=torch.float32, device=out.device)
        output = torch.empty(
            [M, N], dtype=hidden_states.dtype, device=hidden_states.device
        )
        sorted_tokens_idx = sorted_tokens_idx.view(M, top_k)

        torch.ops._C.moe_post(
            x=out,
            moe_index=sorted_tokens_idx,
            normed_scale=normed_score,
            dequant_scale=dequant_scale,
            y=output,
        )
        return output


class KunlunCompressedTensorsWNA16MoEMethod(CompressedTensorsWNA16MoEMethod):

    def process_weights_after_loading_moe_fc(self, layer: torch.nn.Module) -> None:
        """Preprocessing for apply_moe_fc: use parent class method.

        This keeps the original uint8 format needed by apply_moe_fc.
        """
        super().process_weights_after_loading(layer)

    def process_weights_after_loading_moe_fc_v3(self, layer: torch.nn.Module) -> None:
        """Optimized preprocessing for apply_moe_fc_v3.

        This method:
        1. Transpose weights and scales
        2. Convert weights to signed int8 format (XOR 0x88)
        3. Multiply scales by 7.0 and convert to float32
        4. Release unused parameters to save memory

        Memory optimization:
        - Process one tensor at a time and release memory immediately
        - Modify .data directly instead of creating new Parameter to avoid double memory
        """
        # Release unused parameters FIRST to free memory before processing
        # These are created by parent class but not needed for Kunlun implementation
        del layer.w13_weight_shape
        del layer.w2_weight_shape
        del layer.w13_weight_g_idx
        del layer.w2_weight_g_idx
        del layer.w13_g_idx_sort_indices
        del layer.w2_g_idx_sort_indices

        with torch.no_grad():
            # Process w13 weights: transpose -> view as int8 -> in-place XOR 0x88
            # Modify .data directly to avoid creating new Parameter (saves memory)
            w13_data = layer.w13_weight_packed.data.transpose(1, 2).contiguous()
            w13_data = w13_data.view(torch.int8)
            w13_data.bitwise_xor_(0x88)  # in-place XOR
            layer.w13_weight_packed.data = (
                w13_data  # Direct assignment, no new Parameter
            )

            # Process w2 weights: same as w13
            w2_data = layer.w2_weight_packed.data.transpose(1, 2).contiguous()
            w2_data = w2_data.view(torch.int8)
            w2_data.bitwise_xor_(0x88)  # in-place XOR
            layer.w2_weight_packed.data = w2_data  # Direct assignment, no new Parameter

            # Process w13 scale: use in-place operations to reduce memory
            w13_scale_data = layer.w13_weight_scale.data.transpose(1, 2).contiguous()
            w13_scale_data.mul_(7.0)  # in-place multiply
            w13_scale_data = w13_scale_data.to(torch.float32)  # type conversion
            layer.w13_weight_scale.data = w13_scale_data  # Direct assignment

            # Process w2 scale: same as w13
            w2_scale_data = layer.w2_weight_scale.data.transpose(1, 2).contiguous()
            w2_scale_data.mul_(7.0)  # in-place multiply
            w2_scale_data = w2_scale_data.to(torch.float32)  # type conversion
            layer.w2_weight_scale.data = w2_scale_data  # Direct assignment

    def apply_moe_fc(
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
        """Original implementation using dequant and fused_moe."""
        # dequant packed weights to float16
        w13_weight = dequant_int4_kunlun(
            weight_packed_uint8=layer.w13_weight_packed,
            scale=self.moe_quant_config.w1_scale,
        )
        w2_weight = dequant_int4_kunlun(
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

    def apply_moe_fc_v3(
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
        """Optimized implementation using preprocessed weights and fuse_moe_ct_w4a16."""
        # EP mode is not supported for int4 packed weights
        if self.moe.use_ep:
            raise NotImplementedError(
                "EP mode is not supported for int4 packed weights yet."
            )

        # Use preprocessed weights and scales (processed in process_weights_after_loading)
        # Weights are already int8 (XOR 0x88) and scales are already float32 (multiplied by 7.0)
        return ops.fused_moe_ct_w4a16(
            hidden_states=x,
            w13_weight_packed_signed=layer.w13_weight_packed,
            w2_weight_packed_signed=layer.w2_weight_packed,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            router_logits=router_logits,
            moe_top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )


# Set methods based on USE_MOE_FC environment variable
if USE_MOE_FC:
    KunlunCompressedTensorsWNA16MoEMethod.process_weights_after_loading = (
        KunlunCompressedTensorsWNA16MoEMethod.process_weights_after_loading_moe_fc
    )
    KunlunCompressedTensorsWNA16MoEMethod.apply = (
        KunlunCompressedTensorsWNA16MoEMethod.apply_moe_fc
    )
    print("USE_MOE_FC=1: Using apply_moe_fc (original implementation with dequant)")
else:
    KunlunCompressedTensorsWNA16MoEMethod.process_weights_after_loading = (
        KunlunCompressedTensorsWNA16MoEMethod.process_weights_after_loading_moe_fc_v3
    )
    KunlunCompressedTensorsWNA16MoEMethod.apply = (
        KunlunCompressedTensorsWNA16MoEMethod.apply_moe_fc_v3
    )
    print("USE_MOE_FC=0 (default): Using apply_moe_fc_v3 (optimized implementation)")


# monkey patch
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_moe,
)

compressed_tensors_moe.CompressedTensorsW8A8Int8MoEMethod = (
    KunlunCompressedTensorsW8A8Int8MoEMethod
)
compressed_tensors_moe.CompressedTensorsMoEMethod = KunlunCompressedTensorsMoEMethod
compressed_tensors_moe.CompressedTensorsWNA16MoEMethod = (
    KunlunCompressedTensorsWNA16MoEMethod
)
KunlunCompressedTensorsWNA16MoEMethod.__name__ = "CompressedTensorsWNA16MoEMethod"

logger.info_once(
    "[Monkey Patch Applied] >>> vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.CompressedTensorsW8A8Int8MoEMethod \
      --> vllm_kunlun.ops.quantization.compressed_tensors_moe.KunlunCompressedTensorsW8A8Int8MoEMethod"
)
logger.info_once(
    "[Monkey Patch Applied] >>> vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.CompressedTensorsMoEMethod \
      --> vllm_kunlun.ops.quantization.compressed_tensors_moe.KunlunCompressedTensorsMoEMethod"
)
logger.info_once(
    "[Monkey Patch Applied] >>> vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.CompressedTensorsWNA16MoEMethod \
      --> vllm_kunlun.ops.quantization.compressed_tensors_moe.KunlunCompressedTensorsWNA16MoEMethod"
)
