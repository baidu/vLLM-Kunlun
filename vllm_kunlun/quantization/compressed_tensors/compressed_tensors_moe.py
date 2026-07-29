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

from typing import Callable, Optional, Union

import torch
from compressed_tensors import CompressionFormat
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
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

from vllm.model_executor.utils import set_weight_attrs

from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops
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
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
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
        elif _is_packed_int4_weight(weight_quant, format):
            # W4A8: packed int4 weights + dynamic per-token int8 activations.
            logger.info_once("Using KunlunCompressedTensorsW4A8MoEMethod")
            return KunlunCompressedTensorsW4A8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )


def _is_packed_int4_weight(weight_quant, format) -> bool:
    return (
        weight_quant is not None
        and weight_quant.num_bits == 4
        and weight_quant.type == "int"
        and weight_quant.symmetric
        and weight_quant.strategy in ("channel", "group")
        and weight_quant.actorder != "group"
        and format == CompressionFormat.pack_quantized.value
    )


class KunlunCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsW8A8Int8MoEMethod):
    def __init__(self, weight_quant, input_quant, moe, layer_name=None):
        # Bypass the parent __init__: it selects an upstream int8 MoE backend
        # (cutlass/triton) that does not exist on Kunlun. The kunlun kernels are
        # driven directly from apply_monolithic.
        CompressedTensorsMoEMethod.__init__(self, moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.static_input_scales = not input_quant.dynamic

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # NOTE: kunlun_ops use max as scale
        with torch.no_grad():
            layer.w13_weight_scale.mul_(127.0)
            layer.w2_weight_scale.mul_(127.0)

    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden_states = x
        global_num_experts, up_gate_size, _ = layer.w13_weight.shape
        M, N = hidden_states.shape
        hidden_dim = layer.w2_weight.shape[1]
        top_k = self.moe.experts_per_token
        scoring_func = layer.scoring_func
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
                bias=layer.e_score_correction_bias,
                n_group=layer.num_expert_group,
                topk_group=layer.topk_group,
                scale=getattr(layer, "routed_scaling_factor", 1.0),
            )
        else:
            raise ValueError(f"Unsupported scoring_func: {scoring_func}")

        if M * top_k > 768:
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
            del expert_m
        else:
            sorted_tokens_idx, sorted_tokens_num_lod, moe_expand = (
                torch.ops.xspeedgate_ops.moe_pre_small(
                    topk_ids,
                    global_num_experts,
                    index_have_neg=False,
                    sort_mode=True,
                    x=hidden_states,
                )
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
        del x_q, x_scale, sorted_tokens_num_lod

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


class KunlunCompressedTensorsW4A8MoEMethod(CompressedTensorsWNA16MoEMethod):
    """Packed int4 expert weights with dynamic per-token int8 activations.

    Weights come from compressed-tensors ``pack-quantized`` checkpoints; the
    GEMMs run through the Kunlun ``moe_fc_v3`` int4 kernel instead of the
    CUDA/Triton kernels.
    """

    def __init__(self, weight_quant, input_quant, moe, layer_name=None):
        # Bypass the parent __init__, which rejects channel-wise weight scales.
        CompressedTensorsMoEMethod.__init__(self, moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        self.group_size = weight_quant.group_size

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # pack-quantized packs nibbles along the input dim, i.e.
        # [out, in // packed_factor]. The upstream loader only transposes
        # loaded weights for a hardcoded list of upstream method classes, so
        # register params in checkpoint layout and shard the packed dim for w2.
        pf = self.packed_factor
        if hidden_size % pf or intermediate_size_per_partition % pf:
            raise ValueError(
                f"Packed int4 MoE requires hidden_size ({hidden_size}) and "
                f"intermediate size per partition "
                f"({intermediate_size_per_partition}) divisible by {pf}."
            )
        extra_weight_attrs.update({"is_transposed": False, "quant_method": "channel"})
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        up_gate_size = w13_num_shards * intermediate_size_per_partition

        def _param(*shape, dtype):
            return torch.nn.Parameter(
                torch.empty(*shape, dtype=dtype), requires_grad=False
            )

        params = {
            "w13_weight_packed": _param(
                num_experts, up_gate_size, hidden_size // pf, dtype=torch.int32
            ),
            "w2_weight_packed": _param(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // pf,
                dtype=torch.int32,
            ),
            "w13_weight_scale": _param(
                num_experts, up_gate_size, 1, dtype=params_dtype
            ),
            "w2_weight_scale": _param(num_experts, hidden_size, 1, dtype=params_dtype),
            "w13_weight_shape": _param(num_experts, 2, dtype=torch.int64),
            "w2_weight_shape": _param(num_experts, 2, dtype=torch.int64),
        }
        for name, param in params.items():
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for unused in ("w13_weight_shape", "w2_weight_shape"):
            if hasattr(layer, unused):
                delattr(layer, unused)

        with torch.no_grad():
            for weight_name, scale_name in (
                ("w13_weight_packed", "w13_weight_scale"),
                ("w2_weight_packed", "w2_weight_scale"),
            ):
                packed = getattr(layer, weight_name)
                data = packed.data.view(torch.int8)
                # pack-quantized stores nibbles offset by 8; the kernel wants
                # two's-complement int4.
                data.bitwise_xor_(0x88)
                packed.data = data

                scale = getattr(layer, scale_name)
                scale_data = scale.data.float()
                # kunlun kernels take the per-channel max, not the step size.
                scale_data.mul_(7.0)
                scale.data = scale_data

    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.moe.use_ep:
            raise NotImplementedError(
                "Expert parallelism is not supported for packed int4 MoE weights."
            )

        hidden_states = x
        global_num_experts, up_gate_size, _ = layer.w13_weight_packed.shape
        M, N = hidden_states.shape
        hidden_dim = layer.w2_weight_packed.shape[1]
        top_k = self.moe.experts_per_token

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
        if layer.scoring_func == "softmax":
            torch.ops._C.moe_softmax_topk_norm(
                x=router_logits,
                normed_score=normed_score,
                topk_index=topk_ids,
                block_statistic=None,
                stable=True,
            )
        elif layer.scoring_func == "sigmoid":
            torch.ops._C.moe_sigmoid_group_topk_norm(
                x=router_logits,
                norm_score=normed_score,
                topk_index=topk_ids,
                block_static=block_statistic,
                bias=layer.e_score_correction_bias,
                n_group=layer.num_expert_group,
                topk_group=layer.topk_group,
                scale=getattr(layer, "routed_scaling_factor", 1.0),
            )
        else:
            raise ValueError(f"Unsupported scoring_func: {layer.scoring_func}")

        if M * top_k > 768:
            moe_expand = torch.empty(
                (M * top_k, N), dtype=hidden_states.dtype, device=hidden_states.device
            )
            expert_m = torch.zeros(
                global_num_experts, dtype=torch.int32, device=hidden_states.device
            )
            sorted_tokens_num_lod = torch.zeros(
                global_num_experts + 1, dtype=torch.int32, device=hidden_states.device
            )
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
            del expert_m
        else:
            sorted_tokens_idx, sorted_tokens_num_lod, moe_expand = (
                torch.ops.xspeedgate_ops.moe_pre_small(
                    topk_ids,
                    global_num_experts,
                    index_have_neg=False,
                    sort_mode=True,
                    x=hidden_states,
                )
            )

        moe_expand = moe_expand.view(M * top_k, hidden_dim)

        def _quant(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            q = torch.empty(t.shape, dtype=torch.int8, device=t.device)
            scale = torch.empty((t.shape[0], 1), dtype=torch.float32, device=t.device)
            torch.ops._C.quant2d(t, q, scale, force_sdnn=True)
            return q, scale

        # The packed-int4 kernel requires int8 activations, a per-token
        # activation scale column and a 2D output.
        y = torch.empty(
            M * top_k,
            up_gate_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        x_q, x_scale = _quant(moe_expand)
        del moe_expand
        torch.ops._C.moe_fc_v3(
            x=x_q,
            x_perchannel_max=x_scale,
            weight=layer.w13_weight_packed,
            w_perchannel_max=layer.w13_weight_scale,
            sorted_tokens_num_lod=sorted_tokens_num_lod,
            sorted_tokens_idx=sorted_tokens_idx,
            moe_topk=top_k,
            y=y,
            use_pack_int4=True,
            sort_mode=True,
        )

        d = up_gate_size // 2
        out1 = torch.empty(M * top_k, d, dtype=y.dtype, device=y.device)
        torch.ops._C.silu_and_mul(out1, y)
        del y

        out = torch.empty(
            M * top_k,
            hidden_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        x_q, x_scale = _quant(out1)
        del out1
        torch.ops._C.moe_fc_v3(
            x=x_q,
            x_perchannel_max=x_scale,
            weight=layer.w2_weight_packed,
            w_perchannel_max=layer.w2_weight_scale,
            sorted_tokens_num_lod=sorted_tokens_num_lod,
            sorted_tokens_idx=sorted_tokens_idx,
            moe_topk=top_k,
            y=out,
            use_pack_int4=True,
            sort_mode=True,
        )
        del x_q, x_scale, sorted_tokens_num_lod

        dequant_scale = torch.ones([M, top_k], dtype=torch.float32, device=out.device)
        output = torch.empty(
            [M, N], dtype=hidden_states.dtype, device=hidden_states.device
        )
        torch.ops._C.moe_post(
            x=out.view(M, top_k, hidden_dim),
            moe_index=sorted_tokens_idx.view(M, top_k),
            normed_scale=normed_score,
            dequant_scale=dequant_scale,
            y=output,
        )
        return output


class KunlunCompressedTensorsWNA16MoEMethod(CompressedTensorsWNA16MoEMethod):

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
