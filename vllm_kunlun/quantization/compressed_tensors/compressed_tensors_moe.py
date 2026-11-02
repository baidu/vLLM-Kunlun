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
from torch.library import custom_op
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
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops
from vllm_kunlun.quantization.kernels.quant_ops import (
    convert_packed_int4,
    dequant_int4_native,
)

from .compressed_tensors import validate_int4_format, validate_w4a8_int

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
        # Qwen3.5 checkpoints can target fused expert tensors explicitly,
        # without the per-expert '.0.gate_proj' names used above.
        if (ckpt_gate, ckpt_up) == ("gate_proj", "up_proj"):
            fused_names = ["gate_up_proj", "gate_up_proj", ckpt_down]
            for i, (scheme, name, projection) in enumerate(
                zip(all_scheme_dicts, unfused_names, fused_names)
            ):
                # None also means explicitly ignored; retry only unmatched names.
                if scheme is None and not should_ignore_layer(
                    name,
                    ignore=quant_config.ignore,
                    fused_mapping=quant_config.packed_modules_mapping,
                ):
                    all_scheme_dicts[i] = quant_config.get_scheme_dict(
                        layer, f"{layer_name}.{projection}"
                    )
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
        quant_format = quant_config.quant_format if format is None else format
        validate_int4_format(weight_quant, quant_format)

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
        elif quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
            validate_w4a8_int(
                weight_quant,
                input_quant,
                scheme_dict.get("output_activations"),
            )
            return KunlunCompressedTensorsW4A8IntMoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )


@custom_op("vllm::kunlun_w4a8_int_moe", mutates_args=())
def w4a8_int_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    s13: torch.Tensor,
    s2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    m, hidden = x.shape
    top_k = topk_ids.shape[1]
    experts = w13.shape[0]
    if m == 0:
        return torch.empty_like(x)
    ids = topk_ids.to(torch.int32).contiguous()
    if m * top_k <= 768:
        indices, lod, expanded = torch.ops.xspeedgate_ops.moe_pre_small(
            ids,
            experts,
            index_have_neg=False,
            sort_mode=True,
            x=x.contiguous(),
        )
    else:
        stats = torch.zeros(12, experts, dtype=torch.int32, device=x.device)
        counts = torch.empty(experts, dtype=torch.int32, device=x.device)
        lod = torch.empty(experts + 1, dtype=torch.int32, device=x.device)
        indices = torch.empty(m * top_k, dtype=torch.int32, device=x.device)
        expanded = torch.empty(m * top_k, hidden, dtype=x.dtype, device=x.device)
        torch.ops._C.gen_block_statistic(ids, stats)
        torch.ops._C.moe_pre_sorted(
            x.contiguous(), ids, stats, expanded, indices, counts, lod
        )
    expanded = expanded.reshape(m * top_k, hidden)

    def gemm(inputs, weight, scales):
        q = torch.empty_like(inputs, dtype=torch.int8)
        maxima = torch.empty(inputs.shape[0], 1, dtype=torch.float32, device=x.device)
        torch.ops._C.quant2d(inputs, q, maxima, force_sdnn=True)
        output = torch.empty(m, top_k, weight.shape[1], dtype=x.dtype, device=x.device)
        if scales.shape[-1] > 1:
            import kunlun_ops

            kunlun_ops.moe_fc_v3(
                x=q,
                weight=weight,
                sorted_tokens_num_lod=lod,
                # moe_pre already expanded and sorted every token/expert pair.
                # V3 contiguous mode consumes and emits those rows directly;
                # moe_post below restores token order and applies router weights.
                sorted_tokens_idx=torch.arange(
                    m * top_k, dtype=torch.int32, device=x.device
                ),
                moe_topk=1,
                y=output.view(m * top_k, -1),
                x_perchannel_max=maxima.expand(-1, scales.shape[-1]).contiguous(),
                w_perchannel_max=scales,
                tgemm_type="int8_wo_t",
                scale_n=1,
                scale_k=2 * weight.shape[-1] // scales.shape[-1],
                use_pack_int4=True,
            )
            return output
        torch.ops._C.moe_fc(
            x=q,
            weight=weight,
            sorted_tokens_num_lod=lod,
            sorted_tokens_idx=indices,
            moe_topk=top_k,
            y=output,
            x_perchannel_max=maxima,
            w_perchannel_max=scales,
            topk_ids=ids,
            use_pack_int4=True,
        )
        return output

    gate_up = gemm(expanded, w13, s13)
    activated = torch.empty(m, top_k, w13.shape[1] // 2, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(activated, gate_up)
    down = gemm(activated.reshape(m * top_k, -1), w2, s2)
    output = torch.empty_like(x)
    torch.ops._C.moe_post(
        x=down,
        moe_index=indices.reshape(m, top_k),
        normed_scale=topk_weights.float().contiguous(),
        dequant_scale=torch.ones(m, top_k, dtype=torch.float32, device=x.device),
        y=output,
    )
    return output


@w4a8_int_moe.register_fake
def _fake_w4a8_int_moe(x, w13, w2, s13, s2, topk_weights, topk_ids):
    return torch.empty_like(x)


class KunlunCompressedTensorsW4A8IntMoEMethod(FusedMoEMethodBase):
    def __init__(self, weight_quant, input_quant, moe):
        super().__init__(moe)
        activation = getattr(moe.activation, "value", moe.activation)
        if moe.use_ep or activation != "silu" or moe.has_bias or moe.is_lora_enabled:
            raise ValueError(
                "Kunlun W4A8 MoE supports SiLU gated experts without EP, bias or LoRA"
            )
        if any(
            getattr(moe, name, None) is not None
            for name in (
                "swiglu_limit",
                "swiglu_alpha",
                "swiglu_beta",
            )
        ):
            raise ValueError("Kunlun W4A8 MoE does not support modified SwiGLU")
        if moe.in_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Kunlun W4A8 MoE requires FP16 or BF16 activations")
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.group_size = (
            weight_quant.group_size if weight_quant.strategy == "group" else -1
        )
        # Two INT4 values share each byte; groups must end on byte boundaries.
        if self.group_size != -1 and (
            self.group_size % 2
            or moe.hidden_dim % self.group_size
            or moe.intermediate_size_per_partition % self.group_size
        ):
            raise ValueError(
                "Kunlun packed W4A8 requires even group sizes "
                "aligned with TP partitions"
            )

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if hidden_size % 8 or intermediate_size_per_partition % 8:
            raise ValueError("Packed W4A8 requires K partitions divisible by 8")
        # Allocate directly in checkpoint layout [E, N, K/8]. This also
        # lets the expert loader shard without transposing or unpacking.
        for name, n, k in (
            ("w13_weight", 2 * intermediate_size_per_partition, hidden_size),
            ("w2_weight", hidden_size, intermediate_size_per_partition),
        ):
            groups = 1 if self.group_size == -1 else k // self.group_size
            for suffix, shape, dtype in (
                ("_packed", (num_experts, n, k // 8), torch.int32),
                ("_scale", (num_experts, n, groups), torch.float32),
                ("_shape", (num_experts, 2), torch.int64),
            ):
                param = torch.nn.Parameter(
                    torch.empty(shape, dtype=dtype), requires_grad=False
                )
                set_weight_attrs(param, extra_weight_attrs)
                if suffix == "_scale":
                    set_weight_attrs(
                        param, {"quant_method": self.weight_quant.strategy}
                    )
                layer.register_parameter(name + suffix, param)

    @property
    def is_monolithic(self):
        # Let vLLM's router preserve top-k, renormalization and scoring semantics.
        return False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        return None

    def get_fused_moe_quant_config(self, layer):
        # This method owns quantization and the packed GEMMs.
        return None

    def process_weights_after_loading(self, layer):
        for name in ("w13_weight", "w2_weight"):
            loaded_name = name + "_packed"
            weight = getattr(layer, loaded_name)
            setattr(
                layer,
                loaded_name,
                torch.nn.Parameter(convert_packed_int4(weight), requires_grad=False),
            )
            delattr(layer, name + "_shape")
            scales = getattr(layer, name + "_scale")
            setattr(
                layer,
                name + "_scale",
                torch.nn.Parameter(
                    scales.float().mul(7.0).contiguous(), requires_grad=False
                ),
            )

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        if getattr(layer, "apply_router_weight_on_input", False):
            raise ValueError(
                "Kunlun W4A8 MoE applies routing weights to expert outputs only"
            )
        return w4a8_int_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
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

    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden_states = x
        global_num_experts, up_gate_size, _ = layer.w13_weight.shape
        M, N = hidden_states.shape
        hidden_dim = layer.w2_weight.shape[1]
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
            (
                sorted_tokens_idx,
                sorted_tokens_num_lod,
                moe_expand,
            ) = torch.ops.xspeedgate_ops.moe_pre_small(
                topk_ids,
                global_num_experts,
                index_have_neg=False,
                sort_mode=True,
                x=hidden_states,
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
