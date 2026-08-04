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
import torch.nn.functional as F
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
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

from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops
from vllm_kunlun.quantization.kernels.quant_ops import dequant_int4_native

from vllm_kunlun.adapters.dsv4.moe_int8_factory import (
    create_modular_v4_method as _dsv4_create_v4_i8_method,
    int8_w8a8_route_native_enabled as _dsv4_i8_native_enabled,
)

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
            scoring_func = getattr(layer, "scoring_func", "softmax")
            if scoring_func == "sqrtsoftplus":
                logger.info_once(
                    "Using KunlunCompressedTensorsW8A8Int8MoEMethodV4 "
                    "(modular, scoring_func=sqrtsoftplus)"
                )
                return _dsv4_create_v4_i8_method(
                    weight_quant, input_quant, layer
                )
            return KunlunCompressedTensorsW8A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        # TODO: @liwei support w4a8
        # elif quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
        #     return CompressedTensorsW4A8Int8MoEMethod(
        #         weight_quant, input_quant, layer.moe_config
        #     )
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
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


class KunlunCompressedTensorsW8A8Int8MoEMethodV4(CompressedTensorsW8A8Int8MoEMethod):
    """DeepSeek-V4 W8A8 INT8 MoE method for Kunlun (modular, native INT8 pipeline).

    V4's routing uses sqrtsoftplus scoring plus a per-layer hash table, which
    is computed by the community router before this method is invoked. That
    means this method sees pre-computed ``topk_weights`` and ``topk_ids``
    (modular path, ``is_monolithic=False``), unlike the sibling
    ``KunlunCompressedTensorsW8A8Int8MoEMethod`` which expects raw
    ``router_logits`` and drives softmax/sigmoid routing inside the kernel.

    Default (KUNLUN_INT8_MOE_NATIVE=1): native grouped-GEMM pipeline
    (moe_pre_sorted -> quant2d -> INT8 moe_fc -> silu_and_mul -> quant2d ->
    INT8 moe_fc -> moe_post). Set KUNLUN_INT8_MOE_NATIVE=0 to force a torch
    per-expert loop fallback (W8A16, significantly slower).
    """

    _kunlun_int8_moe_v4 = True

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe,
        layer_name: Optional[str] = None,
    ):
        # Deliberately bypass CompressedTensorsW8A8Int8MoEMethod.__init__: it
        # runs the community int8 oracle (select_int8_moe_backend), which has
        # no OOT backend and raises NotImplementedError for the V4 deployment
        # config. Same escape hatch as KunlunFp8MoEMethod.
        FusedMoEMethodBase.__init__(self, moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        per_channel = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL
            and input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel or not input_quant.dynamic:
            raise ValueError(
                "Kunlun V4 INT8 MoE requires channelwise weights with dynamic "
                f"per-token activations, got {weight_quant}, {input_quant}"
            )
        self.static_input_scales = False
        # Unused: this method supplies its own apply() instead of a kernel
        # selected by the oracle. moe_kernel stays None (set by
        # FusedMoEMethodBase.__init__) so the framework treats this method as
        # non-modular.
        self.int8_backend = None
        self.experts_cls = None

    @property
    def is_monolithic(self) -> bool:
        return False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Native path (KUNLUN_INT8_MOE_NATIVE=1): kunlun native ``moe_fc``
        # expects per-channel abs-max, not the per-channel scale stored by
        # compressed-tensors (max/127). Match the monolithic INT8 method:
        # multiply the scales in place once, at load time. This method is
        # only invoked once per layer by the vLLM loader, so the in-place
        # ``mul_`` is safe; do NOT call it twice or scales become 127^2 x.
        # Torch fallback path (KUNLUN_INT8_MOE_NATIVE=0): keep weights as
        # loaded so ``_dequant_expert`` produces the correct bf16 tensor.
        if _dsv4_i8_native_enabled():
            with torch.no_grad():
                layer.w13_weight_scale.mul_(127.0)
                layer.w2_weight_scale.mul_(127.0)

    def maybe_make_prepare_finalize(self, routing_tables=None):
        return None

    def _dequant_expert(self, layer: torch.nn.Module, expert_id: int):
        """Dequantize a single expert's INT8 weights to BF16 (per-channel).

        w13: [2I, H] int8 * [2I, 1] fp32 -> [2I, H] bf16
        w2:  [H, I]  int8 * [H, 1]  fp32 -> [H, I]  bf16

        Fp32 intermediate: scales are loaded as fp32 params, so casting them
        to bf16 for the multiply loses precision on some checkpoints. This
        method is only used by the torch fallback path
        (KUNLUN_INT8_MOE_NATIVE=0); the native pipeline consumes the raw INT8
        tensors and per-channel abs-max scale directly via ``moe_fc``.
        """
        w13_scale = layer.w13_weight_scale[expert_id].to(torch.float32)
        w2_scale = layer.w2_weight_scale[expert_id].to(torch.float32)
        w13 = (
            layer.w13_weight[expert_id].to(torch.float32) * w13_scale
        ).to(torch.bfloat16)
        w2 = (
            layer.w2_weight[expert_id].to(torch.float32) * w2_scale
        ).to(torch.bfloat16)
        return w13, w2

    def _apply_torch_fallback(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
    ):
        """Correctness fallback: Python per-expert dequant + F.linear."""
        x_flat = x.reshape(-1, x.shape[-1])
        # Route metadata is small (M * top_k ints); CPU sync is unavoidable
        # for the per-expert gather anyway and matches the FP8 fallback.
        weights_cpu = topk_weights.reshape(-1, topk_weights.shape[-1]).cpu()
        ids_cpu = topk_ids.reshape(-1, topk_ids.shape[-1]).cpu()
        output = torch.zeros_like(x_flat)
        for expert_id in torch.unique(ids_cpu).tolist():
            token_rows_cpu, choices_cpu = torch.where(ids_cpu == expert_id)
            token_rows = token_rows_cpu.to(x_flat.device)
            expert_x = x_flat[token_rows].to(torch.bfloat16)
            w13, w2 = self._dequant_expert(layer, expert_id)
            gate, up = F.linear(expert_x, w13).chunk(2, dim=-1)
            expert_y = F.linear(F.silu(gate) * up, w2)
            expert_weights = (
                weights_cpu[token_rows_cpu, choices_cpu]
                .to(expert_y.dtype)
                .to(expert_y.device)
            )
            expert_y = expert_y * expert_weights.unsqueeze(-1)
            output.index_add_(0, token_rows, expert_y.to(output.dtype))
        return output.view_as(x)

    def _apply_native_int8_grouped(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
    ):
        """Native INT8 grouped-GEMM pipeline (modular).

        Mirrors the monolithic ``apply_monolithic`` of the sibling INT8 method
        but skips the in-kernel routing step: routing is done upstream by the
        community sqrtsoftplus + hash router, so we consume ``topk_ids`` /
        ``topk_weights`` directly. Pipeline: ``gen_block_statistic`` ->
        ``moe_pre_sorted`` -> ``quant2d`` -> INT8 ``moe_fc`` (gate+up) ->
        ``silu_and_mul`` -> ``quant2d`` -> INT8 ``moe_fc`` (down) ->
        ``moe_post``.
        """
        hidden_states = x.reshape(-1, x.shape[-1]).contiguous()
        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        M, N = hidden_states.shape
        dev = hidden_states.device
        E, up_gate_size, _ = layer.w13_weight.shape  # [E, 2I, H]
        hidden_dim = layer.w2_weight.shape[1]         # H
        top_k = topk_ids.shape[-1]

        topk_ids_i32 = (
            topk_ids.reshape(M, top_k).contiguous().to(torch.int32)
        )
        topk_w_f32 = (
            topk_weights.reshape(M, top_k).contiguous().to(torch.float32)
        )

        # num_blocks=12 mirrors ``apply_monolithic`` above; it is the
        # per-XPU-cluster block count used by ``gen_block_statistic`` /
        # ``moe_pre_sorted`` on this kunlun target.
        num_blocks = 12
        block_statistic = torch.zeros(
            num_blocks, E, dtype=torch.int32, device=dev
        )
        torch.ops._C.gen_block_statistic(topk_ids_i32, block_statistic)

        moe_expand = torch.empty(
            M * top_k, N, dtype=hidden_states.dtype, device=dev
        )
        expert_m = torch.zeros(E, dtype=torch.int32, device=dev)
        sorted_tokens_num_lod = torch.zeros(
            E + 1, dtype=torch.int32, device=dev
        )
        sorted_tokens_idx = torch.zeros(
            M * top_k, dtype=torch.int32, device=dev
        )
        torch.ops._C.moe_pre_sorted(
            x=hidden_states,
            topk_index=topk_ids_i32,
            block_statistic=block_statistic,
            moe_expand=moe_expand,
            moe_index=sorted_tokens_idx,
            expert_m=expert_m,
            sorted_tokens_num_lod=sorted_tokens_num_lod,
        )
        del expert_m, block_statistic

        # INT8 GEMM 1: gate + up.
        x_q = torch.empty(
            M * top_k, N, dtype=torch.int8, device=dev
        )
        x_scale = torch.empty(
            (M * top_k, 1), dtype=torch.float32, device=dev
        )
        torch.ops._C.quant2d(moe_expand, x_q, x_scale, force_sdnn=True)
        del moe_expand

        y = torch.empty(
            M, top_k, up_gate_size,
            dtype=hidden_states.dtype, device=dev,
        )
        torch.ops._C.moe_fc(
            x=x_q,
            x_perchannel_max=x_scale,
            weight=layer.w13_weight,
            w_perchannel_max=layer.w13_weight_scale,
            sorted_tokens_num_lod=sorted_tokens_num_lod,
            sorted_tokens_idx=sorted_tokens_idx,
            moe_topk=top_k,
            y=y,
            topk_ids=topk_ids_i32,
            act=None,
        )
        del x_q, x_scale

        # SwiGLU activation.
        d = y.shape[-1] // 2
        out1 = torch.empty(
            (*y.shape[:-1], d), dtype=y.dtype, device=dev
        )
        torch.ops._C.silu_and_mul(out1, y)
        del y
        out1 = out1.reshape(-1, d)

        # INT8 GEMM 2: down.
        x_q2 = torch.empty(out1.shape, dtype=torch.int8, device=dev)
        x_scale2 = torch.empty(
            (out1.shape[0], 1), dtype=torch.float32, device=dev,
        )
        torch.ops._C.quant2d(out1, x_q2, x_scale2, force_sdnn=True)
        del out1

        out = torch.empty(
            M, top_k, hidden_dim,
            dtype=hidden_states.dtype, device=dev,
        )
        torch.ops._C.moe_fc(
            x=x_q2,
            x_perchannel_max=x_scale2,
            weight=layer.w2_weight,
            w_perchannel_max=layer.w2_weight_scale,
            sorted_tokens_num_lod=sorted_tokens_num_lod,
            sorted_tokens_idx=sorted_tokens_idx,
            moe_topk=top_k,
            y=out,
            topk_ids=topk_ids_i32,
            act=None,
        )
        del x_q2, x_scale2, sorted_tokens_num_lod

        # Weighted scatter-back. topk_weights already carry the routing
        # weight (community sqrtsoftplus router normalises upstream), so
        # dequant_scale is a no-op.
        dequant_scale = torch.ones(
            [M, top_k], dtype=torch.float32, device=dev
        )
        output = torch.empty(
            [M, N], dtype=hidden_states.dtype, device=dev
        )
        torch.ops._C.moe_post(
            x=out,
            moe_index=sorted_tokens_idx.view(M, top_k),
            normed_scale=topk_w_f32,
            dequant_scale=dequant_scale,
            y=output,
        )
        return output.view_as(x)

    # Log the native->fallback transition at most once per method instance
    # (matches the FP8 path). Persistent failures then stay silent instead
    # of flooding the log every forward step.
    _native_fallback_warned = False

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
    ):
        # Shared experts are executed separately by FusedMoERunner for this
        # non-modular method (mirrors the FP8 path).
        del shared_experts, shared_experts_input
        if _dsv4_i8_native_enabled():
            try:
                return self._apply_native_int8_grouped(
                    layer, x, topk_weights, topk_ids
                )
            except Exception as ex:
                if not type(self)._native_fallback_warned:
                    type(self)._native_fallback_warned = True
                    logger.warning(
                        "Native INT8 MoE path failed (%r); falling back to "
                        "torch per-expert loop. Subsequent failures will be "
                        "silent.",
                        ex,
                    )
        return self._apply_torch_fallback(
            layer, x, topk_weights, topk_ids
        )


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
