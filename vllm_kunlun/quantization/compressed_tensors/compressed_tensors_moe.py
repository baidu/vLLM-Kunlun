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

import os
from typing import Callable, Optional, Union

import torch
import xspeedgate_ops  # noqa: F401  (register torch.ops.xspeedgate_ops)
from compressed_tensors import CompressionFormat
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
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

import vllm_kunlun.platforms.envs as kunlun_envs
from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops
from vllm_kunlun.quantization.kernels.quant_ops import dequant_int4_native

logger = init_logger(__name__)

_MOE_PRE_QUANT_MIN_TOKENS = 256


def _use_xspeed_minimax_shared_gate(num_fused_shared_experts: int) -> bool:
    """Use the graph-safe MiniMax top-k kernel for the fused shared slot."""
    return (
        num_fused_shared_experts == 1
        and not kunlun_envs.VLLM_KUNLUN_DISABLE_XSPEED_SHARED_GATE
    )


def _xspeed_minimax_shared_gate(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return routed top-k plus the always-selected MiniMax shared expert.

    FusedMoE applies ``routed_scaling_factor`` to the combined expert output
    later.  Keeping ``apply_routed_scaling_factor_on_output=False`` makes the
    routed weights sum to one and pre-divides the shared route by that factor,
    preserving the original independent-shared-expert math.
    """
    return torch.ops.xspeedgate_ops.minimax_m3_moe_fused_gate(
        router_logits.contiguous(),
        correction_bias.contiguous(),
        1,  # MiniMax-M3 uses ungrouped routing.
        1,
        top_k,
        num_fused_shared_experts,
        routed_scaling_factor,
        False,
    )


def _should_pre_quantize_moe_input(num_tokens: int, top_k: int) -> bool:
    if kunlun_envs.VLLM_KUNLUN_DISABLE_MOE_PRE_QUANT:
        return False

    # The extra scale-reorder kernel regresses the captured decode buckets.
    # Keep the optimization for large prefill batches where reduced BF16
    # traffic and quantization work amortize the additional launch.
    return num_tokens >= _MOE_PRE_QUANT_MIN_TOKENS


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
        moe,
    ):
        # Skip select_int8_moe_backend() in parent - this is a monolithic
        # implementation that uses torch.ops._C.moe_fc etc. directly.
        # The parent's __init__ calls select_int8_moe_backend() which only
        # supports Triton backend (not available on XPU for Int8).
        # Set attributes directly without calling nn.Module.__init__ to
        # avoid torch 2.9 call_super_init check.
        self.moe = moe
        self.moe_quant_config = None
        self.moe_kernel = None
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.static_input_scales = False
        self.int8_backend = None
        self.experts_cls = None
        self.moe_weight_scale_supported = set()
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self._non_persistent_buffers_set = set()
        self.training = True

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
        # Read routing params from layer attributes as source of truth,
        # falling back to function defaults (used when caller doesn't pass them).
        scoring_func = getattr(layer, 'scoring_func', scoring_func)
        num_expert_group = getattr(layer, 'num_expert_group', num_expert_group)
        topk_group = getattr(layer, 'topk_group', topk_group)
        e_score_correction_bias = getattr(layer, 'e_score_correction_bias', e_score_correction_bias)
        routed_scaling_factor = getattr(layer, 'routed_scaling_factor', routed_scaling_factor)
        output_routed_scaling_factor = getattr(
            layer, "kunlun_output_routed_scaling_factor",
            getattr(getattr(layer, "runner", None), "routed_scaling_factor",
                    routed_scaling_factor))
        global_num_experts = getattr(layer, 'global_num_experts', global_num_experts)
        # Read SwiGLU-OAI params from layer attributes (MiniMax-M3: alpha=1.702, beta=1.0, limit=7.0)
        swiglu_alpha = getattr(layer, 'swiglu_alpha', 1.0)
        swiglu_beta = getattr(layer, 'swiglu_beta', 1.0)
        swiglu_limit = getattr(layer, 'swiglu_limit', None)
        global_num_experts, up_gate_size, _ = layer.w13_weight.shape
        M, N = hidden_states.shape
        top_k = self.moe.experts_per_token
        num_fused_shared_experts = getattr(
            layer, "num_kunlun_fused_shared_experts", 0)
        normed_score = torch.empty(
            M, top_k, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, top_k, dtype=torch.int32, device=hidden_states.device)

        router_logits = router_logits.float()
        # block_statistic produced by the fused gate (kunlun_ops.moe_fused_gate),
        # reused by the moe_pre_sorted stage below to skip a redundant per-layer
        # gen_block_statistic launch. Stays None on paths that don't produce it.
        _gate_block_stat = None
        if scoring_func == "softmax":
            torch.ops._C.moe_softmax_topk_norm(
                x=router_logits,
                normed_score=normed_score,
                topk_index=topk_ids,
                block_statistic=None,
                stable=True,
            )
        elif scoring_func == "sigmoid":
            _M = router_logits.shape[0]
            _E = router_logits.shape[1]
            _is_group = (num_expert_group is not None and num_expert_group > 1
                         and topk_group is not None and topk_group < num_expert_group)
            # bias must be [E]/[1,E] fp32; tolerate None (no routing bias).
            if e_score_correction_bias is not None:
                _bias_1d = e_score_correction_bias.to(torch.float32).view(_E)
            else:
                _bias_1d = torch.zeros(_E, dtype=torch.float32,
                                       device=router_logits.device)

            # === P1: fused MoE gate (sigmoid+bias+top-k) ===
            # Use kunlun_ops.moe_fused_gate for the M3 n_group=1 path: it fuses
            # sigmoid+bias+top-k+sum-norm AND emits block_statistic as a third
            # output, letting moe_pre_sorted below skip a separate per-layer
            # gen_block_statistic launch (57 launches/forward saved). Keep the
            # group-limited PyTorch fallback for configs this kernel misses.
            _use_fused_gate = (
                os.environ.get("VLLM_KUNLUN_DISABLE_FUSED_MOE_GATE", "0") != "1"
                and not _is_group
                and top_k <= 16
            )
            if _use_fused_gate:
                import kunlun_ops as _kops_gate
                if num_fused_shared_experts > 0:
                    routed_top_k = top_k - num_fused_shared_experts
                    if _use_xspeed_minimax_shared_gate(
                        num_fused_shared_experts
                    ):
                        # The dedicated op directly emits routed top-k plus the
                        # fixed shared-expert slot.  Unlike the shared branch of
                        # kunlun_ops.moe_fused_gate, it is stable for M=1/8 CUDA
                        # Graph replay.  Build block statistics separately below.
                        _score, _tid = _xspeed_minimax_shared_gate(
                            router_logits,
                            _bias_1d,
                            top_k,
                            num_fused_shared_experts,
                            float(output_routed_scaling_factor),
                        )
                    else:
                        # Strict A/B fallback: mature routed-only gate followed
                        # by explicit construction of the shared route.
                        _routed_score = torch.empty(
                            _M,
                            routed_top_k,
                            dtype=torch.float32,
                            device=router_logits.device,
                        )
                        _routed_tid = torch.empty(
                            _M,
                            routed_top_k,
                            dtype=torch.int32,
                            device=router_logits.device,
                        )
                        _routed_block_stat = torch.empty(
                            12,
                            _E,
                            dtype=torch.int32,
                            device=router_logits.device,
                        )
                        _kops_gate.moe_fused_gate(
                            router_logits.contiguous(),
                            _bias_1d.view(1, _E).contiguous(),
                            1,       # num_expert_group (M3: 1)
                            1,       # topk_group (M3: 1)
                            routed_top_k,
                            0,       # shared experts are appended below
                            1.0,     # routed weights are normalized already
                            _routed_score,  # output_score
                            _routed_tid,    # topk_index
                            _routed_block_stat,
                        )
                        _score = torch.empty(
                            _M,
                            top_k,
                            dtype=torch.float32,
                            device=router_logits.device,
                        )
                        _tid = torch.empty(
                            _M,
                            top_k,
                            dtype=torch.int32,
                            device=router_logits.device,
                        )
                        _score[:, :routed_top_k].copy_(_routed_score)
                        _tid[:, :routed_top_k].copy_(_routed_tid)
                        shared_weight = (
                            1.0 / float(output_routed_scaling_factor)
                        )
                        for shared_idx in range(num_fused_shared_experts):
                            col = routed_top_k + shared_idx
                            _score.narrow(1, col, 1).fill_(shared_weight)
                            _tid.narrow(1, col, 1).fill_(_E + shared_idx)
                    _gate_block_stat = None
                else:
                    _score = torch.empty(_M, top_k, dtype=torch.float32,
                                         device=router_logits.device)
                    _tid = torch.empty(_M, top_k, dtype=torch.int32,
                                       device=router_logits.device)
                    _gate_block_stat = torch.empty(
                        12, _E, dtype=torch.int32,
                        device=router_logits.device)
                    _kops_gate.moe_fused_gate(
                        router_logits.contiguous(),
                        _bias_1d.view(1, _E).contiguous(),
                        1,       # num_expert_group (M3: 1)
                        1,       # topk_group (M3: 1)
                        top_k,
                        0,       # n_share_experts_fusion
                        1.0,     # routed weights are normalized already
                        _score,  # output_score
                        _tid,    # topk_index
                        _gate_block_stat,  # block_statistic, reused below
                    )
                normed_score = _score
                topk_ids = _tid
            elif not _is_group:
                _bias_2d = _bias_1d.view(1, _E).contiguous()
                _block_stat = torch.empty(12, _E, dtype=torch.int32,
                                          device=router_logits.device)
                torch.ops._C.moe_sigmoid_group_topk_norm(
                    router_logits,
                    topk_ids,
                    normed_score,
                    _block_stat,
                    _bias_2d,
                    1.0,
                    1,
                    1,
                )
                _gate_block_stat = _block_stat
            else:
                # Group-limited routing fallback (pure PyTorch) for configs the
                # fused kernels' n_group=1 path does not cover.
                scores = torch.sigmoid(router_logits)  # [M, E]
                scores_for_choice = (
                    scores + _bias_1d
                    if e_score_correction_bias is not None
                    else scores
                )
                _gs = _E // num_expert_group
                _gs_max = scores_for_choice.view(_M, num_expert_group, _gs).max(dim=-1).values
                _, _selected_grp = torch.topk(_gs_max, topk_group, dim=-1)
                _mask_3d = torch.zeros(_M, num_expert_group, _gs, dtype=torch.bool,
                                       device=scores_for_choice.device)
                _mask_3d.scatter_(dim=1, index=_selected_grp.unsqueeze(-1), value=True)
                scores_for_choice = torch.where(_mask_3d.view(_M, _E), scores_for_choice,
                                     torch.tensor(float("-inf"), device=scores_for_choice.device))
                _, idx = torch.topk(scores_for_choice, top_k, dim=-1)
                topk_weights = scores.gather(1, idx)
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                topk_weights = topk_weights.to(torch.float32)
                topk_ids.copy_(idx.to(torch.int32))
                normed_score.copy_(topk_weights)

        # === Item ④: fused Kunlun MoE dispatch / FC / activation / sum pipeline ===
        # Replaces the handwritten torch path (argsort/unique/_q/index_add_) with the
        # M2.7-validated _C op chain, swapping only the activation step to SwigluOAI
        # (swiglu_bias) for M3. All these _C ops are registered in M3's _custom_ops.py.
        import kunlun_ops as _kops  # noqa: F401  (kept for parity / debug use)
        _dev = hidden_states.device
        top_k = topk_ids.shape[1]
        _hidden_dim = layer.w2_weight.shape[1]

        # [CLAMP-FIX] dummy_run/cudagraph warmup synthetic routing can emit topk_ids
        # past the physical expert count; clamp so block/sort kernels stay in range.
        # Real decode ids are already valid -> no-op.
        topk_ids = topk_ids.clamp_(0, global_num_experts - 1)

        # This is a process-static fallback, so specializing it during the
        # initial model trace is intentional. When disabled, preserve the
        # original FX topology and allocation order for strict A/B comparison.
        _disable_pre_quant = kunlun_envs.VLLM_KUNLUN_DISABLE_MOE_PRE_QUANT
        if _disable_pre_quant:
            moe_expand = torch.empty(
                (M * top_k, N), dtype=hidden_states.dtype, device=_dev)
            expert_m = torch.empty(
                global_num_experts, dtype=torch.int32, device=_dev)
            sorted_tokens_num_lod = torch.empty(
                global_num_experts + 1, dtype=torch.int32, device=_dev)
            sorted_tokens_idx = torch.empty(
                M * top_k, dtype=torch.int32, device=_dev)

        # Reuse the block_statistic emitted by the fused gate when it is
        # consistent with the (post-clamp) topk_ids, i.e. the fused-gate path
        # ran AND the clamp above is a no-op. Gate ids are always in range for
        # the gate's expert width, so reuse is valid exactly when that width
        # matches global_num_experts. This static shape compare is graph-safe.
        if (
            _gate_block_stat is not None
            and _gate_block_stat.shape[1] == global_num_experts
        ):
            _block_stat_mp = _gate_block_stat
        else:
            _block_stat_mp = torch.zeros(12, global_num_experts, dtype=torch.int32, device=_dev)
            torch.ops._C.gen_block_statistic(topk_ids, _block_stat_mp)

        if _disable_pre_quant:
            torch.ops._C.moe_pre_sorted(
                x=hidden_states,
                topk_index=topk_ids,
                block_statistic=_block_stat_mp,
                moe_expand=moe_expand,
                moe_index=sorted_tokens_idx,
                expert_m=expert_m,
                sorted_tokens_num_lod=sorted_tokens_num_lod,
            )
        else:
            # Keep only the shape-dependent choice inside an opaque op. vLLM
            # traces first with max_num_batched_tokens and reuses that bytecode
            # without guards, so a Python M-based branch here is not valid.
            _xq, _xs, sorted_tokens_idx, _expert_m, sorted_tokens_num_lod = (
                torch.ops._C.moe_pre_sorted_quant(
                    x=hidden_states,
                    topk_index=topk_ids,
                    block_statistic=_block_stat_mp,
                    enable_pre_quant=True,
                    min_pre_quant_tokens=_MOE_PRE_QUANT_MIN_TOKENS,
                ))

        _UG = layer.w13_weight.shape[1]
        y = torch.empty(M, top_k, _UG, dtype=hidden_states.dtype, device=_dev)
        if _disable_pre_quant:
            moe_expand = moe_expand.view(M * top_k, _hidden_dim)
            _xq = torch.empty(
                moe_expand.shape, dtype=torch.int8, device=_dev)
            _xs = torch.empty(
                (moe_expand.shape[0], 1), dtype=torch.float32, device=_dev)
            torch.ops._C.quant2d(
                moe_expand, _xq, _xs, force_sdnn=True)
        torch.ops._C.moe_fc(
            x=_xq,
            x_perchannel_max=_xs,
            weight=layer.w13_weight,
            w_perchannel_max=layer.w13_weight_scale,
            sorted_tokens_num_lod=sorted_tokens_num_lod,
            sorted_tokens_idx=sorted_tokens_idx,
            moe_topk=top_k,
            y=y,
            topk_ids=topk_ids,
            act=None,
        )

        # SwigluOAI activation followed by per-route dynamic INT8 quant.
        # The default path fuses activation and quantization for every shape,
        # including decode. The environment fallback preserves the old FX
        # topology for strict A/B comparison.
        _d = y.shape[-1] // 2
        _y2d = y.reshape(-1, y.shape[-1])
        _activation_beta = (
            swiglu_beta if swiglu_limit is not None else 0.0)
        _activation_limit = (
            swiglu_limit if swiglu_limit is not None else 0.0)
        if kunlun_envs.VLLM_KUNLUN_DISABLE_FUSED_SWIGLU_QUANT:
            out1 = torch.empty(
                y.shape[:-1] + (_d,), dtype=y.dtype, device=_dev)
            _o2d = out1.reshape(-1, _d)
            _kops.swiglu_bias(
                _y2d, _o2d, recv_counts=None,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=_activation_beta,
                swiglu_limit=_activation_limit)
            out1 = out1.reshape(-1, out1.shape[-1])
            _xq = torch.empty(out1.shape, dtype=torch.int8, device=_dev)
            _xs = torch.empty(
                (out1.shape[0], 1), dtype=torch.float32, device=_dev)
            torch.ops._C.quant2d(out1, _xq, _xs, force_sdnn=True)
            del out1
        else:
            _xq, _xs = torch.ops._C.moe_swiglu_quant(
                x=_y2d,
                alpha=swiglu_alpha,
                beta=_activation_beta,
                limit=_activation_limit,
            )
        del y

        if _disable_pre_quant:
            del moe_expand
        out = torch.empty(M, top_k, layer.w2_weight.shape[1],
                          dtype=hidden_states.dtype, device=_dev)
        torch.ops._C.moe_fc(
            x=_xq,
            x_perchannel_max=_xs,
            weight=layer.w2_weight,
            w_perchannel_max=layer.w2_weight_scale,
            sorted_tokens_num_lod=sorted_tokens_num_lod,
            sorted_tokens_idx=sorted_tokens_idx,
            moe_topk=top_k,
            y=out,
            topk_ids=topk_ids,
            act=None,
        )

        # moe_post: reorder back to token order + weight by normed_score + sum.
        # dequant_scale all-ones (already per-token dequantized inside moe_fc).
        _dequant = getattr(layer, "_kunlun_moe_dequant_ones", None)
        if (
            _dequant is None
            or _dequant.device != _dev
            or _dequant.dtype != torch.float32
            or _dequant.shape[0] < M
            or _dequant.shape[1] != top_k
        ):
            _dequant = torch.ones([M, top_k], dtype=torch.float32, device=_dev)
            layer._kunlun_moe_dequant_ones = _dequant
        else:
            _dequant = _dequant[:M, :top_k]
        output = torch.empty([M, N], dtype=hidden_states.dtype, device=_dev)
        torch.ops._C.moe_post(
            x=out,
            moe_index=sorted_tokens_idx.view(M, top_k),
            normed_scale=normed_score,
            dequant_scale=_dequant,
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
