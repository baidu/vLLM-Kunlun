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
        # The fused RoutedExperts layer is resolved by looking up its
        # per-expert projection names. Different checkpoints / vllm builds
        # name these either gate/up/down_proj (HF-canonical) or w1/w3/w2
        # (Mixtral-style: w1=gate, w3=up, w2=down). Some vllm builds do not
        # remap w1/w2/w3 -> gate/up/down in the compressed-tensors target
        # map, so try both conventions and use the first that resolves.
        name_variant_suffixes = [
            [".0.gate_proj", ".0.up_proj", ".0.down_proj"],
            [".0.w1", ".0.w3", ".0.w2"],
        ]
        scheme_dict = None
        for proj_suffixes in name_variant_suffixes:
            dicts = [
                quant_config.get_scheme_dict(layer, layer_name + p)
                for p in proj_suffixes
            ]
            if any(d is None for d in dicts):
                continue
            # multiple schemes found
            if not all(cur_dict == dicts[0] for cur_dict in dicts):
                raise ValueError(
                    "All MoE projections need to have same "
                    "quantization scheme but found multiple"
                )
            scheme_dict = dicts[0]
            break

        if scheme_dict is None:  # ignored / unquantized layer
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


class KunlunCompressedTensorsWNA16MoEMethod(CompressedTensorsWNA16MoEMethod):

    def __init__(
        self,
        weight_quant,
        input_quant,
        moe: "FusedMoEConfig",  # type: ignore # noqa: F821
        layer_name: Optional[str] = None,
    ):
        # Skip parent __init__ which hard-asserts `strategy == "group"`.
        # Call grandparent directly so both "group" and "channel" work on P800.
        FusedMoEMethodBase.__init__(self, moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        assert self.strategy in ("group", "channel"), (
            f"Unsupported strategy: {self.strategy}, expected 'group' or 'channel'"
        )
        self.group_size = weight_quant.group_size if self.strategy == "group" else -1
        assert weight_quant.actorder != "group", (
            "grouped actorder isn't supported by this kernel"
        )
        assert weight_quant.symmetric, (
            "Only symmetric quantization is supported for MoE"
        )
        assert self.num_bits in WNA16_SUPPORTED_BITS, (
            f"Unsupported num_bits: {self.num_bits}, expected {WNA16_SUPPORTED_BITS}"
        )
        # We skip the parent __init__ (it selects a GPU marlin/flashinfer
        # backend via an oracle and asserts group-only). Newer vllm's inherited
        # create_weights / get_weight_shape need these attributes explicitly.
        # is_transposed=True -> "Marlin" (transposed) weight layout, which is
        # identical to the layout this plugin's process_weights_after_loading
        # (transpose(1,2)+XOR 0x88) and fused_moe_ct_w4a16 expect.
        self.symmetric = weight_quant.symmetric
        self.actorder = weight_quant.actorder
        self.is_transposed = True


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Preprocess loaded weights for ``ops.fused_moe_ct_w4a16``.

        1. Drop params created by the parent create_weights that Kunlun doesn't
           use (shapes / g_idx / g_idx sort indices), to free memory.
        2. Transpose packed weights, reinterpret as int8, in-place XOR 0x88 to
           convert uint4-packed -> signed-int representation.
        3. Transpose scales, cast to float32, multiply by 7.0.
        Modify ``.data`` in place to avoid doubling memory.
        """
        del layer.w13_weight_shape
        del layer.w2_weight_shape
        del layer.w13_weight_g_idx
        del layer.w2_weight_g_idx
        del layer.w13_g_idx_sort_indices
        del layer.w2_g_idx_sort_indices
        with torch.no_grad():
            w13_data = layer.w13_weight_packed.data.transpose(1, 2).contiguous()
            w13_data = w13_data.view(torch.int8)
            w13_data.bitwise_xor_(0x88)
            layer.w13_weight_packed.data = w13_data

            w2_data = layer.w2_weight_packed.data.transpose(1, 2).contiguous()
            w2_data = w2_data.view(torch.int8)
            w2_data.bitwise_xor_(0x88)
            layer.w2_weight_packed.data = w2_data

            w13_scale_data = layer.w13_weight_scale.data.transpose(1, 2).contiguous()
            w13_scale_data = w13_scale_data.to(torch.float32)
            w13_scale_data.mul_(7.0)
            layer.w13_weight_scale.data = w13_scale_data

            w2_scale_data = layer.w2_weight_scale.data.transpose(1, 2).contiguous()
            w2_scale_data = w2_scale_data.to(torch.float32)
            w2_scale_data.mul_(7.0)
            layer.w2_weight_scale.data = w2_scale_data

    @property
    def topk_indices_dtype(self):
        # Kunlun's moe_pre_sorted / gen_block_statistic both take int32.
        return torch.int32

    @property
    def is_monolithic(self) -> bool:
        # TP-only keeps the monolithic kernel that fuses routing in: it is
        # the production-validated path and the faster one. With EP enabled,
        # switch to vLLM's modular path, where routing is done by the runner's
        # select_experts and this method only performs the expert_map remap
        # plus the local-expert GEMMs. DeepEP ships Modular-only
        # prepare/finalize implementations, so EP has to take that path.
        return not self.moe.use_ep

    def select_gemm_impl(self, prepare_finalize, layer):
        """Build the modular experts object the DeepEP backends compose with.

        With an all2all backend that provides a prepare/finalize pair, vLLM
        builds a FusedMoEModularMethod out of that pair plus the experts
        implementation returned here instead of calling ``apply()`` above.

        Only the Standard activation format (a flat ``[M*topk, K]`` dispatch, as
        used by DeepEP high-throughput) is implemented. ``BatchedExperts``
        prepare/finalize pairs need a masked experts stage; refuse them here
        rather than letting ``modular_kernel._post_init_setup`` fail on the
        format mismatch.
        """
        # FusedMoEModularMethod.apply reads layer.w13_weight / layer.w2_weight,
        # while this path repacks the int4 weights into *_weight_packed. Expose
        # the packed tensors under the names vLLM expects (plain attributes, so
        # they are not registered as parameters a second time).
        if not hasattr(layer, "w13_weight"):
            layer.w13_weight = layer.w13_weight_packed.data
            layer.w2_weight = layer.w2_weight_packed.data

        import vllm.model_executor.layers.fused_moe.modular_kernel as mk

        if prepare_finalize.activation_format != (
            mk.FusedMoEActivationFormat.Standard
        ):
            raise NotImplementedError(
                "%s only implements the Standard activation format, got %s"
                % (type(self).__name__, prepare_finalize.activation_format)
            )
        return _kunlun_experts_cls()(self.moe, self.moe_quant_config, layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        # topk_ids are global expert ids while this rank only holds
        # num_local_experts weight slots. Follow vLLM's own approach
        # (moe_align_block_size.py:101 expert_ids = expert_map[expert_ids]):
        # remap along the expert dimension only and keep the M*topk row
        # layout unchanged.
        expert_map = getattr(layer, "expert_map", None)
        ids = topk_ids
        weights = topk_weights
        if expert_map is not None:
            local = expert_map.to(ids.device)[ids.long()]
            keep = local >= 0
            # Use where rather than multiplying by zero: a remote slot may
            # hold NaN, and NaN * 0 is still NaN.
            weights = torch.where(keep, weights, torch.zeros_like(weights))
            # Placeholder rows must not all be parked on local expert 0:
            # that expert would own ~15/16 of all rows, and the grouped GEMM
            # is known to misbehave under such skew (the sibling
            # moe_fc_mn_kmn out-of-bounds bug is distribution dependent too).
            # Spread them round-robin by pair index: the total row count stays
            # M*topk while per-expert row counts stay close to the TP case.
            n_local = layer.w13_weight_packed.shape[0]
            filler = torch.arange(
                local.numel(), device=local.device, dtype=local.dtype
            ).remainder_(n_local).view_as(local)
            ids = torch.where(keep, local, filler)
        return ops.moe_ct_w4a16_experts(
            hidden_states=x,
            w13_weight_packed_signed=layer.w13_weight_packed,
            w2_weight_packed_signed=layer.w2_weight_packed,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            topk_ids=ids,
            topk_weights=weights,
            moe_top_k=layer.top_k,
        )

    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """W4A16 MoE via the Kunlun fused kernel (routing done inside).

        Routing params are read from the RoutedExperts ``layer`` attributes.
        Weights are already signed int8 (XOR 0x88) and scales float32*7.0 after
        process_weights_after_loading.
        """
        if self.moe.use_ep:
            raise NotImplementedError(
                "EP mode is not supported for int4 packed weights yet."
            )
        return ops.fused_moe_ct_w4a16(
            hidden_states=x,
            w13_weight_packed_signed=layer.w13_weight_packed,
            w2_weight_packed_signed=layer.w2_weight_packed,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            router_logits=router_logits,
            moe_top_k=layer.top_k,
            renormalize=layer.renormalize,
            use_grouped_topk=layer.use_grouped_topk,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            scoring_func=layer.scoring_func,
            e_score_correction_bias=layer.e_score_correction_bias,
        )


# The RoutedExperts weight loader gates the compressed-tensors packed-weight
# transpose (loaded_weight.t()) on an exact class-name allowlist
# (routed_experts.py: "CompressedTensorsWNA16MoEMethod", etc.). Our subclass
# name is not in that list, so the transpose would be skipped and the checkpoint
# [N, K_packed] weights would be sharded on the wrong dim. Masquerade the class
# name so the loader applies the transpose.
KunlunCompressedTensorsWNA16MoEMethod.__name__ = "CompressedTensorsWNA16MoEMethod"
KunlunCompressedTensorsWNA16MoEMethod.__qualname__ = "CompressedTensorsWNA16MoEMethod"


_KUNLUN_EXPERTS_CLS = None


def _kunlun_experts_cls():
    """Define the modular experts class lazily to avoid import cycles."""
    global _KUNLUN_EXPERTS_CLS
    if _KUNLUN_EXPERTS_CLS is not None:
        return _KUNLUN_EXPERTS_CLS

    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceNoOP,
    )

    class KunlunW4A16Experts(mk.FusedMoEExpertsModular):
        """Kunlun W4A16 grouped GEMM as a modular-kernel experts stage."""

        def __init__(self, moe_config, quant_config, layer=None):
            super().__init__(moe_config, quant_config)
            # The per-channel dequant scales live on the layer, not in
            # FusedMoEQuantConfig, so keep a handle on it.
            self._layer = layer

        @property
        def expects_unquantized_inputs(self) -> bool:
            # W4A16: the weights are quantized, the activations are not.
            return True

        @staticmethod
        def activation_format() -> "mk.FusedMoEActivationFormat":
            return mk.FusedMoEActivationFormat.Standard

        @staticmethod
        def _supports_current_device() -> bool:
            return True

        @staticmethod
        def _supports_activation(activation) -> bool:
            return activation == MoEActivation.SILU

        @staticmethod
        def _supports_no_act_and_mul() -> bool:
            # silu_and_mul is fused inside the Kunlun expert pipeline.
            return False

        @staticmethod
        def _supports_parallel_config(moe_parallel_config) -> bool:
            return True

        @staticmethod
        def _supports_quant_scheme(weight_key, activation_key) -> bool:
            # W4A16: quantized weights, unquantized activations.
            return activation_key is None

        def finalize_weight_and_reduce_impl(self):
            # moe_post already scales by topk_weights and sums over topk.
            return TopKWeightAndReduceNoOP()

        def workspace_shapes(
            self,
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        ):
            return ((0,), (0,), (M, K))

        def apply(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        ):
            ids = topk_ids
            weights = topk_weights
            # DeepEP dispatch already returns local expert ids with -1 for the
            # slots that belong to another rank; the non-DeepEP path hands us
            # global ids plus an expert_map. Normalise both here.
            if expert_map is not None:
                local = expert_map.to(ids.device)[ids.long()]
            else:
                local = ids
            keep = local >= 0
            if not bool(keep.all()):
                # Zero the weight rather than the id: a remote slot may hold a
                # non-finite value, and NaN * 0 is still NaN.
                weights = torch.where(keep, weights, torch.zeros_like(weights))
                # Spread the dropped rows over all local experts instead of
                # parking them all on expert 0, so the grouped GEMM sees a
                # per-expert row count close to the non-EP case.
                n_local = w1.shape[0]
                filler = (
                    torch.arange(
                        local.numel(), device=local.device, dtype=local.dtype
                    )
                    .remainder_(n_local)
                    .view_as(local)
                )
                ids = torch.where(keep, local, filler)
            else:
                ids = local
            out = ops.moe_ct_w4a16_experts(
                hidden_states=hidden_states,
                w13_weight_packed_signed=w1,
                w2_weight_packed_signed=w2,
                w13_scale=(
                    self._layer.w13_weight_scale
                    if self._layer is not None
                    else self.w1_scale
                ),
                w2_scale=(
                    self._layer.w2_weight_scale
                    if self._layer is not None
                    else self.w2_scale
                ),
                topk_ids=ids,
                topk_weights=weights,
                moe_top_k=topk_ids.shape[-1],
            )
            output.copy_(out)

    _KUNLUN_EXPERTS_CLS = KunlunW4A16Experts
    return _KUNLUN_EXPERTS_CLS

