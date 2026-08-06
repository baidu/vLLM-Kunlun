"""
Kunlun optimized FusedMoE - replaces UnquantizedFusedMoEMethod
Uses monolithic mode to receive router_logits directly and call KunlunOps.fused_moe
"""

import logging

import torch
import torch.nn.functional as F
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

from vllm_kunlun.ops.activation import swiglu
from vllm_kunlun.ops.fp8 import dequantize_fp8_blocks

import os as _os
# FP8 MoE dequant-weight cache. OFF by default: caching all experts in
# BF16 can double MoE weight memory (256 experts) and OOM. Enable only
# after validating headroom on a real FP8 run: KUNLUN_MOE_WEIGHT_CACHE=1
_MOE_WEIGHT_CACHE = _os.environ.get("KUNLUN_MOE_WEIGHT_CACHE", "0") == "1"
# FP8 MoE forward backend selector. Default ON: use the native BF16
# grouped-GEMM pipeline (dequant active experts per-step + moe_fc).
# Set KUNLUN_FP8_MOE_NATIVE=0 to force the legacy per-expert Python
# loop (correctness fallback, ~10s/token).
_FP8_MOE_NATIVE = _os.environ.get("KUNLUN_FP8_MOE_NATIVE", "1") == "1"


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


_logger = logging.getLogger("vllm_kunlun.ops.fused_moe")
_logged_routing_metadata = False


class KunlunFp8MoEMethod(Fp8MoEMethod):
    """Correctness-oriented FP8 MoE method for Kunlun.

    Selected expert weights are dequantized (block FP8 -> BF16) and executed
    with BF16 linear layers in a per-expert loop. This favours correctness and
    simplicity over throughput; a fused on-device FP8 MoE kernel is the intended
    future optimization.

    Idempotency marker _kunlun_fp8_moe lets the plugin post-import hook in
    vllm_kunlun/__init__.py detect that upstream Fp8MoEMethod has
    already been replaced with this class.
    """

    _kunlun_fp8_moe = True

    def __init__(self, quant_config, layer):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        if not self.block_quant:
            raise NotImplementedError(
                "Kunlun FP8 MoE correctness fallback requires block-quantized weights"
            )
        self.weight_scale_name = "weight_scale_inv"
        self.fp8_backend = None
        self.experts_cls = None

    @property
    def is_monolithic(self) -> bool:
        return False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        FusedMoEMethodBase.process_weights_after_loading(self, layer)
        # Keep weights as fp8 on device (saves memory).
        # Dequant happens lazily per-expert in _expert_weights via CPU path.

    def maybe_make_prepare_finalize(self, routing_tables=None):
        return None

    def _expert_weights(self, layer, expert_id):
        # Optional cache of dequantized BF16 expert weights (static after
        # loading). Avoids re-dequantizing block-FP8 weights every step.
        # Gated by KUNLUN_MOE_WEIGHT_CACHE to avoid OOM by default.
        cache = None
        if _MOE_WEIGHT_CACHE:
            cache = getattr(layer, "_kunlun_moe_bf16_cache", None)
            if cache is None:
                cache = {}
                layer._kunlun_moe_bf16_cache = cache
            hit = cache.get(expert_id)
            if hit is not None:
                return hit
        w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")[expert_id]
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")[expert_id]
        w13 = dequantize_fp8_blocks(layer.w13_weight[expert_id], w13_scale).to(
            layer.w13_weight.device
        )
        w2 = dequantize_fp8_blocks(layer.w2_weight[expert_id], w2_scale).to(
            layer.w2_weight.device
        )
        if cache is not None:
            try:
                cache[expert_id] = (w13, w2)
            except Exception:
                pass
        return (w13, w2)

    def _apply_per_expert_loop(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
    ):
        """Correctness fallback: original Python per-expert loop (slow)."""
        x_flat = x.reshape(-1, x.shape[-1])
        weights_cpu = topk_weights.reshape(-1, topk_weights.shape[-1]).cpu()
        ids_cpu = topk_ids.reshape(-1, topk_ids.shape[-1]).cpu()
        output = torch.zeros_like(x_flat)
        for expert_id in torch.unique(ids_cpu).tolist():
            token_rows_cpu, choices_cpu = torch.where(ids_cpu == expert_id)
            token_rows = token_rows_cpu.to(x_flat.device)
            expert_x = x_flat[token_rows].to(torch.bfloat16)
            w13, w2 = self._expert_weights(layer, expert_id)
            gate, up = F.linear(expert_x, w13).chunk(2, dim=-1)
            expert_y = F.linear(F.silu(gate) * up, w2)
            expert_weights = weights_cpu[token_rows_cpu, choices_cpu].to(
                expert_y.dtype
            ).to(expert_y.device)
            expert_y = expert_y * expert_weights.unsqueeze(-1)
            output.index_add_(0, token_rows, expert_y.to(output.dtype))
        return output.view_as(x)

    def _apply_native_bf16_grouped(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
    ):
        """Native BF16 grouped-GEMM pipeline.

        Per-step dequant of only the routed (active) experts' FP8 weights to
        BF16, then reuse the same on-device grouped-GEMM pipeline that the
        INT8 MoE path uses (moe_pre_sorted -> moe_fc -> silu -> moe_fc ->
        moe_post). Skips quant2d because BF16 moe_fc is supported directly.
        """
        dev = x.device
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]).contiguous()
        if x_flat.dtype != torch.bfloat16:
            x_flat = x_flat.to(torch.bfloat16)
        M, H = x_flat.shape
        TOPK = topk_ids.shape[-1]
        N_2I = layer.w13_weight.shape[1]  # 2 * I (already TP-sliced)
        I = layer.w2_weight.shape[-1]     # I (already TP-sliced)

        topk_ids_flat = (
            topk_ids.reshape(M, TOPK).contiguous().to(torch.int32)
        )
        topk_w_flat = (
            topk_weights.reshape(M, TOPK).contiguous().to(torch.float32)
        )

        # Compact remap: only routed experts participate. Cheap CPU unique
        # (routing buffer is tiny; CPU sync is unavoidable for weight
        # gather anyway).
        ids_cpu = topk_ids_flat.cpu()
        unique_cpu, inv_cpu = torch.unique(
            ids_cpu, sorted=True, return_inverse=True
        )
        K = int(unique_cpu.numel())
        compact_ids = inv_cpu.to(dev).to(torch.int32).reshape(M, TOPK).contiguous()

        # Dequant active experts to BF16, stacked into [K, N_2I, H] / [K, H, I].
        w13_scales = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scales = getattr(layer, f"w2_{self.weight_scale_name}")
        w13_active = torch.empty(K, N_2I, H, dtype=torch.bfloat16, device=dev)
        w2_active = torch.empty(K, H, I, dtype=torch.bfloat16, device=dev)
        for k, e_idx in enumerate(unique_cpu.tolist()):
            w13_active[k].copy_(
                dequantize_fp8_blocks(layer.w13_weight[e_idx], w13_scales[e_idx])
            )
            w2_active[k].copy_(
                dequantize_fp8_blocks(layer.w2_weight[e_idx], w2_scales[e_idx])
            )

        # Sort + gather via native op.
        num_blocks = 12
        block_stat = torch.zeros(
            num_blocks, K, dtype=torch.int32, device=dev
        )
        torch.ops._C.gen_block_statistic(compact_ids, block_stat)

        moe_expand = torch.empty(M * TOPK, H, dtype=torch.bfloat16, device=dev)
        sorted_idx = torch.zeros(M * TOPK, dtype=torch.int32, device=dev)
        expert_m = torch.zeros(K, dtype=torch.int32, device=dev)
        lod = torch.zeros(K + 1, dtype=torch.int32, device=dev)
        torch.ops._C.moe_pre_sorted(
            x=x_flat,
            topk_index=compact_ids,
            block_statistic=block_stat,
            moe_expand=moe_expand,
            moe_index=sorted_idx,
            expert_m=expert_m,
            sorted_tokens_num_lod=lod,
        )
        del expert_m, block_stat

        # Grouped GEMM 1: gate_up projection.
        y1 = torch.empty(M, TOPK, N_2I, dtype=torch.bfloat16, device=dev)
        torch.ops._C.moe_fc(
            x=moe_expand,
            weight=w13_active,
            sorted_tokens_num_lod=lod,
            sorted_tokens_idx=sorted_idx,
            moe_topk=TOPK,
            y=y1,
            act=None,
            topk_ids=compact_ids,
        )
        del moe_expand

        # SwiGLU activation. Honour `swiglu_limit` (10.0 for DeepSeek-V4);
        # dropping it inflates the sink token's output, see
        # `vllm_kunlun/ops/activation.py::swiglu`.
        out1 = swiglu(
            y1.reshape(-1, N_2I), getattr(layer, "swiglu_limit", None)
        )
        del y1

        # Grouped GEMM 2: down projection.
        y2 = torch.empty(M, TOPK, H, dtype=torch.bfloat16, device=dev)
        torch.ops._C.moe_fc(
            x=out1,
            weight=w2_active,
            sorted_tokens_num_lod=lod,
            sorted_tokens_idx=sorted_idx,
            moe_topk=TOPK,
            y=y2,
            act=None,
            topk_ids=compact_ids,
        )
        del out1, lod

        # Weighted scatter-back: normed_scale = topk_weights (fp32).
        dq = torch.ones(M, TOPK, dtype=torch.float32, device=dev)
        output = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        torch.ops._C.moe_post(
            x=y2,
            moe_index=sorted_idx.view(M, TOPK),
            normed_scale=topk_w_flat,
            dequant_scale=dq,
            y=output,
        )
        return output.view(orig_shape).to(x.dtype)

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
    ):
        # FusedMoERunner executes shared experts separately for this non-modular
        # fallback. Accept both arguments to match the vLLM 0.25.1 contract.
        del shared_experts, shared_experts_input
        global _logged_routing_metadata
        if not _logged_routing_metadata:
            ids_c = topk_ids.reshape(-1, topk_ids.shape[-1])
            _logger.warning(
                "FP8 MoE routing metadata: shape=%s dtype=%s "
                "backend=%s (KUNLUN_FP8_MOE_NATIVE=%s)",
                tuple(ids_c.shape),
                ids_c.dtype,
                "native-bf16-grouped"
                if _FP8_MOE_NATIVE
                else "per-expert-loop",
                _os.environ.get("KUNLUN_FP8_MOE_NATIVE", "1"),
            )
            _logged_routing_metadata = True

        if _FP8_MOE_NATIVE:
            try:
                return self._apply_native_bf16_grouped(
                    layer, x, topk_weights, topk_ids
                )
            except Exception as ex:
                _logger.warning(
                    "Native BF16 MoE path failed (%r); falling back to "
                    "per-expert loop for this step.",
                    ex,
                )
        return self._apply_per_expert_loop(
            layer, x, topk_weights, topk_ids
        )


# NOTE: The replacement of upstream Fp8MoEMethod with KunlunFp8MoEMethod
# is performed by the centralized post-import hook in vllm_kunlun/__init__.py
# (target: vllm.model_executor.layers.quantization.fp8), consistent with the
# rest of the plugin's patching architecture. Importing this module no longer
# has a registration side effect.
