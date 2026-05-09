#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# Author: Tang Shiwen, Li Wei
# Email: tangshiwen@baidu.com, liwei157@baidu.com
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

from typing import Callable

import torch
import xspeedgate_ops  # noqa: F401
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method

from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops

SMALL_BATCH_FUSED_MOE_THRESHOLD = 400
MOE_WNA16_BLOCK_SIZE_M = 16
_MOE_WNA16_GEMM_PACK_ROWS = 4
_MOE_WNA16_GEMM_PACK_COLS = 16
_MOE_WNA16_GEMM_PACK_SIZE = _MOE_WNA16_GEMM_PACK_ROWS * _MOE_WNA16_GEMM_PACK_COLS
_MOE_WNA16_GEMM_ALIGNED_ATTR = "_kunlun_moe_wna16_gemm_aligned"


def _align_qweight_to_moe_wna16_gemm(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.numel() == 0 or qweight.numel() % _MOE_WNA16_GEMM_PACK_SIZE != 0:
        return qweight.contiguous()
    return (
        qweight.reshape(-1, _MOE_WNA16_GEMM_PACK_ROWS, _MOE_WNA16_GEMM_PACK_COLS)
        .transpose(-1, -2)
        .reshape_as(qweight)
        .contiguous()
    )


def _restore_qweight_from_moe_wna16_gemm(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.numel() == 0 or qweight.numel() % _MOE_WNA16_GEMM_PACK_SIZE != 0:
        return qweight.contiguous()
    return (
        qweight.reshape(-1, _MOE_WNA16_GEMM_PACK_COLS, _MOE_WNA16_GEMM_PACK_ROWS)
        .transpose(-1, -2)
        .reshape_as(qweight)
        .contiguous()
    )


def dequant_awq_moe(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor | None,
    group_size: int,
    qweight_is_gemm_aligned: bool = False,
) -> torch.Tensor:
    """Dequantize AWQ MoE weights in xspeedgate layout.

    Layout:
        qweight: [E, N, K // 2]
        scale:   [E, N, K // group_size]
        zp:      [E, N // 2, K // group_size]
        output:  [E, N, K]
    """
    assert qweight.dim() == 3, f"Expected 3D input for MoE, got {qweight.dim()}D"

    if qweight_is_gemm_aligned:
        qweight = _restore_qweight_from_moe_wna16_gemm(qweight)

    fpweight = []
    for expert_idx in range(qweight.shape[0]):
        expert_weight = torch.empty(
            qweight.shape[1],
            qweight.shape[2] * 2,
            dtype=scale.dtype,
            device=qweight.device,
        )
        torch.ops.xspeedgate_ops.dequant_int4(
            qweight[expert_idx],
            expert_weight,
            scale[expert_idx],
            zp[expert_idx] if zp is not None else None,
            group_size,
        )
        fpweight.append(expert_weight)

    return torch.stack(fpweight, dim=0)


def _route_moe(
    router_logits: torch.Tensor,
    top_k: int,
    scoring_func: str,
    num_expert_group: int | None,
    topk_group: int | None,
    e_score_correction_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = router_logits.shape[0]
    router_logits = router_logits.to(torch.float)

    normed_score = torch.empty(
        batch_size,
        top_k,
        dtype=torch.float32,
        device=router_logits.device,
    )
    topk_ids = torch.empty(
        batch_size,
        top_k,
        dtype=torch.int32,
        device=router_logits.device,
    )

    if scoring_func == "softmax":
        torch.ops._C.moe_softmax_topk_norm(
            x=router_logits,
            normed_score=normed_score,
            topk_index=topk_ids,
            block_statistic=None,
            stable=False,
        )
    elif scoring_func == "sigmoid":
        block_statistic = torch.zeros(
            12,
            router_logits.shape[-1],
            dtype=torch.int32,
            device=router_logits.device,
        )
        torch.ops._C.moe_sigmoid_group_topk_norm(
            x=router_logits,
            topk_index=topk_ids,
            norm_score=normed_score,
            block_static=block_statistic,
            bias=e_score_correction_bias,
            scale=1.0,
            n_group=num_expert_group,
            topk_group=topk_group,
        )
    else:
        raise ValueError(f"Unsupported scoring_func: {scoring_func}")

    return normed_score, topk_ids


def _build_small_batch_routing(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    block_size_m: int = MOE_WNA16_BLOCK_SIZE_M,
)     -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    flat_topk_ids = topk_ids.reshape(-1).to(torch.int32)
    if flat_topk_ids.numel() == 0:
        return None

    device = topk_ids.device
    numel = flat_topk_ids.numel()
    max_num_tokens_padded = numel + num_local_experts * (block_size_m - 1)
    max_num_m_blocks = (max_num_tokens_padded + block_size_m - 1) // block_size_m

    sorted_token_idx = torch.empty(
        max_num_tokens_padded, dtype=torch.int32, device=device
    )
    expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    sorted_token_pads = torch.empty(1, dtype=torch.int32, device=device)

    torch.ops._C.moe_align_block_size(
        flat_topk_ids,
        num_local_experts,
        block_size_m,
        sorted_token_idx,
        expert_ids,
        sorted_token_pads,
    )

    return sorted_token_idx, expert_ids, sorted_token_pads


def _moe_wna16_gemm(
    x: torch.Tensor,
    output: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    weight_bits: int,
) -> None:
    torch.ops.xspeedgate_ops.moe_wna16_gemm(
        x,
        output,
        qweight,
        scale,
        zp,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        MOE_WNA16_BLOCK_SIZE_M,
        0,
        0,
        weight_bits,
    )


def fused_moe_wna16(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    group_size: int,
    weight_bits: int,
    ep_rank: int,
    top_k: int,
    renormalize: bool,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert ep_rank == 0, "fused_moe_wna16 expects non-EP execution with ep_rank == 0"

    if x.shape[0] * top_k >= SMALL_BATCH_FUSED_MOE_THRESHOLD:
        w13_weight = dequant_awq_moe(
            qweight=w13_qweight,
            scale=w13_scale,
            zp=w13_zp,
            group_size=group_size,
            qweight_is_gemm_aligned=True,
        )
        w2_weight = dequant_awq_moe(
            qweight=w2_qweight,
            scale=w2_scale,
            zp=w2_zp,
            group_size=group_size,
            qweight_is_gemm_aligned=True,
        )
        return ops.fused_moe(
            x,
            w13_weight,
            w2_weight,
            router_logits,
            ep_rank,
            top_k,
            renormalize=renormalize,
            inplace=True,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

    normed_score, topk_ids = _route_moe(
        router_logits=router_logits,
        top_k=top_k,
        scoring_func=scoring_func,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        e_score_correction_bias=e_score_correction_bias,
    )

    routing = _build_small_batch_routing(
        topk_ids=topk_ids,
        num_local_experts=w13_qweight.shape[0],
        block_size_m=MOE_WNA16_BLOCK_SIZE_M,
    )
    if routing is None:
        return torch.zeros(
            x.shape[0],
            w2_qweight.shape[1],
            dtype=x.dtype,
            device=x.device,
        )

    sorted_token_ids, expert_ids, num_tokens_post_padded = routing

    gate_up = torch.zeros(
        x.shape[0],
        top_k,
        w13_qweight.shape[1],
        dtype=x.dtype,
        device=x.device,
    )
    _moe_wna16_gemm(
        x=x,
        output=gate_up,
        qweight=w13_qweight,
        scale=w13_scale,
        zp=w13_zp,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        top_k=top_k,
        weight_bits=weight_bits,
    )

    act = torch.empty(
        x.shape[0],
        top_k,
        w13_qweight.shape[1] // 2,
        dtype=x.dtype,
        device=x.device,
    )
    torch.ops._C.silu_and_mul(act, gate_up)

    out = torch.zeros(
        x.shape[0] * top_k,
        1,
        w2_qweight.shape[1],
        dtype=x.dtype,
        device=x.device,
    )
    _moe_wna16_gemm(
        x=act.reshape(-1, act.shape[-1]),
        output=out,
        qweight=w2_qweight,
        scale=w2_scale,
        zp=w2_zp,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        top_k=1,
        weight_bits=weight_bits,
    )

    return (
        out.view(x.shape[0], top_k, -1)
        .mul(normed_score.to(x.dtype).unsqueeze(-1))
        .sum(dim=1)
        .to(x.dtype)
    )


class KunlunMoeWNA16Method(MoeWNA16Method):

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, _MOE_WNA16_GEMM_ALIGNED_ATTR, False):
            return
        with torch.no_grad():
            layer.w13_qweight = torch.nn.Parameter(
                _align_qweight_to_moe_wna16_gemm(layer.w13_qweight.data),
                requires_grad=False,
            )
            layer.w2_qweight = torch.nn.Parameter(
                _align_qweight_to_moe_wna16_gemm(layer.w2_qweight.data),
                requires_grad=False,
            )
        setattr(layer, _MOE_WNA16_GEMM_ALIGNED_ATTR, True)

    def _get_moe_quant_config(self, layer: torch.nn.Module):
        return self.moe_quant_config or self.get_fused_moe_quant_config(layer)

    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        quant_config = self._get_moe_quant_config(layer)
        assert quant_config is not None
        assert not self.moe.use_ep, "KunlunMoeWNA16Method only supports non-EP mode"
        assert getattr(layer, "w13_bias", None) is None, "KunlunMoeWNA16Method does not support w13_bias"
        assert getattr(layer, "w2_bias", None) is None, "KunlunMoeWNA16Method does not support w2_bias"

        return fused_moe_wna16(
            x=x,
            router_logits=router_logits,
            w13_qweight=layer.w13_qweight,
            w2_qweight=layer.w2_qweight,
            w13_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            w13_zp=quant_config.w1_zp,
            w2_zp=quant_config.w2_zp,
            group_size=layer.group_size,
            weight_bits=self.quant_config.weight_bits,
            ep_rank=self.moe.ep_rank,
            top_k=layer.moe_config.experts_per_token,
            renormalize=layer.renormalize,
            use_grouped_topk=layer.use_grouped_topk,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            scoring_func=layer.scoring_func,
            e_score_correction_bias=layer.e_score_correction_bias,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        quant_config = self._get_moe_quant_config(layer)
        assert quant_config is not None
        assert not self.moe.use_ep, "KunlunMoeWNA16Method only supports non-EP mode"
        assert getattr(layer, "w13_bias", None) is None, "KunlunMoeWNA16Method does not support w13_bias"
        assert getattr(layer, "w2_bias", None) is None, "KunlunMoeWNA16Method does not support w2_bias"

        return fused_moe_wna16(
            x=x,
            router_logits=router_logits,
            w13_qweight=layer.w13_qweight,
            w2_qweight=layer.w2_qweight,
            w13_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            w13_zp=quant_config.w1_zp,
            w2_zp=quant_config.w2_zp,
            group_size=layer.group_size,
            weight_bits=self.quant_config.weight_bits,
            ep_rank=self.moe.ep_rank,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )
