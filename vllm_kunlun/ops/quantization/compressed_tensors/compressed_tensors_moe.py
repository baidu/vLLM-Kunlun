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

from typing import Callable, Optional, Union

import torch
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsW8A8Int8MoEMethod,
)


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
        num_blocks, global_num_experts, dtype=torch.int32, device=hidden_states.device
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

    out = torch.empty(
        M,
        top_k,
        layer.w2_weight.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    out1 = out1.reshape(-1, out1.shape[-1])
    x_shape = out1.shape
    x_q = torch.empty(x_shape, dtype=torch.int8, device=moe_expand.device)
    x_scale = torch.empty(
        (x_shape[0], 1), dtype=torch.float32, device=moe_expand.device
    )
    torch.ops._C.quant2d(out1, x_q, x_scale, force_sdnn=True)

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

    dequant_scale = torch.ones([M, top_k], dtype=torch.float32, device=out.device)
    output = torch.empty([M, N], dtype=hidden_states.dtype, device=hidden_states.device)
    sorted_tokens_idx = sorted_tokens_idx.view(M, top_k)

    torch.ops._C.moe_post(
        x=out,
        moe_index=sorted_tokens_idx,
        normed_scale=normed_score,
        dequant_scale=dequant_scale,
        y=output,
    )
    return output


CompressedTensorsW8A8Int8MoEMethod.process_weights_after_loading = (
    process_weights_after_loading
)
CompressedTensorsW8A8Int8MoEMethod.apply = apply
