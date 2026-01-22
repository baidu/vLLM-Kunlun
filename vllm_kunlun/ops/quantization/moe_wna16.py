#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# Author: Tang Shiwen, Li Wei
# Email: tangshiwen@baidu.com, liwei157@baidu.com
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

import torch
from typing import Optional, Callable, Union

from vllm.distributed import get_tp_group
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method
from vllm.model_executor.utils import set_weight_attrs

from vllm_kunlun.ops.quantization.kernels.quant_ops import dequant_int4
from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops


def convert_awq_tensor_for_kunlun(
    packed: torch.Tensor,
    tensor_type: str,
    num_bits: int = 4,
    align_type: int = 0,
):
    """
    Convert AWQ-packed int4 weights to Kunlun XPU format.
    Input: packed[N, K], dtype=int32, saved as AWQ order
    Output:
        weight: packed_reordered[N, K*4], dtype=int8, saved as Kunlun order
        zeros: zeros_reordered[N, K*8], dtype=float16
    """
    N, K = packed.shape
    assert num_bits == 4, "Only int4 supported now"
    shifts_from_int32 = torch.arange(
        0, 32, num_bits, device=packed.device, dtype=torch.int32
    )
    shifts_back_int8 = torch.arange(
        0, 8, num_bits, device=packed.device, dtype=torch.int32
    )

    if tensor_type == "qweight":  # pack weight

        if align_type == 0:  # normal mode
            # Unpack AWQ order:[0, 2, 4, 6, 1, 3, 5, 7]
            unpacked_awq = (packed.unsqueeze(-1) >> shifts_from_int32) & 0xF
            AWQ_TO_KUNLUN_ORDER_NORMAL = [0, 4, 1, 5, 2, 6, 3, 7]
            unpacked_kunlun = unpacked_awq[..., AWQ_TO_KUNLUN_ORDER_NORMAL]
            shifts_back_int8 = shifts_back_int8.repeat(4)

        elif align_type == 1:  # fast mode
            # Unpack AWQ order: [0, 2, 4, ..., 123, 125, 127]
            unpacked_awq = (
                packed.view(N, K // 16, 16).unsqueeze(-1) >> shifts_from_int32
            ) & 0xF
            unpacked_awq = unpacked_awq.reshape(N, K // 16, 128)
            # Reverse AWQ order and convert to KUNLUN order
            AWQ_TO_KUNLUN_ORDER_FAST = [
                j + 8 * i
                for i in range(8)
                for j in [0, 64, 4, 68, 1, 65, 5, 69, 2, 66, 6, 70, 3, 67, 7, 71]
            ]
            unpacked_kunlun = unpacked_awq[..., AWQ_TO_KUNLUN_ORDER_FAST]
            shifts_back_int8 = shifts_back_int8.repeat(64)

        else:
            raise NotImplementedError

        # Pack to int8, order[1, 0]
        packed_kunlun = (
            (unpacked_kunlun << shifts_back_int8)
            .view(*unpacked_kunlun.shape[:-1], -1, 2)
            .sum(dim=-1)
            .to(torch.int8)
            .reshape(N, -1)
        )

    elif tensor_type == "qzeros":  # pack zero points
        unpacked_awq = (packed.unsqueeze(-1) >> shifts_from_int32) & 0xF
        AWQ_TO_NORMAL_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
        unpacked_kunlun = unpacked_awq[..., AWQ_TO_NORMAL_ORDER]
        shifts_back_int8 = shifts_back_int8.repeat(4)
        packed_kunlun = (
            (unpacked_kunlun << shifts_back_int8)
            .view(*unpacked_kunlun.shape[:-1], -1, 2)
            .sum(dim=-1)
            .to(torch.uint8)
            .reshape(N, -1)
        )

    else:
        raise NotImplementedError()

    return packed_kunlun.T.contiguous()


class KunlunMoeWNA16Method(MoeWNA16Method):

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

        wrapped_weight_loader = type(self).get_weight_loader(
            layer, extra_weight_attrs["weight_loader"]
        )
        extra_weight_attrs["weight_loader"] = wrapped_weight_loader

        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2
                * intermediate_size_per_partition
                // self.quant_config.bit8_pack_factor,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.bit8_pack_factor,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

    @staticmethod
    def get_weight_loader(layer, weight_loader):

        def patched_moe_wna16_weight_loader(
            param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
        ):

            if "g_idx" in weight_name:
                return False if return_success else None
            if not layer.quant_config.has_zp and "qzeros" in weight_name:
                return False if return_success else None

            device = get_tp_group().device
            loaded_weight = loaded_weight.to(device)

            orig_method = layer.quant_config.linear_quant_method

            if layer.quant_config.linear_quant_method == "awq":
                assert layer.quant_config.weight_bits == 4

                if "weight" in weight_name:

                    # TODO(hack): Temporary workaround for a packing conflict between
                    # dequant_int4 and tensor-parallel (TP) sharding. When align_type=1,
                    # the weights cannot be packed correctly after TP slicing, leading
                    # to invalid packed values. This should be revisited once the
                    # sharding/packing logic is refactored.
                    layer.align_type = 0

                    loaded_weight = convert_awq_tensor_for_kunlun(
                        packed=loaded_weight,
                        tensor_type="qweight",
                        align_type=layer.align_type,
                    )
                elif "zeros" in weight_name:
                    loaded_weight = convert_awq_tensor_for_kunlun(
                        packed=loaded_weight, tensor_type="qzeros", align_type=0
                    )
                else:
                    loaded_weight = loaded_weight.T

                layer.quant_config.linear_quant_method = "_patched_awq"

            try:
                return MoeWNA16Method.get_weight_loader(layer, weight_loader)(
                    param,
                    loaded_weight,
                    weight_name,
                    shard_id,
                    expert_id,
                    return_success=return_success,
                )
            finally:
                layer.quant_config.linear_quant_method = orig_method

        return patched_moe_wna16_weight_loader

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

        w13_weight = dequant_int4(
            qweight=layer.w13_qweight,
            scale=self.moe_quant_config.w1_scale,
            zp=self.moe_quant_config.w1_zp,
            int4_signed=False,
            use_mode_fast=layer.align_type,
        )

        w2_weight = dequant_int4(
            qweight=layer.w2_qweight,
            scale=self.moe_quant_config.w2_scale,
            zp=self.moe_quant_config.w2_zp,
            int4_signed=False,
            use_mode_fast=layer.align_type,
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


from vllm.model_executor.layers.quantization import moe_wna16

moe_wna16.MoeWNA16Method = KunlunMoeWNA16Method
print(
    "[Monkey Patch Applied] >>> vllm.model_executor.layers.quantization.moe_wna16.MoeWNA16Method \
      --> vllm_kunlun.ops.quantization.moe_wna16.KunlunMoeWNA16Method"
)
