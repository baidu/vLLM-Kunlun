#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# Author: Li Wei, Pan Xiakai, You Zeyu
# Email: liwei157@baidu.com
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
from vllm.logger import init_logger
from typing import Optional
from vllm.model_executor.layers.quantization.awq import AWQLinearMethod
logger = init_logger(__name__)

class KunlunAWQLinearMethod(AWQLinearMethod):
    def repack_int4_for_kunlun(self, packed: torch.Tensor, num_bits: int = 4):
        """Convert AWQ-packed int4 weights to Kunlun XPU format.
        Input:  packed[N, K], dtype=int32, saved as AWQ order
        Output: packed_reordered[N, K], dtype=int32, saved as Kunlun order
        """
        N, K = packed.shape
        self.align_type = 1 if K % 8 == 0 else 0
        assert num_bits == 4, "Only int4 supported now"
        shifts = torch.arange(0, 32, num_bits, device=packed.device, dtype=torch.int32)

        if self.align_type == 0:  # NORMAL MODE
            # Unpack AWQ order:[0, 2, 4, 6, 1, 3, 5, 7]
            unpacked_awq = (packed.unsqueeze(-1) >> shifts) & 0xF  # [N, K, 8]

            # Reverse AWQ order and convert to KUNLUN order
            AWQ_TO_KUNLUN_ORDER_NORMAL = [4, 0, 5, 1, 6, 2, 7, 3]
            # [0,2,4,6,1,3,5,7] --> [1, 0, 3, 2, 5, 4, 7, 6]
            unpacked_kunlun = unpacked_awq[..., AWQ_TO_KUNLUN_ORDER_NORMAL]  # [N, K, 8]

            # Pack to int32, order[6, 7, 4, 5, 2, 3, 0, 1]
            packed_kunlun = (unpacked_kunlun << shifts).sum(
                dim=-1, dtype=torch.int32
            )  # [N, K]
        elif self.align_type == 1:  # FAST MODEL
            # Unpack AWQ order
            unpacked_awq = (
                packed.view(N, K // 8, 8).unsqueeze(-1) >> shifts
            ) & 0xF  # [N, K//8, 8, 8]

            # Reverse AWQ order and convert to KUNLUN order
            AWQ_TO_KUNLUN_ORDER_FAST = [
                32, 0, 36, 4, 33, 1, 37, 5,
                34, 2, 38, 6, 35, 3, 39, 7,
                40, 8, 44, 12, 41, 9, 45, 13,
                42, 10, 46, 14, 43, 11, 47, 15,
                48, 16, 52, 20, 49, 17, 53, 21,
                50, 18, 54, 22, 51, 19, 55, 23,
                56, 24, 60, 28, 57, 25, 61, 29,
                58, 26, 62, 30, 59, 27, 63, 31
            ]
            unpacked_awq = unpacked_awq.reshape(N, K // 8, 64)
            unpacked_kunlun = unpacked_awq[..., AWQ_TO_KUNLUN_ORDER_FAST]  # [N, K//8, 64]

            # Pack to int32
            unpacked_kunlun = unpacked_kunlun.reshape(N, K // 8, 8, 8)
            packed_kunlun = (
                (unpacked_kunlun << shifts).sum(dim=-1, dtype=torch.int32).reshape(N, K)
            )  # [N, K]
        else:
            raise NotImplementedError

        return packed_kunlun


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.warning_once(f"Repacking INT4 for XPU ...")
        layer.qweight = torch.nn.Parameter(
            (
                self.repack_int4_for_kunlun(layer.qweight.data)
                if layer.qweight.data.dtype == torch.int32
                else layer.qweight.data
            ),
            requires_grad=False,
        )
        layer.qzeros = torch.nn.Parameter(
            (
                self.repack_int4_for_kunlun(layer.qzeros.data)
                if layer.qzeros.data.dtype == torch.int32
                else layer.qzeros.data
            ),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)


    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        # num_tokens >= threshold
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256

        if FP16_MATMUL_HEURISTIC_CONDITION:
            out = torch.ops._C.awq_dequantize(
                qweight, scales, qzeros, quant_type=0, align_type=self.align_type
            )
            out = torch.matmul(reshaped_x, out)
        else:
            out = torch.ops._C.awq_gemm(
                reshaped_x, qweight, scales, qzeros, align_type=self.align_type
            )
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)


# monkey patch
from vllm.model_executor.layers.quantization import awq

awq.AWQLinearMethod = KunlunAWQLinearMethod
print(
    "[Monkey Patch Applied] >>> vllm.model_executor.layers.quantization.awq.AWQLinearMethod \
      --> vllm_kunlun.ops.quantization.awq.KunlunAWQLinearMethod"
)
