#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# Author: Li Wei, You Zeyu
# Email: liwei157@baidu.com, youzeyu@baidu.com
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


from typing import Optional, Union

import torch
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.gptq import (
    ExllamaState,
    GPTQConfig,
    GPTQLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_linear_quant_method,
)

from vllm_kunlun.quantization.utils import _remove_quantization_method

logger = init_logger(__name__)


# reove the original gptq quantization method
_remove_quantization_method("gptq")


# register the kunlun gptq quantization method
@register_quantization_config("gptq")
class KunlunGPTQConfig(GPTQConfig):
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["GPTQLinearMethod", "QuantizeMethodBase"] | None:
        if isinstance(layer, FusedMoE):
            from .moe_wna16 import MoeWNA16Config

            config = {
                "quant_method": "gptq",
                "bits": self.weight_bits,
                "group_size": self.group_size,
                "sym": True,  # GPTQ typically uses symmetric quantization
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)

        return get_linear_quant_method(self, layer, prefix, KunlunGPTQLinearMethod)


class KunlunGPTQLinearMethod(GPTQLinearMethod):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # for torch.compile
        logger.warning_once("Repacking INT4 for XPU ...")
        layer.qzeros = Parameter(
            (
                self.repack_int4_for_kunlun(
                    layer.qzeros.data, self.quant_config.weight_bits
                )
                if self.quant_config.weight_bits == 4
                else layer.qzeros.data
            ),
            requires_grad=False,
        )
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.g_idx = Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)

        # exllama needs to shuffle the weight after the weight is loaded
        # here we do the shuffle on first forward pass
        if layer.exllama_state == ExllamaState.UNINITIALIZED:
            if self.quant_config.desc_act:
                layer.g_idx.data = torch.argsort(layer.g_idx).to(torch.int)
            else:
                layer.g_idx.data = torch.empty(
                    (0,), dtype=torch.int, device=layer.g_idx.device
                )
            layer.exllama_state = ExllamaState.READY

            # No need shuffle on xpu
            # ops.gptq_shuffle(layer.qweight, layer.g_idx,
            #                  self.quant_config.weight_bits)

    def repack_int4_for_kunlun(self, packed: torch.Tensor, num_bits: int = 4):
        N, K = packed.shape
        assert num_bits == 4, "Only int4 supported now"
        shifts = torch.arange(0, 32, num_bits, device=packed.device, dtype=torch.int32)

        # Unpack int32 to int4 values
        unpacked_gptq = (
            packed.view(N, K // 8, 8).unsqueeze(-1) >> shifts
        ) & 0xF  # [N, K//8, 8, 8]

        # Convert to KUNLUN order
        GPTQ_TO_KUNLUN_ORDER_FAST = [i + d for i in range(32) for d in [32, 0]]
        unpacked_gptq = unpacked_gptq.reshape(N, K // 8, 64)
        unpacked_kunlun = unpacked_gptq[..., GPTQ_TO_KUNLUN_ORDER_FAST]  # [N, K//8, 64]

        # Pack to int32
        unpacked_kunlun = unpacked_kunlun.reshape(N, K // 8, 8, 8)
        packed_kunlun = (
            (unpacked_kunlun << shifts).sum(dim=-1, dtype=torch.int32).reshape(N, K)
        )  # [N, K]

        return packed_kunlun

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.qweight.shape[-1],)
        reshaped_x = x.reshape(-1, x.shape[-1])

        output = torch.ops.xspeedgate_ops.gptq_gemm(
            reshaped_x,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            layer.g_idx,
            layer.exllama_state == ExllamaState.READY,
            self.quant_config.weight_bits,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
