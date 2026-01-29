#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
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

from typing import Optional
import torch
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
    CompressedTensorsKVCacheMethod,
    CompressedTensorsLinearTransformMethod,
    get_linear_transform_schemes,
)
from vllm_kunlun.ops.quantization.compressed_tensors.compressed_tensors_moe import (
    KunlunCompressedTensorsMoEMethod,
)


class KunlunCompressedTensorsConfig(CompressedTensorsConfig):
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            # collect schemes
            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)
            input_tfms, output_tfms = get_linear_transform_schemes(
                layer, prefix, self.transform_config, self.packed_modules_mapping
            )

            # choose quantization method
            quant_method: LinearMethodBase = UnquantizedLinearMethod()
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                quant_method = CompressedTensorsLinearMethod(self)

            # choose transform method
            if any((input_tfms, output_tfms)):
                return CompressedTensorsLinearTransformMethod.from_schemes(
                    quant_method, quant_scheme, input_tfms, output_tfms
                )

            else:
                return quant_method

        if isinstance(layer, Attention):
            return CompressedTensorsKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            return KunlunCompressedTensorsMoEMethod.get_moe_method(self, layer)
        return None


# monkey patch
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)

compressed_tensors.CompressedTensorsConfig = KunlunCompressedTensorsConfig
print(
    "[Monkey Patch Applied] >>> vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors.CompressedTensorsConfig \
      --> vllm_kunlun.ops.quantization.compressed_tensors.KunlunCompressedTensorsConfig"
)
