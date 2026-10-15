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

from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationType
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsKVCacheMethod,
    CompressedTensorsLinearMethod,
    CompressedTensorsLinearTransformMethod,
    get_linear_transform_schemes,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_embedding import (  # noqa: E501
    CompressedTensorsEmbeddingWNA16Int,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_kunlun.quantization.utils import _remove_quantization_method

from .compressed_tensors_moe import KunlunCompressedTensorsMoEMethod

# reove the original compressed-tensors quantization methods
_remove_quantization_method("compressed-tensors")


# register the kunlun compressed-tensors quantization methods
@register_quantization_config("compressed-tensors")
class KunlunCompressedTensorsConfig(CompressedTensorsConfig):
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.model_executor.layers.attention import (
            Attention,  # Avoid circular import
        )

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

        if isinstance(layer, ParallelLMHead):
            try:
                quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)
            except ValueError:
                quant_scheme = None
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                return CompressedTensorsLinearMethod(self)

        # ParallelLMHead subclasses VocabParallelEmbedding but is handled above as
        # a linear; only true embedding lookups land here.
        if isinstance(layer, VocabParallelEmbedding):
            scheme_dict = self.get_scheme_dict(layer, layer_name=prefix)
            weight_quant = scheme_dict.get("weights") if scheme_dict else None
            if weight_quant is None:
                return None  # unquantized embedding
            if not (
                isinstance(weight_quant, QuantizationArgs)
                and self._is_wNa16_group_channel(weight_quant, None)
                and weight_quant.type == QuantizationType.INT
            ):
                raise ValueError(
                    "compressed-tensors embeddings only support weight-only INT "
                    f"group/channel (WNA16) quantization, got: {weight_quant}"
                )
            return CompressedTensorsEmbeddingWNA16Int(weight_quant)

        if isinstance(layer, RoutedExperts):
            return KunlunCompressedTensorsMoEMethod.get_moe_method(self, layer, prefix)
        return None
