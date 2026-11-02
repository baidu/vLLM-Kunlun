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
from compressed_tensors.quantization import QuantizationArgs
from vllm.config import get_current_vllm_config
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
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
    CompressedTensorsWNA16,
)
from vllm.scalar_type import scalar_types

from vllm_kunlun.quantization.utils import _remove_quantization_method


def validate_int4_format(
    weight: QuantizationArgs | None, quant_format: str | None
) -> None:
    """Reject unpacked INT4 before selecting a Dense or MoE scheme."""
    if (
        weight is not None
        and weight.type == "int"
        and weight.num_bits == 4
        and quant_format != "pack-quantized"
    ):
        raise ValueError(
            "Kunlun INT4 weights require pack-quantized checkpoints; "
            f"got {quant_format!r}."
        )


def validate_w4a8_int(
    weight: QuantizationArgs,
    activation: QuantizationArgs,
    output: QuantizationArgs | None = None,
) -> None:
    """Check Kunlun constraints after upstream W4A8 scheme matching.

    Upstream checks bit widths, weight symmetry and dynamic per-token inputs.
    Symmetric activations remain a Kunlun requirement.
    """
    if not (
        weight.type == "int"
        and weight.strategy in ("channel", "group")
        and weight.actorder is None
        and activation.type == "int"
        and activation.symmetric
        and output is None
    ):
        raise ValueError(
            "Kunlun W4A8 requires symmetric per-channel or grouped INT4 "
            "weights and dynamic symmetric per-token INT8 activations, "
            "without activation ordering or output quantization."
        )


# reove the original compressed-tensors quantization methods
_remove_quantization_method("compressed-tensors")


# register the kunlun compressed-tensors quantization methods
@register_quantization_config("compressed-tensors")
class KunlunCompressedTensorsConfig(CompressedTensorsConfig):
    def _get_scheme_from_parts(
        self,
        weight_quant,
        input_quant,
        output_quant=None,
        format=None,
        layer_name=None,
    ):
        quant_format = self.quant_format if format is None else format
        validate_int4_format(weight_quant, quant_format)
        if input_quant is not None and self._is_dynamic_token_w4a8_int(
            weight_quant, input_quant
        ):
            validate_w4a8_int(
                weight_quant,
                input_quant,
                output_quant,
            )
            if (
                weight_quant.strategy == "group"
                and not get_current_vllm_config().model_config.is_moe
            ):
                raise ValueError(
                    "Kunlun W4A8 group quantization is supported only for MoE "
                    "models; Dense models require per-channel weights "
                    f"(layer {layer_name!r})."
                )
            return KunlunCompressedTensorsW4A8IntPacked(weight_quant, layer_name)
        return super()._get_scheme_from_parts(
            weight_quant, input_quant, output_quant, format, layer_name
        )

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
        if isinstance(layer, RoutedExperts):
            from .compressed_tensors_moe import KunlunCompressedTensorsMoEMethod

            return KunlunCompressedTensorsMoEMethod.get_moe_method(self, layer, prefix)
        return None


class KunlunCompressedTensorsW4A8IntPacked(CompressedTensorsWNA16):
    """Reuse the upstream packed loader with Kunlun's dynamic INT8 GEMM."""

    def __init__(self, weight_quant, layer_name=None):
        super().__init__(
            strategy=weight_quant.strategy,
            num_bits=weight_quant.num_bits,
            group_size=weight_quant.group_size,
            layer_name=layer_name,
        )
        # Select the W4A8 kernel, not the weight-only uint4b8 kernels.
        self.quant_type = scalar_types.int4

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        # Preserve checkpoint scale values without changing the activation dtype
        # or the upstream parameter's TP-aware loading metadata.
        layer.weight_scale.data = torch.empty_like(
            layer.weight_scale.data, dtype=torch.float32
        )

    @classmethod
    def get_min_capability(cls) -> int:
        return 1
