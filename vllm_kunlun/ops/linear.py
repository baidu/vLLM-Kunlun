# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.logger import init_logger

logger = init_logger(__name__)


def get_weights(self):
    """get_weights"""
    if hasattr(self, "kunlun_linear_weights"):
        return self.kunlun_linear_weights
    weights = torch.nn.Parameter(self.weight.to(torch.float32))
    self.register_parameter("kunlun_linear_weights", weights)
    return self.kunlun_linear_weights


def get_weights_half(self):
    """get_weights_half"""
    if hasattr(self, "kunlun_linear_weights_half"):
        return self.kunlun_linear_weights_half
    weights = torch.nn.Parameter(self.weight.to(torch.float16))


ReplicatedLinear.get_weights = get_weights
ReplicatedLinear.get_weights_half = get_weights_half


def create_weights(
    self,
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    weight = Parameter(
        torch.empty(
            sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype
        ),
        requires_grad=False,
    )
    set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
    layer.register_parameter("weight", weight)
    set_weight_attrs(weight, extra_weight_attrs)


# rewrite create_weights and remove weight_loader_v2 to suport cuda graph
UnquantizedLinearMethod.create_weights = create_weights
WEIGHT_LOADER_V2_SUPPORTED.remove("UnquantizedLinearMethod")