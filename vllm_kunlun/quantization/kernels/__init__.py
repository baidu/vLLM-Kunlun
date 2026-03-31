from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    _POSSIBLE_INT8_KERNELS,
)
from vllm.platforms import PlatformEnum

from .exllama import _POSSIBLE_KERNELS, KunlunExllamaLinearKernel
from .scale_mm import KunlunScaledMMLinearKernel

_POSSIBLE_INT8_KERNELS[PlatformEnum.OOT] = [KunlunScaledMMLinearKernel]


__all__ = [
    "KunlunScaledMMLinearKernel",
    "KunlunExllamaLinearKernel",
    "_POSSIBLE_INT8_KERNELS",
    "_POSSIBLE_KERNELS",
]
