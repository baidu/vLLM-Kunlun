from vllm.model_executor.kernels.linear import _POSSIBLE_INT8_KERNELS
from vllm.platforms import PlatformEnum

from .exllama import _POSSIBLE_KERNELS, KunlunExllamaLinearKernel
from .scale_mm import KunlunScaledMMLinearKernel, KunlunW4A8IntLinearKernel

_POSSIBLE_INT8_KERNELS[PlatformEnum.OOT] = [KunlunScaledMMLinearKernel]
_POSSIBLE_KERNELS[PlatformEnum.OOT] = [
    KunlunW4A8IntLinearKernel,
    KunlunExllamaLinearKernel,
]


__all__ = [
    "KunlunW4A8IntLinearKernel",
    "KunlunScaledMMLinearKernel",
    "KunlunExllamaLinearKernel",
    "_POSSIBLE_INT8_KERNELS",
    "_POSSIBLE_KERNELS",
]
