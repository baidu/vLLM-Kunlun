from vllm.model_executor.kernels.linear import (
    _POSSIBLE_FP8_BLOCK_KERNELS,
    _POSSIBLE_FP8_KERNELS,
    _POSSIBLE_INT8_KERNELS,
)
from vllm.platforms import PlatformEnum

from .exllama import _POSSIBLE_KERNELS, KunlunExllamaLinearKernel
from .fp8 import KunlunFP8ScaledMMLinearKernel, KunlunFp8BlockScaledMMKernel
from .scale_mm import KunlunScaledMMLinearKernel

_POSSIBLE_INT8_KERNELS[PlatformEnum.OOT] = [KunlunScaledMMLinearKernel]
_POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = [KunlunFP8ScaledMMLinearKernel]
_POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT] = [KunlunFp8BlockScaledMMKernel]


__all__ = [
    "KunlunFP8ScaledMMLinearKernel",
    "KunlunFp8BlockScaledMMKernel",
    "KunlunScaledMMLinearKernel",
    "KunlunExllamaLinearKernel",
    "_POSSIBLE_INT8_KERNELS",
    "_POSSIBLE_KERNELS",
]
