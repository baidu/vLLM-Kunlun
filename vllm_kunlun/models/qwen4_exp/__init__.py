# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp model package."""

from typing import TYPE_CHECKING, Any

from .common.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

if TYPE_CHECKING:
    from .nvidia.model import Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration
    from .nvidia.mtp import Qwen4ExpMTP


def __getattr__(name: str) -> Any:
    if name in {
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpMTP",
    }:
        # The subpackage is still named ``nvidia`` because it is a port of the
        # reference CUDA implementation; on Kunlun the ops inside it are routed
        # to kunlun_ops / XSpeedGate. The reference layout also has an ``amd``
        # variant, which is not ported here. Imported lazily so that merely
        # registering the architecture does not pull in the decoder stack.
        from .nvidia.model import Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration
        from .nvidia.mtp import Qwen4ExpMTP

        return {
            "Qwen4ExpForCausalLM": Qwen4ExpForCausalLM,
            "Qwen4ExpForConditionalGeneration": (Qwen4ExpForConditionalGeneration),
            "Qwen4ExpMTP": Qwen4ExpMTP,
        }[name]
    raise AttributeError(name)


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMTP",
]
