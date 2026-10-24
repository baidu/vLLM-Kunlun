# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse official MTP weights, sharing and forward with the Kunlun layer."""

import vllm.model_executor.models.qwen3_5_mtp as _up

from vllm_kunlun.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm_kunlun.models.qwen3_next import QwenNextMixtureOfExperts

_up.Qwen3_5DecoderLayer = Qwen3_5DecoderLayer
# MoE metadata must recognize the same Kunlun layer classes as the decoder.
_up.Qwen3_5MoeMTP.set_moe_parameters = QwenNextMixtureOfExperts.set_moe_parameters
_up.Qwen3_5MoeMTP.update_physical_experts_metadata = (
    QwenNextMixtureOfExperts.update_physical_experts_metadata
)
