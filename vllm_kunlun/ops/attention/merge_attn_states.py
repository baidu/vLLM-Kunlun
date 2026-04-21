# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from typing import Optional

import torch


def _get_kunlun_ops():
    return importlib.import_module("kunlun_ops")


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    return _get_kunlun_ops().attention_merge_stage(
        prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
    )
