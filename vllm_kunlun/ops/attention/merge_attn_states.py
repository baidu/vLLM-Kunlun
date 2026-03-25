# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import kunlun_ops
import torch


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:

    return kunlun_ops.attention_merge_stage(
        prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
    )
