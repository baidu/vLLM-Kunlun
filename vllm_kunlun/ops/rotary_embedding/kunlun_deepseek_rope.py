#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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
#
"""Kunlun-optimized DeepseekScalingRotaryEmbedding registered via OOT mechanism."""

import logging
from typing import Optional, Tuple

import torch
import xspeedgate_ops  # noqa
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.rotary_embedding import DeepseekScalingRotaryEmbedding

logger = logging.getLogger("vllm_kunlun.ops.rotary_embedding")

_oot_deepseek_rotary_init_logged = False


@CustomOp.register_oot(name="DeepseekScalingRotaryEmbedding")
class KunlunDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    """
    Kunlun-optimized DeepseekScalingRotaryEmbedding registered via OOT mechanism.
    """

    def __init__(self, *args, **kwargs):
        global _oot_deepseek_rotary_init_logged
        super().__init__(*args, **kwargs)
        if not _oot_deepseek_rotary_init_logged:
            logger.info(
                "[KunlunOOT] KunlunDeepseekScalingRotaryEmbedding.__init__ called (OOT instantiation)"
            )
            _oot_deepseek_rotary_init_logged = True

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Kunlun-optimized forward_oot for DeepseekScalingRotaryEmbedding."""
        return torch.ops.xspeedgate_ops.flashinfer_rotary_embedding(
            positions=positions,
            rotary_dim=self.rotary_dim,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox_style=self.is_neox_style,
            query=query,
            key=key,
            offsets=offsets,
        )
