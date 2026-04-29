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
"""Kunlun-optimized MRotaryEmbedding (Multi-modal RoPE) registered via OOT mechanism."""

import logging
from typing import Optional, Tuple

import torch
import xspeedgate_ops  # noqa
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

logger = logging.getLogger("vllm_kunlun.ops.rotary_embedding")

_oot_mrotary_init_logged = False


@CustomOp.register_oot(name="MRotaryEmbedding")
class KunlunMRotaryEmbedding(MRotaryEmbedding):
    """
    Kunlun-optimized MRotaryEmbedding (Multi-modal RoPE) registered via OOT mechanism.
    """

    def __init__(self, *args, **kwargs):
        global _oot_mrotary_init_logged
        super().__init__(*args, **kwargs)
        if not _oot_mrotary_init_logged:
            logger.info(
                "[KunlunOOT] KunlunMRotaryEmbedding.__init__ called (OOT instantiation)"
            )
            _oot_mrotary_init_logged = True

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Kunlun-optimized forward_oot for MRotaryEmbedding."""
        assert positions.ndim == 2
        assert key is not None

        query, key = torch.ops.xspeedgate_ops.mrotary_embedding_fwd_v0(
            query,
            key,
            positions.to(dtype=torch.int32),
            self.cos_sin_cache,
            self.mrope_interleaved,
            self.is_neox_style,
            self.head_size,
            self.rotary_dim,
            self.mrope_section[0],
            self.mrope_section[1],
            self.mrope_section[2],
        )

        return query, key
