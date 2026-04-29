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
"""Kunlun-optimized RotaryEmbedding registered via OOT mechanism."""

import logging
from typing import Optional, Tuple

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

logger = logging.getLogger("vllm_kunlun.ops.rotary_embedding")

_oot_rotary_init_logged = False


@CustomOp.register_oot(name="RotaryEmbedding")
class KunlunRotaryEmbedding(RotaryEmbedding):
    """
    Kunlun-optimized RotaryEmbedding registered via OOT mechanism.

    This class replaces the default RotaryEmbedding when instantiated through
    vLLM's CustomOp registry. When code calls RotaryEmbedding(...), vLLM's
    CustomOp.__new__ checks op_registry_oot and returns KunlunRotaryEmbedding instance.
    """

    def __init__(self, *args, **kwargs):
        global _oot_rotary_init_logged
        super().__init__(*args, **kwargs)
        if not _oot_rotary_init_logged:
            logger.info(
                "[KunlunOOT] KunlunRotaryEmbedding.__init__ called (OOT instantiation)"
            )
            _oot_rotary_init_logged = True

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Kunlun-optimized forward_oot using Kunlun RoPE kernels."""
        from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops

        if (
            self.cos_sin_cache.device != query.device
            or self.cos_sin_cache.dtype != query.dtype
        ):
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)

        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            batched_rotary = getattr(ops, "batched_rotary_embedding", None)
            if batched_rotary is not None:
                batched_rotary(
                    positions,
                    query,
                    key,
                    self.head_size,
                    self.cos_sin_cache,
                    self.is_neox_style,
                    self.rotary_dim,
                    offsets,
                )
            else:
                # Fallback to the base implementation when Kunlun does not
                # provide a batched_rotary_embedding kernel.
                return super().forward_native(
                    positions,
                    query,
                    key,
                    offsets=offsets,
                )
        else:
            query, key = ops.rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        return query, key
