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
"""
Kunlun-optimized Rotary Embedding implementations using vLLM's CustomOp.register_oot mechanism.

Design:
- Uses @CustomOp.register_oot to register Kunlun-optimized RotaryEmbedding classes
- These classes automatically replace the default implementations when instantiated
- Since KunlunPlatform uses _enum=PlatformEnum.OOT, dispatch_forward() selects
  forward_oot, so we implement forward_oot

OOT Mechanism:
- When code calls RotaryEmbedding(...), vLLM's CustomOp.__new__ checks op_registry_oot
- If "RotaryEmbedding" is found in OOT registry, it returns KunlunRotaryEmbedding instance instead
- This is the official vLLM way to replace operators without modifying source code
"""

import logging
from typing import Optional, Tuple

import torch
import xspeedgate_ops  # noqa
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    MRotaryEmbedding,
    RotaryEmbedding,
)

logger = logging.getLogger("vllm_kunlun.ops.rotary_embedding")

# Track if OOT classes have logged (for logging once per type)
_oot_rotary_init_logged = False
_oot_mrotary_init_logged = False
_oot_deepseek_rotary_init_logged = False


# =============================================================================
# OOT-registered Kunlun RotaryEmbedding classes
# =============================================================================


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


# =============================================================================
# Utility functions (kept for compatibility)
# =============================================================================


def Split_Norm_Rope(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    max_position_embeddings: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Split + Norm + RoPE operation."""
    num_tokens = qkv.shape[0]
    rotary_dim = head_dim
    q_emb_out = torch.empty(
        (num_tokens, q_head_num * head_dim), dtype=qkv.dtype, device=qkv.device
    )
    k_emb_out = torch.empty(
        (num_tokens, kv_head_num * head_dim), dtype=qkv.dtype, device=qkv.device
    )
    v_out = torch.empty(
        (num_tokens, kv_head_num * head_dim), dtype=qkv.dtype, device=qkv.device
    )
    torch.ops._C.split_norm_rope_neox(
        q_emb_out,
        k_emb_out,
        v_out,
        qkv,
        cos_sin_cache,
        q_norm_weight,
        k_norm_weight,
        positions,
        num_tokens,
        max_position_embeddings,
        q_head_num,
        kv_head_num,
        head_dim,
        rotary_dim,
    )
    return q_emb_out, k_emb_out, v_out


# Log that OOT registration is complete
logger.info(
    "[KunlunOOT] Registered KunlunRotaryEmbedding, KunlunMRotaryEmbedding, "
    "KunlunDeepseekScalingRotaryEmbedding via CustomOp.register_oot"
)
