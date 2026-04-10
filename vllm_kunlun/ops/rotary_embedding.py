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
from typing import Any, Optional, Tuple

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


@CustomOp.register_oot(name="Gemma4RotaryEmbedding")
class Gemma4RotaryEmbedding(RotaryEmbedding):
    """Gemma4 proportional RoPE.

    Extends RotaryEmbedding (which provides standard neox-style rotation
    via ops.rotary_embedding CUDA kernel) but overrides the inv_freq
    computation to match HF's _compute_proportional_rope_parameters:
    - Frequency exponents use head_dim (not rotary_dim) as denominator
    - Non-rotated dims are zero-padded (cos=1, sin=0 = identity rotation)

    When partial_rotary_factor=1.0 (the default for some variants), ALL dims are
    rotated and this is equivalent to standard RotaryEmbedding with
    head_dim-scaled frequencies.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        # Number of rotation angle pairs (from partial_rotary_factor)
        self.rope_angles = rotary_dim // 2
        # Non-rotated angle pairs per half
        self.nope_angles = (head_size // 2) - self.rope_angles

        # Important: set rotary_dim = head_size so the base class's
        # forward_static applies rotation to ALL dims of the cos/sin cache.
        # The non-rotated dims will have cos=1, sin=0 (identity) thanks
        # to our _compute_inv_freq zero-padding.
        super().__init__(
            head_size,
            head_size,  # rotary_dim = head_size (full application)
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute frequencies matching HF proportional RoPE.

        Key difference from base: exponent denominator is head_size (not
        rotary_dim), and non-rotated dims are zero-padded.
        """
        # HF formula: base ** (arange(0, 2*rope_angles, 2) / head_dim)
        freq_exponents = (
            torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float) / self.head_size
        )
        inv_freq = 1.0 / (base**freq_exponents)

        # Zero-pad for non-rotated dims (identity rotation: cos=1, sin=0)
        if self.nope_angles > 0:
            inv_freq = torch.cat(
                [
                    inv_freq,
                    torch.zeros(self.nope_angles, dtype=torch.float),
                ]
            )
        return inv_freq

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", rope_angles={self.rope_angles}, nope_angles={self.nope_angles}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


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


_ROPE_DICT: dict[tuple[Any, ...], RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    dual_chunk_attention_config: dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_parameters is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_parameters_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_parameters.items()
        }
        rope_parameters_args = tuple(rope_parameters_tuple.items())
    else:
        rope_parameters_args = None

    if dual_chunk_attention_config is not None:
        dual_chunk_attention_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in dual_chunk_attention_config.items()
            if k != "sparse_attention_config"
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())
    else:
        dual_chunk_attention_args = None

    rope_parameters = rope_parameters or {}
    base = rope_parameters.get("rope_theta", 10000)
    scaling_type = rope_parameters.get("rope_type", "default")
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)

    if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
        raise ValueError(f"{partial_rotary_factor=} must be between 0.0 and 1.0")
    rotary_dim = int(head_size * partial_rotary_factor)

    key = (
        head_size,
        rotary_dim,
        max_position,
        is_neox_style,
        rope_parameters_args,
        dual_chunk_attention_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if scaling_type == "default":
        if "mrope_section" in rope_parameters:
            rotary_emb = MRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                mrope_section=rope_parameters["mrope_section"],
                mrope_interleaved=rope_parameters.get("mrope_interleaved", False),
            )
        else:
            rotary_emb = RotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
            )
    elif scaling_type == "proportional":
        # Proportional RoPE is used by Gemma4 for global (full) attention.
        # Gemma4 uses a sparse/fractional RoPE with cross-mixing between halves.
        rotary_emb = Gemma4RotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
        )
    else:
        raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


# Log that OOT registration is complete
logger.info(
    "[KunlunOOT] Registered KunlunRotaryEmbedding, KunlunMRotaryEmbedding, "
    "KunlunDeepseekScalingRotaryEmbedding via CustomOp.register_oot"
)
