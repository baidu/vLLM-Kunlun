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
        """Kunlun-optimized forward_oot using the Kunlun RoPE kernel.

        [20260629] Un-bypassed (item ⑤). The previous body short-circuited to
        super().forward_native(...) as a debug measure ("test if XPU kernel is
        garbage root cause"). The kernel was re-verified against an fp32-accurate
        partial-rotary reference (head_dim=128, rotary_dim=64, neox) in
        test/test_rope_kernel.py: kernel-vs-fp32 mean_abs ~3e-4, i.e. no worse
        (slightly better) than the native bf16 path it replaces. Same kernel runs
        un-bypassed in M2.7 production. Details mirror M2.7's forward_oot:
        match cos_sin_cache dtype/device once outside torch.compile tracing,
        squeeze a 4D cache to 2D, and make it contiguous before the kernel call.
        """
        from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops

        cos_sin_cache = self.cos_sin_cache
        if (
            cos_sin_cache.device != query.device
            or cos_sin_cache.dtype != query.dtype
        ):
            cos_sin_cache = cos_sin_cache.to(query.device, dtype=query.dtype)
            if not torch.compiler.is_compiling():
                self.cos_sin_cache = cos_sin_cache

        # The kernel expects a 2D cache [max_pos, rot_dim]; some build paths
        # produce [1, 1, max_pos, rot_dim]. Squeeze defensively (M3 default is 2D).
        if cos_sin_cache.ndim == 4:
            cos_sin_cache = cos_sin_cache.squeeze(0).squeeze(0)
        cos_sin_cache = cos_sin_cache.contiguous()

        # KunlunOps.rotary_embedding requires a real key tensor. Native RoPE
        # supports key=None, but silently falling back would bypass the XPU op.
        if key is None:
            raise NotImplementedError(
                "KunlunRotaryEmbedding.forward_oot requires key for the XPU "
                "rotary_embedding kernel; key=None is not wired to a Kunlun op."
            )

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
                    cos_sin_cache,
                    self.is_neox_style,
                    self.rotary_dim,
                    offsets,
                )
            else:
                raise NotImplementedError(
                    "KunlunRotaryEmbedding.forward_oot received offsets, but "
                    "KunlunOps.batched_rotary_embedding is not available. "
                    "Refusing to silently bypass the XPU op."
                )
        else:
            query, key = ops.rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                cos_sin_cache,
                self.is_neox_style,
            )
        return query, key
