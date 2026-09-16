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
from typing import List, Optional, Tuple

import torch
from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

logger = logging.getLogger("vllm_kunlun.ops.rotary_embedding")

_oot_rotary_init_logged: set = set()


class KunlunRopeForwardMixin:
    """Kunlun fused RoPE forward, reusable by any RotaryEmbedding subclass that
    only changes how ``cos_sin_cache`` is computed.

    The Kunlun kernel takes ``cos_sin_cache`` as a plain argument and always
    applies the same rotation, so a subclass that merely overrides
    ``_compute_inv_freq`` / ``_compute_cos_sin_cache`` can share this forward
    unchanged. Subclasses that override ``forward_native`` apply the rotation
    differently and must not use it -- see ``register_derived_ropes``.

    Must be listed first in the bases, otherwise ``CustomOp.forward_oot``
    (which falls back to the PyTorch-native rotation) wins the MRO.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls_name = type(self).__name__
        if cls_name not in _oot_rotary_init_logged:
            logger.info("[KunlunOOT] %s.__init__ called (OOT instantiation)", cls_name)
            _oot_rotary_init_logged.add(cls_name)

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
                # provide a batched_rotary_embedding kernel. super() resolves
                # past this mixin to the wrapped upstream class.
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


@CustomOp.register_oot(name="RotaryEmbedding")
class KunlunRotaryEmbedding(KunlunRopeForwardMixin, RotaryEmbedding):
    """
    Kunlun-optimized RotaryEmbedding registered via OOT mechanism.

    This class replaces the default RotaryEmbedding when instantiated through
    vLLM's CustomOp registry. When code calls RotaryEmbedding(...), vLLM's
    CustomOp.__new__ checks op_registry_oot and returns KunlunRotaryEmbedding
    instance.
    """


def _iter_subclasses(root: type):
    """Depth-first walk over every (transitive) subclass of *root*."""
    for sub in root.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


def register_derived_ropes() -> List[str]:
    """Register the upstream RotaryEmbedding subclasses that can share the
    Kunlun fused kernel.

    ``CustomOp.__new__`` resolves the OOT registry by exact ``cls.__name__``, so
    registering ``RotaryEmbedding`` does not cover its subclasses. A model whose
    config asks for rope scaling (yarn, llama3, linear, ntk, ...) instantiates
    the upstream subclass, which has no ``forward_oot`` of its own and therefore
    silently falls back to the PyTorch-native rotation -- slow, and an illegal
    memory access on XPU under large positions / high concurrency.

    Most of those subclasses only override ``_compute_inv_freq`` /
    ``_compute_cos_sin_cache``, i.e. they change what goes into the cache, not
    how the rotation is applied. Since ``cos_sin_cache`` is just an argument of
    the Kunlun kernel, they can reuse it as-is. "Does not override
    ``forward_native``" is exactly that condition, and it is checkable at
    runtime, so new upstream subclasses get covered automatically instead of
    waiting for someone to notice the missing registration.

    Subclasses that do override ``forward_native`` (XDRotaryEmbedding takes 4-way
    multimodal positions, FourierRotaryEmbedding has per-token frequencies) are
    left untouched: they keep the upstream native path, which is slow but
    correct. Supporting them needs a hand-written forward_oot, the way
    kunlun_mrope.py and kunlun_deepseek_rope.py do it.
    """
    registered: List[str] = []
    # Snapshot first: registering adds new subclasses to the tree being walked.
    for cls in list(_iter_subclasses(RotaryEmbedding)):
        if cls.__module__.startswith("vllm_kunlun"):
            continue  # our own Kunlun classes show up in the walk too
        if cls.__name__ == "Gemma4RotaryEmbedding":
            # Keep Gemma4 on vLLM's native path until proportional RoPE has
            # been validated against the Kunlun kernel.
            continue
        if cls.__name__ in op_registry_oot:
            continue  # already covered by a hand-written registration
        if cls.forward_native is not RotaryEmbedding.forward_native:
            continue  # applies the rotation differently, kernel does not apply
        CustomOp.register_oot(name=cls.__name__)(
            type(f"Kunlun{cls.__name__}", (KunlunRopeForwardMixin, cls), {})
        )
        registered.append(cls.__name__)
    return registered
