"""Extension point for custom ``get_rope()`` implementations on Kunlun.

All concrete RoPE classes are imported lazily so that this module can be
imported at any time without triggering CustomOp-related circular imports.

Extensibility
-------------
To add a new custom RoPE type, add an entry to ``_ROPE_TYPE_REGISTRY``::

    _ROPE_TYPE_REGISTRY["my_new_type"] = _RopeEntry(
        module="vllm_kunlun.ops.rotary_embedding.my_module",
        cls_name="MyNewRotaryEmbedding",
        build=_build_my_new_rope,
    )

Then any model can use it via::

    from vllm_kunlun.ops.rotary_embedding.rope_ext import get_rope
    self.rotary_emb = get_rope(head_size, max_position, rope_parameters=...)
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Registry infrastructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RopeEntry:
    """Describes how to build a custom RoPE instance for a given rope_type."""

    module: str  # fully-qualified module path
    cls_name: str  # class name inside *module*
    build: Callable  # (cls, head_size, max_position, is_neox_style,
    #  rope_parameters, dtype) -> RotaryEmbedding


def _resolve_cls(entry: _RopeEntry):
    """Import and return the concrete class described by *entry*."""
    mod = importlib.import_module(entry.module)
    return getattr(mod, entry.cls_name)


# ---------------------------------------------------------------------------
# Type registry — add new entries here
# ---------------------------------------------------------------------------

_ROPE_TYPE_REGISTRY: dict[str, _RopeEntry] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_rope(
    head_size,
    max_position,
    is_neox_style=True,
    rope_parameters=None,
    dtype=None,
    **kwargs,
):
    """Drop-in replacement for ``vllm…get_rope()`` with custom type support.

    If *rope_parameters* contains a ``rope_type`` registered in
    ``_ROPE_TYPE_REGISTRY``, the corresponding Kunlun implementation is
    used.  Otherwise falls through to vLLM's original ``get_rope()``.
    """
    if rope_parameters:
        rope_type = rope_parameters.get("rope_type")
        entry = _ROPE_TYPE_REGISTRY.get(rope_type)
        if entry is not None:
            cls = _resolve_cls(entry)
            return entry.build(
                cls, head_size, max_position, is_neox_style, rope_parameters, dtype
            )

    from vllm.model_executor.layers.rotary_embedding import get_rope as _vllm_get_rope

    return _vllm_get_rope(
        head_size, max_position, is_neox_style, rope_parameters, dtype, **kwargs
    )
