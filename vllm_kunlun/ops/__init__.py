# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Unified registration entry point for Kunlun-optimized operators.

This package initializer performs two kinds of registration when the vLLM
Kunlun plugin is loaded:

* It ensures that the ``torch.library`` module defining custom operators has
  been loaded.
* It imports Python modules containing ``CustomOp.register_oot`` decorators,
  adding Kunlun implementations to vLLM's out-of-tree (OOT) registry.

These imports have registration side effects; the imported module names are
only retained to satisfy unused-import checks. The initializer is kept small
because every ``vllm_kunlun.ops.<submodule>`` import executes it first. Adding
unrelated imports here could introduce circular dependencies and unnecessary
startup work, so only modules containing registration logic are imported.
"""

import sys

# The plugin startup path normally loads the registration module directly
# before importing this package. This fallback handles direct
# ``import vllm_kunlun.ops`` calls and ensures that custom operator schemas
# and implementations are available. The sys.modules check prevents them
# from being registered more than once.
if "_vllm_kunlun_custom_ops_registration" not in sys.modules:
    from . import _custom_ops as _custom_ops  # noqa: F401

# These imports execute the OOT registration decorators in each module,
# registering Kunlun implementations for fused MoE, LayerNorm, Linear,
# Rotary Embedding, and vocabulary-parallel Embedding. The aliases are
# intentionally private because this package exposes the registration
# results rather than the module objects themselves.
from . import activation as _activation  # noqa: E402,F401
from . import fused_moe as _fused_moe  # noqa: F401
from . import layernorm as _layernorm  # noqa: E402,F401
from . import linear as _linear  # noqa: E402,F401
from . import rotary_embedding as _rotary_embedding  # noqa: E402,F401
from . import vocab_parallel_embedding as _vocab_parallel_embedding  # noqa: E402,F401

# Set only after every registration module has finished importing. The plugin
# import hook uses this sentinel to distinguish a complete package from a
# partially initialized module in sys.modules.
_KUNLUN_OOT_REGISTRATIONS_LOADED = True

# No public names need to be exposed through
# ``from vllm_kunlun.ops import *``; operators and OOT implementations are
# looked up by name in their respective registries.
__all__ = []
