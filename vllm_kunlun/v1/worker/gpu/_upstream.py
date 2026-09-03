# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helper to execute a genuine upstream ``vllm`` module, bypassing the Kunlun
``sys.modules`` swap installed by
``vllm_kunlun/registration/module_redirects.py``.

The V2 model-runner replacement modules under ``vllm_kunlun.v1.worker.gpu.*``
reuse the pure-Python parts of their upstream counterparts (state classes,
constants, dataclasses) and only override the functions / methods that launch
Triton kernels. To get at the genuine upstream code they must load the real
file directly -- a plain ``import vllm.v1.worker.gpu.<leaf>`` would resolve to
the swapped Kunlun module (i.e. themselves) and recurse.
"""

import importlib
import importlib.util
import os
import sys


def load_upstream(vllm_module_name: str):
    """Load and execute the genuine upstream vllm module file.

    Args:
        vllm_module_name: Dotted path of the upstream module, e.g.
            ``"vllm.v1.worker.gpu.buffer_utils"``.

    Returns:
        The executed upstream module object, cached under a private name so
        repeated calls are cheap and idempotent.
    """
    private = "_kunlun_upstream_" + vllm_module_name.replace(".", "_")
    cached = sys.modules.get(private)
    if cached is not None:
        return cached

    parts = vllm_module_name.split(".")
    parent_name = ".".join(parts[:-1])
    leaf = parts[-1]

    # The parent package (e.g. ``vllm.v1.worker.gpu``) is NOT swapped, so this
    # resolves to the real on-disk package and gives us its directory.
    parent = importlib.import_module(parent_name)
    parent_dir = os.path.dirname(parent.__file__)
    file_path = os.path.join(parent_dir, leaf + ".py")

    spec = importlib.util.spec_from_file_location(private, file_path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so any self-reference resolves to this instance.
    sys.modules[private] = module
    spec.loader.exec_module(module)
    return module


def reexport(upstream_module, namespace: dict) -> None:
    """Copy the upstream module's whole public surface into ``namespace``.

    A ``sys.modules`` swap is a *whole module* replacement: the replacement has
    to provide every name any upstream consumer might
    ``from <module> import X``, or the import blows up -- even for features
    Kunlun never intends to support. Hand-maintained re-export lists drift
    silently whenever upstream adds a symbol (this is exactly how
    ``vllm.v1.worker.gpu.model_states.mamba_hybrid`` failed to import
    ``MambaSpecDecodeGPUContext``).

    The set of names upstream can legally import from a module is exactly the
    set of non-dunder keys of its ``__dict__``, so copying that wholesale is
    both correct and drift-proof. Callers must invoke this *before* defining
    their own overrides, so the overrides win in the caller's namespace.
    """
    namespace.update(
        {k: v for k, v in vars(upstream_module).items() if not k.startswith("__")}
    )
