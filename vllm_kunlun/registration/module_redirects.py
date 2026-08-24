"""Upstream vLLM modules that are replaced wholesale by Kunlun versions.

Whenever one of the modules listed in ``MODULE_MAPPINGS`` is about to be
imported, the Kunlun replacement is loaded first and registered in
``sys.modules`` under the upstream name.  Every importer therefore receives
the Kunlun implementation transparently, without any change to vLLM code.

Use a redirect (instead of a post-import patch in ``compat_patches``) when
the whole upstream module needs to be swapped out, not just one or two of
its symbols.  To add one, append an entry to ``MODULE_MAPPINGS``; the
replacement module must be importable on its own and should mirror the
public names of the module it replaces.

The redirect is triggered by the custom import hook in ``import_hooks``;
this module only owns the mapping table and the preload logic.
"""

import importlib
import sys

# Upstream module name -> Kunlun replacement module name.
MODULE_MAPPINGS: dict[str, str] = {
    "vllm.compilation.wrapper": "vllm_kunlun.compilation.wrapper",
    "vllm.model_executor.model_loader.bitsandbytes_loader": (
        "vllm_kunlun.models.model_loader.bitsandbytes_loader"
    ),
    "vllm.v1.sample.ops.topk_topp_sampler": (
        "vllm_kunlun.v1.sample.ops.topk_topp_sampler"
    ),
    "vllm.v1.sample.ops.logprobs": "vllm_kunlun.v1.sample.ops.logprobs",
    "vllm.v1.sample.rejection_sampler": "vllm_kunlun.v1.sample.rejection_sampler",
    "vllm.attention.ops.merge_attn_states": (
        "vllm_kunlun.ops.attention.merge_attn_states"
    ),
    "vllm.v1.worker.mamba_utils": "vllm_kunlun.v1.worker.mamba_utils",
}


def preload_mapped(full_name: str) -> None:
    """Load and alias the Kunlun replacement for one upstream module name.

    The replacement is registered under both the upstream name and its own
    name, so later imports of either resolve to the same module object.
    Nothing is done if the upstream name is already in ``sys.modules``: at
    that point someone has the real upstream module, and silently swapping
    it out from under them would be worse than leaving it alone.
    """
    if full_name in sys.modules:
        return
    replacement = MODULE_MAPPINGS[full_name]
    module = importlib.import_module(replacement)
    sys.modules[full_name] = module
    sys.modules[replacement] = module


def preload_import_mappings(module_name, fromlist) -> None:
    """Preload replacements referenced by one absolute import operation.

    Covers both ``import vllm.x`` (the module itself is mapped) and
    ``from vllm.x import y`` (the submodule ``vllm.x.y`` is mapped).
    """
    if module_name in MODULE_MAPPINGS:
        preload_mapped(module_name)

    for name in fromlist or ():
        full_name = f"{module_name}.{name}"
        if full_name in MODULE_MAPPINGS:
            preload_mapped(full_name)
