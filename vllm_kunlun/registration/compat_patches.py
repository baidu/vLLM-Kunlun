"""Built-in post-import compatibility patches for vLLM on Kunlun XPU.

Every patch here exists for the same underlying reason: upstream vLLM
assumes CUDA (or Triton) somewhere, and Kunlun XPU needs a different or
no-op behavior at that spot.

Each patch is a pair of functions operating on the target vLLM module:

- ``*_applied(module)`` reports whether the patch is already in effect.  It
  must also return True when the patch is not needed for this vLLM version,
  so missing attributes are treated as "nothing to do", not as errors.
- ``_apply_*(module)`` performs the patch exactly once.

Two patching styles appear below:

1. Direct: ``_apply_*`` modifies the module attribute itself.
2. Import-for-side-effect: ``_apply_*`` contains only an import statement.
   The imported Kunlun module patches the upstream object when it loads and
   marks it with a ``_kunlun*_patched`` attribute, which the ``*_applied``
   predicate checks.  When you see a lone import, the real patch logic lives
   in that module.

To add a new patch: write an ``*_applied`` / ``_apply_*`` pair following the
conventions above, then append one entry to ``DEFAULT_HOOKS`` at the bottom.
``import_hooks`` registers the table with the dispatcher at import time.
"""

import logging
import sys
from types import ModuleType

# --- vllm.v1.worker.utils: replace KVBlockZeroer --------------------------


def _kv_block_zeroer_applied(module: ModuleType) -> bool:
    """Return whether ``KVBlockZeroer`` is absent or already patched."""
    cls = getattr(module, "KVBlockZeroer", None)
    return cls is None or getattr(cls, "_kunlun_patched", False)


def _apply_kv_block_zeroer(module: ModuleType) -> None:
    """Import the Kunlun module whose side effect patches ``KVBlockZeroer``."""
    if hasattr(module, "KVBlockZeroer"):
        import vllm_kunlun.v1.worker.utils  # noqa: F401


# --- vllm.model_executor.models.qwen3_vl: disable Triton kernels ----------


def _qwen3_vl_applied(module: ModuleType) -> bool:
    """Return whether Triton has already been disabled for Qwen3-VL."""
    return not getattr(module, "HAS_TRITON", False)


def _apply_qwen3_vl_patch(module: ModuleType) -> None:
    """Force the non-Triton code path; Triton kernels cannot run on Kunlun."""
    module.HAS_TRITON = False
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] qwen3_vl HAS_TRITON forced to False"
    )


# --- vllm.v1.worker.block_table: patch slot-mapping computation -----------


def _block_table_applied(module: ModuleType) -> bool:
    """Return whether ``BlockTable`` is absent or has Kunlun slot mapping."""
    cls = getattr(module, "BlockTable", None)
    return cls is None or getattr(cls, "_kunlun_slot_patched", False)


def _apply_block_table_patch(module: ModuleType) -> None:
    """Import the Kunlun ``BlockTable`` implementation for its patch effect."""
    if hasattr(module, "BlockTable"):
        import vllm_kunlun.v1.worker.block_table  # noqa: F401


# --- vllm.v1.structured_output.utils: replace apply_grammar_bitmask -------


def _grammar_bitmask_applied(module: ModuleType) -> bool:
    """Return whether the grammar bitmask helper carries the patch marker."""
    fn = getattr(module, "apply_grammar_bitmask", None)
    return fn is not None and getattr(fn, "_kunlun_patched", False)


def _apply_grammar_bitmask_patch(module: ModuleType) -> None:
    """Import the Kunlun grammar helper, which patches vLLM in place."""
    if hasattr(module, "apply_grammar_bitmask"):
        import vllm_kunlun.v1.structured_output.utils  # noqa: F401


# --- vllm.v1.worker.gpu_worker: skip the CUDA memory-pool context ----------


def _memory_pool_applied(module: ModuleType) -> bool:
    """Return whether the worker memory-pool context is already patched."""
    cls = getattr(module, "Worker", None)
    return cls is None or getattr(cls, "_kunlun_memory_pool_patched", False)


def _apply_memory_pool_patch(module: ModuleType) -> None:
    """Bypass vLLM's CUDA memory-pool context on Kunlun workers.

    CUDA memory pools do not exist on Kunlun devices.  The replacement keeps
    the original behavior on every other platform, so this patch is safe even
    in mixed deployments where the class is shared.
    """
    from contextlib import nullcontext

    original = module.Worker._maybe_get_memory_pool_context

    def patched(self, tag: str):
        from vllm.platforms import current_platform

        if type(current_platform).__name__ == "KunlunPlatform":
            return nullcontext()
        return original(self, tag)

    module.Worker._maybe_get_memory_pool_context = patched
    module.Worker._kunlun_memory_pool_patched = True
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched Worker._maybe_get_memory_pool_context"
    )


# --- vllm.model_executor.warmup.kernel_warmup: skip Triton warmup ----------


def _warmup_applied(module: ModuleType) -> bool:
    """Return whether the unsupported Qwen Triton warmup is disabled."""
    fn = getattr(module, "qwen_triton_warmup", None)
    return fn is not None and getattr(fn, "_kunlun_patched", False)


def _skip_qwen_triton_warmup(*args, **kwargs):
    """No-op replacement: the warmup only compiles Triton kernels, and
    Triton is unavailable on Kunlun."""
    logging.getLogger("vllm_kunlun").info("[KunlunPlugin] Skipping qwen_triton_warmup")


_skip_qwen_triton_warmup._kunlun_patched = True


def _apply_warmup_patch(module: ModuleType) -> None:
    """Replace vLLM's Qwen Triton warmup with the no-op helper."""
    module.qwen_triton_warmup = _skip_qwen_triton_warmup
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched kernel_warmup.qwen_triton_warmup -> no-op"
    )


# --- vllm.model_executor.custom_op: load Kunlun OOT registrations ----------


def _oot_registrations_applied(module: ModuleType) -> bool:
    """Return whether Kunlun out-of-tree operators finished registering."""
    # Without PluggableLayer this vLLM version predates the OOT registration
    # API, so there is nothing to register and the hook reports "done".
    if not hasattr(module, "CustomOp") or not hasattr(module, "PluggableLayer"):
        return True
    ops_module = sys.modules.get("vllm_kunlun.ops")
    return bool(getattr(ops_module, "_KUNLUN_OOT_REGISTRATIONS_LOADED", False))


def _apply_oot_registrations(module: ModuleType) -> None:
    """Import Kunlun operators so their registration decorators execute."""
    import vllm_kunlun.ops  # noqa: F401


# --- compressed_tensors int8 MoE: disable the CUDA backend selector --------


def _int8_moe_applied(module: ModuleType) -> bool:
    """Return whether the compressed-tensors INT8 selector was replaced."""
    if not hasattr(module, "select_int8_moe_backend"):
        return False
    return getattr(module, "_kunlun_select_int8_patched", False)


def _select_int8_moe_backend(config, weight_key=None, activation_key=None):
    """Report no dedicated INT8 MoE backend.

    The upstream selector only knows CUDA backends.  Returning ``(None,
    None)`` steers vLLM to the generic path, where Kunlun's own
    compressed-tensors implementation takes over.
    """
    return None, None


def _apply_int8_moe_patch(module: ModuleType) -> None:
    """Replace vLLM's INT8 MoE backend selector with the Kunlun fallback."""
    if not hasattr(module, "select_int8_moe_backend"):
        return
    module.select_int8_moe_backend = _select_int8_moe_backend
    module._kunlun_select_int8_patched = True


# (target module, is_applied, apply_patch) triples registered by import_hooks.
DEFAULT_HOOKS = (
    ("vllm.v1.worker.utils", _kv_block_zeroer_applied, _apply_kv_block_zeroer),
    (
        "vllm.model_executor.models.qwen3_vl",
        _qwen3_vl_applied,
        _apply_qwen3_vl_patch,
    ),
    ("vllm.v1.worker.block_table", _block_table_applied, _apply_block_table_patch),
    (
        "vllm.v1.structured_output.utils",
        _grammar_bitmask_applied,
        _apply_grammar_bitmask_patch,
    ),
    ("vllm.v1.worker.gpu_worker", _memory_pool_applied, _apply_memory_pool_patch),
    (
        "vllm.model_executor.warmup.kernel_warmup",
        _warmup_applied,
        _apply_warmup_patch,
    ),
    (
        "vllm.model_executor.custom_op",
        _oot_registrations_applied,
        _apply_oot_registrations,
    ),
    (
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
        _int8_moe_applied,
        _apply_int8_moe_patch,
    ),
)
