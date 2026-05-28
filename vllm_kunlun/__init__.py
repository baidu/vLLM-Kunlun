"""vllm kunlun init"""

import builtins
import importlib
import logging
import os
import sys

from vllm.logger import init_logger as init_vllm_logger

OLD_IMPORT_HOOK = builtins.__import__


def _configure_kunlun_logger() -> logging.Logger:
    """Reuse vLLM's handler for the vllm_kunlun logger tree."""
    vllm_logger = init_vllm_logger("vllm")
    kunlun_logger = logging.getLogger("vllm_kunlun")

    if not kunlun_logger.handlers:
        for handler in vllm_logger.handlers:
            kunlun_logger.addHandler(handler)

    kunlun_logger.setLevel(vllm_logger.getEffectiveLevel())
    kunlun_logger.propagate = False
    return kunlun_logger


# Re-entry sentinel for the post-import hooks dispatcher. Some hooks
# trigger their own imports (e.g. importing ``vllm_kunlun.v1.worker.utils``
# to apply the KVBlockZeroer patch), which would re-enter
# ``_custom_import`` recursively. A single dispatcher-level guard is
# sufficient because all hooks are idempotent and we only need one to
# run per real import event.
_POST_IMPORT_DISPATCH_IN_PROGRESS = {"v": False}


_MODULE_MAPPINGS = {
    "vllm.compilation.wrapper": "vllm_kunlun.compilation.wrapper",
    "vllm.model_executor.model_loader.bitsandbytes_loader": "vllm_kunlun.models.model_loader.bitsandbytes_loader",
    "vllm.v1.sample.ops.topk_topp_sampler": "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
    "vllm.v1.sample.ops.logprobs": "vllm_kunlun.v1.sample.ops.logprobs",
    "vllm.v1.sample.rejection_sampler": "vllm_kunlun.v1.sample.rejection_sampler",
    "vllm.v1.spec_decode.dflash": "vllm_kunlun.v1.sample.spec_decode.dflash",
    "vllm.attention.ops.merge_attn_states": "vllm_kunlun.ops.attention.merge_attn_states",
    # "vllm.model_executor.models.config": "vllm_kunlun.models.config",
    # "vllm.v1.worker.mamba_utils": "vllm_kunlun.v1.worker.mamba_utils",
    # "vllm.v1.worker.gpu_model_runner": "vllm_kunlun.v1.worker.gpu_model_runner",
}


# ---------------------------------------------------------------------------
# Post-import hook registry
# ---------------------------------------------------------------------------
# Each entry: (target_module_name, applied_predicate, apply_callable).
#
#   target_module_name  upstream module that must be loaded for this hook
#                       to be applicable. The hook only runs after this
#                       module appears in ``sys.modules``.
#   applied_predicate   ``fn(module) -> bool``. Return True if the patch
#                       has already been applied (cheap, side-effect free).
#                       Used both for idempotency and to short-circuit
#                       once the hook has succeeded.
#   apply_callable      ``fn(module) -> None``. Performs the actual
#                       patch. Must set its own "applied" sentinel so
#                       ``applied_predicate`` returns True afterwards.
#
# To add a new hook: write the apply function (in a dedicated module if
# non-trivial; inline lambda for one-liners), then append a tuple here.
# ---------------------------------------------------------------------------
_POST_IMPORT_HOOKS: list = []


def _register_post_import_hook(target, applied, apply):
    _POST_IMPORT_HOOKS.append((target, applied, apply))


def _dispatch_post_import_hooks():
    """Run every registered post-import hook whose target is loaded.

    Re-entrant safe: importing the kunlun replacement module from within
    a hook re-triggers ``_custom_import`` -> this dispatcher; the
    in-progress sentinel short-circuits the inner call.
    """
    if _POST_IMPORT_DISPATCH_IN_PROGRESS["v"]:
        return
    _POST_IMPORT_DISPATCH_IN_PROGRESS["v"] = True
    try:
        for target, applied, apply in _POST_IMPORT_HOOKS:
            mod = sys.modules.get(target)
            if mod is None:
                continue
            try:
                if applied(mod):
                    continue
                apply(mod)
            except Exception:
                logging.getLogger("vllm_kunlun").exception(
                    "[KunlunPlugin] post-import hook failed for target=%s", target
                )
    finally:
        _POST_IMPORT_DISPATCH_IN_PROGRESS["v"] = False


# --- hook 1: KVBlockZeroer in vllm.v1.worker.utils ------------------------
# Importing the kunlun replacement module triggers an in-place class
# patch (``_kunlun_patched`` flag set on KVBlockZeroer). See
# ``vllm_kunlun/v1/worker/utils.py`` for the actual patch body.
def _kvblockzeroer_applied(mod):
    cls = getattr(mod, "KVBlockZeroer", None)
    return cls is None or getattr(cls, "_kunlun_patched", False)


def _kvblockzeroer_apply(mod):
    if not hasattr(mod, "KVBlockZeroer"):
        return  # upstream module loaded before its class body executed
    import vllm_kunlun.v1.worker.utils  # noqa: F401  (self-applies on import)


_register_post_import_hook(
    "vllm.v1.worker.utils", _kvblockzeroer_applied, _kvblockzeroer_apply
)


# --- hook 2: qwen3_vl HAS_TRITON ------------------------------------------
# Triton kernel ``_bilinear_pos_embed_kernel`` is unsupported on Kunlun XPU.
# Force the module to fall back to native pos-embed interpolation.
def _qwen3vl_applied(mod):
    return not getattr(mod, "HAS_TRITON", False)


def _qwen3vl_apply(mod):
    mod.HAS_TRITON = False
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] qwen3_vl HAS_TRITON forced to False"
    )


_register_post_import_hook(
    "vllm.model_executor.models.qwen3_vl", _qwen3vl_applied, _qwen3vl_apply
)


# --- hook 3: BlockTable.compute_slot_mapping ------------------------------
# Replace the upstream Triton kernel with a torch-native version.
def _block_table_applied(mod):
    cls = getattr(mod, "BlockTable", None)
    return cls is None or getattr(cls, "_kunlun_slot_patched", False)


def _block_table_apply(mod):
    import vllm_kunlun.v1.worker.block_table  # noqa: F401  (self-applies on import)


_register_post_import_hook(
    "vllm.v1.worker.block_table", _block_table_applied, _block_table_apply
)


def _preload_mapped(full_name):
    """Load the kunlun replacement for ``full_name`` into sys.modules."""
    if full_name in sys.modules:
        return
    target_module = _MODULE_MAPPINGS[full_name]
    module = importlib.import_module(target_module)
    sys.modules[full_name] = module
    sys.modules[target_module] = module


def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    try:
        if level == 0:
            # Case 1: `from vllm.x.y import Z` / `import vllm.x.y`
            # Here module_name is the full dotted path of the mapped module.
            if module_name in _MODULE_MAPPINGS:
                _preload_mapped(module_name)

            # Case 2: `from vllm.x import y` where y itself is a mapped submodule.
            # CPython calls __import__("vllm.x", fromlist=("y",)); module_name
            # does not include "y", so we must check each fromlist entry.
            if fromlist:
                for name in fromlist:
                    full = f"{module_name}.{name}"
                    if full in _MODULE_MAPPINGS:
                        _preload_mapped(full)
    except Exception:
        pass

    result = OLD_IMPORT_HOOK(
        module_name, globals=globals, locals=locals, fromlist=fromlist, level=level
    )

    # Run all registered post-import hooks. Each hook checks its own
    # target module presence and idempotency flag; the dispatcher itself
    # has a re-entry guard so hook-triggered imports do not recurse.
    _dispatch_post_import_hooks()

    return result


def import_hook():
    """Apply import hook for VLLM Kunlun"""
    builtins.__import__ = _custom_import


def register():
    """Register the Kunlun platform"""

    logger = _configure_kunlun_logger()
    logger.info("[KunlunPlugin] register() pid=%s", os.getpid())

    # --- block vllm's NVIDIA prebuilt _C / _moe_C from being loaded ---
    # These are imported (via top-level ``import vllm._C`` in
    # ``vllm.platforms.cuda`` / inside ``Platform.import_kernels``) by
    # multiple vllm code paths. On Kunlun XPU they are useless and would
    # pre-register CUDA kernels that clash with the Kunlun
    # ``@custom_op`` / ``@impl(..., "CUDA")`` registrations on
    # PyTorch 2.9+. Stub them out NOW, before any other vllm import
    # has a chance to load them.
    import types as _types

    for _stub in ("vllm._C", "vllm._moe_C"):
        if _stub not in sys.modules:
            sys.modules[_stub] = _types.ModuleType(_stub)

    # --- eagerly register Kunlun custom ops ---
    # We load ``vllm_kunlun/ops/_custom_ops.py`` DIRECTLY via
    # ``spec_from_file_location`` under a private module name, instead of
    # ``import vllm_kunlun.ops`` which would trigger
    # ``vllm_kunlun/ops/__init__.py`` and transitively import
    # ``vllm_kunlun.ops.fused_moe.layer`` →
    # ``vllm.model_executor.layers.fused_moe.config`` →
    # ``vllm.model_executor.layers.quantization.utils.quant_utils`` →
    # ``vllm._custom_ops``. The last step calls
    # ``current_platform.import_kernels()`` while the platform plugin is
    # still mid-registration, which is fragile and was observed to leave
    # the worker process without any custom ops registered.
    #
    # Loading just the bare file registers all 54 Kunlun ops to
    # ``torch.ops._C`` / ``torch.ops._moe_C`` and avoids touching any
    # other vllm internals.
    try:
        import importlib.util as _ilu
        import os as _os

        _ops_file = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            "ops",
            "_custom_ops.py",
        )
        _private = "_vllm_kunlun_custom_ops_registration"
        if _private not in sys.modules:
            _spec = _ilu.spec_from_file_location(_private, _ops_file)
            _mod = _ilu.module_from_spec(_spec)
            sys.modules[_private] = _mod
            _spec.loader.exec_module(_mod)
        logger.info("[KunlunPlugin] vllm_kunlun custom ops registered")
    except Exception:
        logger.exception("[KunlunPlugin] custom ops registration failed")
        raise

    # --- load native extension to register torch.ops._C.weak_ref_tensor ---
    try:
        from . import _kunlun  # noqa: F401

        logger.info("[KunlunPlugin] _kunlun native extension loaded")
    except ImportError as e:
        logger.warning("[KunlunPlugin] Failed to load _kunlun: %s", e)

    # --- import wrapper & patch utils ---
    try:
        from .schema import direct_register_custom_op  # noqa: F401
        from .schema import patch_annotations_for_schema  # noqa: F401

        logger.info("[KunlunPlugin] vllm_utils_wrapper loaded and patched")
    except Exception:
        logger.exception("[KunlunPlugin] wrapper import/patch failed")
        raise

    # --- import hook ---
    try:
        import_hook()
        logger.info("[KunlunPlugin] import_hook() ok")
    except Exception:
        logger.exception("[KunlunPlugin] import_hook() failed")
        raise

    # --- register reasoning parser override (lazy, to avoid circular import) ---
    try:
        from vllm.reasoning import ReasoningParserManager

        # Override the lazy registration path with our custom parser.
        # This happens before vllm's default lazy registration (which is
        # triggered when vllm.reasoning module is imported), so our path
        # takes precedence.
        # Custom parser for Qwen3.5 support
        ReasoningParserManager.register_lazy_module(
            name="qwen3",
            module_path="vllm_kunlun.reasoning.qwen3_reasoning_parser",
            class_name="Qwen3ReasoningParser",
        )
        logger.info("[KunlunPlugin] registered Qwen3ReasoningParser override (lazy)")
    except Exception:
        logger.exception("[KunlunPlugin] Qwen3ReasoningParser registration failed")
        # Non-fatal: continue without the override

    logger.info("[KunlunPlugin] register() done")
    return "vllm_kunlun.platforms.kunlun.KunlunPlatform"


def register_model():
    """Register models for training and inference"""
    from .models import register_model as _reg

    _reg()
