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
    "vllm.attention.ops.merge_attn_states": "vllm_kunlun.ops.attention.merge_attn_states",
    "vllm.v1.worker.mamba_utils": "vllm_kunlun.v1.worker.mamba_utils",
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


# --- hook 3: MiniMax TP QK-RMSNorm -----------------------------------------
# The upstream Triton path is selected when the CUDA-compatible runtime reports
# an available driver. That kernel cannot launch on Kunlun XPU, so force the
# existing eager fallback for tensor-parallel MiniMax models.
def _minimax_rms_norm_tp_applied(mod):
    return not getattr(mod, "HAS_TRITON", False)


def _minimax_rms_norm_tp_apply(mod):
    mod.HAS_TRITON = False
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] minimax rms_norm_tp HAS_TRITON forced to False"
    )


_register_post_import_hook(
    "vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp",
    _minimax_rms_norm_tp_applied,
    _minimax_rms_norm_tp_apply,
)


# --- hook 4: BlockTable.compute_slot_mapping ------------------------------
# Replace the upstream Triton kernel with a torch-native version.
def _block_table_applied(mod):
    cls = getattr(mod, "BlockTable", None)
    return cls is None or getattr(cls, "_kunlun_slot_patched", False)


def _block_table_apply(mod):
    import vllm_kunlun.v1.worker.block_table  # noqa: F401  (self-applies on import)


_register_post_import_hook(
    "vllm.v1.worker.block_table", _block_table_applied, _block_table_apply
)


# --- hook 5: apply_grammar_bitmask in vllm.v1.structured_output.utils -----
# Replace the upstream xgrammar auto backend with torch_native on Kunlun XPU.
def _grammar_bitmask_applied(mod):
    fn = getattr(mod, "apply_grammar_bitmask", None)
    return fn is not None and getattr(fn, "_kunlun_patched", False)


def _grammar_bitmask_apply(mod):
    if not hasattr(mod, "apply_grammar_bitmask"):
        return
    import vllm_kunlun.v1.structured_output.utils  # noqa: F401


_register_post_import_hook(
    "vllm.v1.structured_output.utils", _grammar_bitmask_applied, _grammar_bitmask_apply
)


# --- hook 6: Worker._maybe_get_memory_pool_context -----------------------
# vllm 0.25.1 _maybe_get_memory_pool_context() gates on is_cuda_alike() /
# is_xpu(). KunlunPlatform is OOT so neither returns True, causing it to
# fall through to get_mem_allocator_instance() which raises RuntimeError.
# Patch the method to return nullcontext() for Kunlun.
def _memory_pool_applied(mod):
    cls = getattr(mod, "Worker", None)
    return cls is None or getattr(cls, "_kunlun_memory_pool_patched", False)


def _memory_pool_apply(mod):
    from contextlib import nullcontext as _nullcontext

    _orig = mod.Worker._maybe_get_memory_pool_context

    def _patched(self, tag: str):
        from vllm.platforms import current_platform

        if type(current_platform).__name__ == "KunlunPlatform":
            return _nullcontext()
        return _orig(self, tag)

    mod.Worker._maybe_get_memory_pool_context = _patched
    mod.Worker._kunlun_memory_pool_patched = True
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched Worker._maybe_get_memory_pool_context"
    )


_register_post_import_hook(
    "vllm.v1.worker.gpu_worker", _memory_pool_applied, _memory_pool_apply
)


# --- hook 7: skip qwen_triton_warmup on Kunlun XPU ---
def _qwen_triton_warmup_applied(mod):
    fn = getattr(mod, "qwen_triton_warmup", None)
    return fn is not None and getattr(fn, "_kunlun_patched", False)


def _qwen_triton_warmup_apply(mod):
    def _noop(*args, **kwargs):
        import logging

        logging.getLogger("vllm_kunlun").info(
            "[KunlunPlugin] Skipping qwen_triton_warmup"
        )

    _noop._kunlun_patched = True
    mod.qwen_triton_warmup = _noop
    import logging

    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched kernel_warmup.qwen_triton_warmup -> no-op"
    )


_register_post_import_hook(
    "vllm.model_executor.warmup.kernel_warmup",
    _qwen_triton_warmup_applied,
    _qwen_triton_warmup_apply,
)


# --- hook 8: out-of-tree custom-op registrations --------------------------
# vLLM 0.25.1 discovers OOT pluggable layers only after custom_op is imported.
# Import the Kunlun operators at that point so their decorators are registered.
def _oot_registrations_applied(mod):
    if not hasattr(mod, "CustomOp") or not hasattr(mod, "PluggableLayer"):
        return True
    ops_module = sys.modules.get("vllm_kunlun.ops")
    return bool(getattr(ops_module, "_KUNLUN_OOT_REGISTRATIONS_LOADED", False))


def _oot_registrations_apply(mod):
    import vllm_kunlun.ops  # noqa: F401


_register_post_import_hook(
    "vllm.model_executor.custom_op",
    _oot_registrations_applied,
    _oot_registrations_apply,
)


# --- hook 9: compressed-tensors INT8 MoE backend selector -----------------
# compressed-tensors 0.17.0 asks vLLM to choose a CUDA-specific INT8 MoE
# backend. Return the generic path so Kunlun's implementation handles it.
def _int8_moe_applied(mod):
    if not hasattr(mod, "select_int8_moe_backend"):
        return False
    return getattr(mod, "_kunlun_select_int8_patched", False)


def _select_int8_moe_backend(config, weight_key=None, activation_key=None):
    return None, None


def _int8_moe_apply(mod):
    if not hasattr(mod, "select_int8_moe_backend"):
        return
    mod.select_int8_moe_backend = _select_int8_moe_backend
    mod._kunlun_select_int8_patched = True


_register_post_import_hook(
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
    _int8_moe_applied,
    _int8_moe_apply,
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

    # --- patch torch.accelerator.get_memory_info for Kunlun XPU ---
    # vllm 0.25.1 uses torch.accelerator.get_memory_info() which does not exist
    # in torch_xmlir 2.9. Patch it to use torch.cuda.mem_get_info which works on XPU.
    try:
        import torch as _torch

        def _kunlun_get_memory_info(device=None):
            if device is None:
                idx = _torch.cuda.current_device()
            elif isinstance(device, _torch.device):
                idx = (
                    device.index
                    if device.index is not None
                    else _torch.cuda.current_device()
                )
            elif isinstance(device, int):
                idx = device
            else:
                idx = _torch.cuda.current_device()
            return _torch.cuda.mem_get_info(idx)

        _torch.accelerator.get_memory_info = _kunlun_get_memory_info
        logger.info("[KunlunPlugin] patched torch.accelerator.get_memory_info")
    except Exception:
        logger.exception(
            "[KunlunPlugin] failed to patch torch.accelerator.get_memory_info"
        )
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


def register_reasoning_parser():
    """Register reasoning parsers for inference."""
    from .reasoning import register_reasoning_parser as _reg_reasoning_parser

    _reg_reasoning_parser()


def register_tool_parser():
    """Register tool parsers for inference."""
    from .entrypoints.openai.tool_parsers import (
        register_tool_parser as _reg_tool_parser,
    )

    _reg_tool_parser()
