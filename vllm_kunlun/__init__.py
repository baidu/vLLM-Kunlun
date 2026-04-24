"""vllm kunlun init"""

import builtins
import importlib
import logging
import os
import sys

from .compat import (
    _patch_piecewise_compile_interpreter_call_module,
    _patch_traceable_vllm_parameter_subclasses,
    _patch_v1_block_table_triton_kernel,
    apply_qwen3_moe_loader_compat_patch,
    apply_torch251_compat_shims,
)
from .ops.fused_moe import register_kunlun_fused_moe_ops

apply_torch251_compat_shims()

OLD_IMPORT_HOOK = builtins.__import__


def _has_scaled_int8_quant_op() -> bool:
    import torch

    return hasattr(torch.ops._C, "scaled_int8_quant")


def _has_cache_concat_mla_op() -> bool:
    import torch

    return hasattr(torch.ops._C_cache_ops, "concat_and_cache_mla")


def _has_required_python_custom_ops() -> bool:
    return _has_scaled_int8_quant_op() and _has_cache_concat_mla_op()


def _ensure_python_custom_ops_registered(logger: logging.Logger) -> None:
    if _has_required_python_custom_ops():
        return

    importlib.import_module("vllm_kunlun.ops._custom_ops")
    logger.info("[KunlunPlugin] Python custom op definitions loaded")


def _configure_kunlun_logger() -> logging.Logger:
    """Reuse vLLM's handler for the vllm_kunlun logger tree."""
    from vllm.logger import init_logger as init_vllm_logger

    vllm_logger = init_vllm_logger("vllm")
    kunlun_logger = logging.getLogger("vllm_kunlun")

    if not kunlun_logger.handlers:
        for handler in vllm_logger.handlers:
            kunlun_logger.addHandler(handler)

    kunlun_logger.setLevel(vllm_logger.getEffectiveLevel())
    kunlun_logger.propagate = False
    return kunlun_logger


def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    module_mappings = {
        "vllm.compilation.wrapper": "vllm_kunlun.compilation.wrapper",
        "vllm.v1.worker.utils": "vllm_kunlun.v1.worker.utils",
        "vllm.attention.backends.abstract": "vllm.v1.attention.backend",
        "vllm.model_executor.model_loader.bitsandbytes_loader": "vllm_kunlun.models.model_loader.bitsandbytes_loader",
        "vllm.v1.sample.ops.topk_topp_sampler": "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
        "vllm.v1.sample.rejection_sampler": "vllm_kunlun.v1.sample.rejection_sampler",
        "vllm.attention.ops.common": "vllm.v1.attention.ops.common",
        "vllm.attention.ops.flashmla": "vllm.v1.attention.ops.flashmla",
        "vllm.attention.ops.merge_attn_states": "vllm_kunlun.ops.attention.merge_attn_states",
        "vllm.v1.attention.ops.merge_attn_states": "vllm_kunlun.ops.attention.merge_attn_states",
        "vllm.model_executor.models.config": "vllm_kunlun.models.config",
    }

    if module_name in module_mappings:
        if module_name in sys.modules:
            return sys.modules[module_name]
        target_module = module_mappings[module_name]
        module = importlib.import_module(target_module)
        sys.modules[module_name] = module
        sys.modules[target_module] = module
        return module

    module = OLD_IMPORT_HOOK(
        module_name, globals=globals, locals=locals, fromlist=fromlist, level=level
    )

    if module_name == "vllm.model_executor.models.qwen3_moe":
        try:
            apply_qwen3_moe_loader_compat_patch(module)
        except Exception:
            logging.getLogger("vllm_kunlun").exception(
                "[KunlunPlugin] deferred Qwen3-MoE loader compat patch failed"
            )
            raise
    elif module_name == "vllm.model_executor.parameter":
        try:
            _patch_traceable_vllm_parameter_subclasses(module)
        except Exception:
            logging.getLogger("vllm_kunlun").exception(
                "[KunlunPlugin] deferred vLLM parameter Dynamo compat patch failed"
            )
            raise
    elif module_name == "vllm.compilation.backends":
        try:
            _patch_piecewise_compile_interpreter_call_module(module)
        except Exception:
            logging.getLogger("vllm_kunlun").exception(
                "[KunlunPlugin] deferred PiecewiseCompileInterpreter compat patch failed"
            )
            raise
    elif module_name == "vllm.v1.worker.block_table":
        try:
            _patch_v1_block_table_triton_kernel(module)
        except Exception:
            logging.getLogger("vllm_kunlun").exception(
                "[KunlunPlugin] deferred v1 block_table Triton compat patch failed"
            )
            raise
    elif module_name == "vllm.model_executor.layers.fused_moe.layer":
        try:
            register_kunlun_fused_moe_ops()
        except Exception:
            logging.getLogger("vllm_kunlun").exception(
                "[KunlunPlugin] deferred Kunlun FusedMoE override registration failed"
            )
            raise

    return module


def import_hook():
    """Apply import hook for VLLM Kunlun"""
    builtins.__import__ = _custom_import


def register():
    """Register the Kunlun platform"""

    logger = _configure_kunlun_logger()
    logger.info("[KunlunPlugin] register() pid=%s", os.getpid())

    # --- load native extension to register torch.ops._C.weak_ref_tensor ---
    try:
        from . import _kunlun  # noqa: F401

        logger.info("[KunlunPlugin] _kunlun native extension loaded")
    except ImportError as e:
        logger.warning("[KunlunPlugin] Failed to load _kunlun: %s", e)

    try:
        _ensure_python_custom_ops_registered(logger)
    except Exception:
        logger.exception("[KunlunPlugin] Python custom op registration failed")
        raise

    # --- import wrapper & patch utils ---
    try:
        from .schema import direct_register_custom_op  # noqa: F401
        from .schema import patch_annotations_for_schema  # noqa: F401

        logger.info("[KunlunPlugin] vllm_utils_wrapper loaded and patched")
    except Exception:
        logger.exception("[KunlunPlugin] wrapper import/patch failed")
        raise

    # TODO @xyDong0223 Fix Hear, import failed in v15.1
    # --- optional GLM5 config patch ---
    # if "vllm.transformers_utils.config" in sys.modules:
    #     from .transformer_utils.config import _XPU_CONFIG_REGISTRY
    #     sys.modules["vllm.transformers_utils.config"]._CONFIG_REGISTRY = _XPU_CONFIG_REGISTRY
    #     logger.info("[KunlunPlugin] patched transformers_utils.config")

    # --- patch ModelConfig ---
    # try:
    #     import vllm.config.model as model_module
    #     from .config.model import is_deepseek_mla
    #     model_module.ModelConfig.is_deepseek_mla = property(is_deepseek_mla)
    #     logger.info("[KunlunPlugin] patched ModelConfig.is_deepseek_mla")
    # except Exception:
    #     logger.exception("[KunlunPlugin] ModelConfig patch failed")
    #     raise

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
