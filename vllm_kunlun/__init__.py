"""vllm kunlun init"""

import builtins
import importlib
import logging
import os
import sys

OLD_IMPORT_HOOK = builtins.__import__


def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    try:
        module_mappings = {
            "vllm.compilation.wrapper": "vllm_kunlun.compilation.wrapper",
            "vllm.v1.worker.utils": "vllm_kunlun.v1.worker.utils",
            "vllm.model_executor.model_loader.bitsandbytes_loader": "vllm_kunlun.models.model_loader.bitsandbytes_loader",
            "vllm.v1.sample.ops.topk_topp_sampler": "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
            "vllm.model_executor.layers.sampler": "vllm_kunlun.ops.sample.sampler",
            "vllm.v1.sample.rejection_sampler": "vllm_kunlun.v1.sample.rejection_sampler",
            "vllm.attention.ops.merge_attn_states": "vllm_kunlun.ops.attention.merge_attn_states",
        }

        if module_name in module_mappings:
            if module_name in sys.modules:
                return sys.modules[module_name]
            target_module = module_mappings[module_name]
            module = importlib.import_module(target_module)
            sys.modules[module_name] = module
            sys.modules[target_module] = module
    except Exception:
        pass

    return OLD_IMPORT_HOOK(
        module_name, globals=globals, locals=locals, fromlist=fromlist, level=level
    )


def import_hook():
    """Apply import hook for VLLM Kunlun"""
    builtins.__import__ = _custom_import


def register():
    """Register the Kunlun platform"""

    logger = logging.getLogger("vllm_kunlun")
    logger.info("[KunlunPlugin] register() pid=%s", os.getpid())

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

    logger.info("[KunlunPlugin] register() done")
    return "vllm_kunlun.platforms.kunlun.KunlunPlatform"


def register_model():
    """Register models for training and inference"""
    from .models import register_model as _reg

    _reg()
