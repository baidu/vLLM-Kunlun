"""vllm kunlun init"""
from .platforms import current_platform
import sys
import importlib
import logging
import warnings
import builtins
import os
import time
import vllm.envs as envs

logger = logging.getLogger(__name__)

OLD_IMPORT_HOOK = builtins.__import__
_kv_admission_patched = False
_kv_scheduler_patched = False

def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    global _kv_admission_patched, _kv_scheduler_patched
    try:
        module_mappings = {
            "vllm.compilation.wrapper": "vllm_kunlun.compilation.wrapper",
            "vllm.v1.worker.utils": "vllm_kunlun.v1.worker.utils",
            "vllm.model_executor.model_loader.bitsandbytes_loader": "vllm_kunlun.models.model_loader.bitsandbytes_loader",
            "vllm.v1.sample.ops.topk_topp_sampler": "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
            "vllm.model_executor.layers.sampler": "vllm_kunlun.ops.sample.sampler",
            "vllm.v1.sample.ops.topk_topp_sampler": "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
            "vllm.v1.sample.rejection_sampler": "vllm_kunlun.v1.sample.rejection_sampler",
            "vllm.attention.ops.merge_attn_states": "vllm_kunlun.ops.attention.merge_attn_states",
            "vllm.v1.attention.backends.gdn_attn": "vllm_kunlun.v1.attention.backends.gdn_attn"
        }

        if module_name in module_mappings:
            if module_name in sys.modules:
                return sys.modules[module_name]
            target_module = module_mappings[module_name]
            module = importlib.import_module(target_module)
            sys.modules[module_name] = module
            sys.modules[target_module] = module
    except Exception as e:
        logger.warning("vllm_kunlun: failed to remap module %s: %s", module_name, e)

    result = OLD_IMPORT_HOOK(
        module_name,
        globals=globals,
        locals=locals,
        fromlist=fromlist,
        level=level
    )

    # Apply KV admission gate patch after kv_cache_manager is fully loaded.
    # Deferred to avoid importing vllm internals during early platform registration.
    if (not _kv_admission_patched
            and module_name == "vllm.v1.core.kv_cache_manager"):
        try:
            from vllm_kunlun.patches.kv_admission import apply as _apply_kv
            _apply_kv()
            _kv_admission_patched = True
        except Exception as e:
            logger.warning("vllm_kunlun: failed to apply KV admission patch: %s", e)

    # Apply partial-prefill concurrency limit patch after scheduler is loaded.
    if (not _kv_scheduler_patched
            and module_name == "vllm.v1.core.sched.scheduler"):
        try:
            from vllm_kunlun.patches.kv_admission import apply_scheduler as _apply_sched
            _apply_sched()
            _kv_scheduler_patched = True
        except Exception as e:
            logger.warning("vllm_kunlun: failed to apply scheduler patch: %s", e)

    return result

def import_hook():
    """Apply import hook for VLLM Kunlun"""
    builtins.__import__ = _custom_import

def register():
    """Register the Kunlun platform"""
    from .utils import redirect_output
    from .vllm_utils_wrapper import direct_register_custom_op, patch_annotations_for_schema
    
    # Change for GLM5 and custom model configs.
    import vllm.transformers_utils.config as config_module
    from .transformer_utils.config import _XPU_CONFIG_REGISTRY
    config_module._CONFIG_REGISTRY = _XPU_CONFIG_REGISTRY

    import vllm.transformers_utils.configs as configs_module
    from .transformer_utils.kimi_k25 import KimiK25Config, KimiK25VisionConfig
    setattr(configs_module, "KimiK25Config", KimiK25Config)
    setattr(configs_module, "KimiK25VisionConfig", KimiK25VisionConfig)
    
    import vllm.config.model as model_module
    from .config.model import is_deepseek_mla
    model_module.ModelConfig.is_deepseek_mla = property(is_deepseek_mla)
    
    import_hook()
    return "vllm_kunlun.platforms.kunlun.KunlunPlatform"

def register_model():
    """Register models for training and inference"""
    from .models import register_model as _reg
    _reg()

def register_tool_parser():
    from .entrypoints.openai.tool_parsers import (
        register_tool_parser as _reg_tool_parser,
    )

    _reg_tool_parser()


def register_reasoning_parser():
    from .reasoning import register_reasoning_parser as _reg_reasoning_parser

    _reg_reasoning_parser()
