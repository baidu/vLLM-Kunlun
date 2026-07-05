"""vllm kunlun init"""

import builtins
import importlib
import logging
import os
import sys

from vllm.logger import init_logger as init_vllm_logger

OLD_IMPORT_HOOK = builtins.__import__

# vLLM module → Kunlun replacement module
_MODULE_MAPPINGS = {
    "vllm.compilation.wrapper": "vllm_kunlun.compilation.wrapper",
    "vllm.v1.worker.utils": "vllm_kunlun.v1.worker.utils",
    "vllm.model_executor.model_loader.bitsandbytes_loader": "vllm_kunlun.models.model_loader.bitsandbytes_loader",
    "vllm.v1.sample.ops.topk_topp_sampler": "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
    "vllm.v1.sample.rejection_sampler": "vllm_kunlun.v1.sample.rejection_sampler",
    "vllm.attention.ops.merge_attn_states": "vllm_kunlun.ops.attention.merge_attn_states",
    "vllm.v1.attention.backends.gdn_attn": "vllm_kunlun.v1.attention.backends.gdn_attn",
    "vllm.model_executor.models.config": "vllm_kunlun.models.config",
}


# =========================================================================
# Logger
# =========================================================================


def _apply_mamba_spec_patch(mamba_abstract_module) -> None:
    """Patch MambaBase.get_kv_cache_spec to allow qwen3_5_moe with MTP.

    Upstream vLLM only whitelists "qwen3_next" for mamba + speculative decoding.
    Imports are kept local to avoid triggering a circular import of vllm.config
    when this runs during plugin registration.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    _ALLOWED = {"qwen3_next", "qwen3_5_moe"}

    def _patched_get_kv_cache_spec(self, vllm_config):
        if (
            vllm_config.speculative_config is not None
            and vllm_config.model_config.hf_config.model_type not in _ALLOWED
        ):
            raise NotImplementedError(
                "Mamba with speculative decoding is not supported yet."
            )
        cache_config = vllm_config.cache_config
        return MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=cache_config.mamba_block_size,
            page_size_padded=cache_config.mamba_page_size_padded,
            mamba_type=self.mamba_type,
            mamba_cache_mode=cache_config.mamba_cache_mode,
            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config is not None
                else 0
            ),
        )

    mamba_abstract_module.MambaBase.get_kv_cache_spec = _patched_get_kv_cache_spec


def _apply_qwen35_mtp_speculative_patch(speculative_module) -> None:
    """Patch SpeculativeConfig to recognize Qwen3.5 MoE MTP drafts.

    This is intentionally narrow: only the verified Qwen3.5 MoE target is
    mapped to the local MTP draft implementation.
    """
    if getattr(speculative_module, "_kunlun_qwen35_mtp_patched", False):
        return

    from typing import Literal, get_args

    mtp_types = get_args(speculative_module.MTPModelTypes)
    if "qwen3_5_mtp" not in mtp_types:
        speculative_module.MTPModelTypes = Literal.__getitem__(
            mtp_types + ("qwen3_5_mtp",)
        )

    spec_config_cls = speculative_module.SpeculativeConfig
    original_hf_config_override = spec_config_cls.hf_config_override

    def _patched_hf_config_override(hf_config):
        hf_config = original_hf_config_override(hf_config)
        if hf_config.model_type == "qwen3_5_moe":
            text_config = getattr(hf_config, "text_config", None)
            n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
            if n_predict is None:
                n_predict = getattr(text_config, "mtp_num_hidden_layers", None)
            hf_config.model_type = "qwen3_5_mtp"
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP"],
                }
            )
        return hf_config

    spec_config_cls.hf_config_override = staticmethod(_patched_hf_config_override)
    speculative_module._kunlun_qwen35_mtp_patched = True


def _apply_mtp_mamba_state_patch(module_name: str) -> None:
    if module_name != "vllm.v1.worker.gpu_model_runner":
        return
    try:
        from vllm_kunlun.v1.worker import mtp_mamba_state_patch

        mtp_mamba_state_patch.patch_gpu_model_runner_module(sys.modules[module_name])
    except Exception:
        logging.getLogger("vllm_kunlun").exception(
            "[KunlunPlugin] MTP Mamba state patch failed for %s", module_name
        )


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


# =========================================================================
# Import hook
# =========================================================================


def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    try:
        if module_name in _MODULE_MAPPINGS and module_name not in sys.modules:
            target_module = _MODULE_MAPPINGS[module_name]
            mapped = importlib.import_module(target_module)
            sys.modules[module_name] = mapped
            sys.modules[target_module] = mapped
    except Exception:
        pass

    module = OLD_IMPORT_HOOK(
        module_name, globals=globals, locals=locals, fromlist=fromlist, level=level
    )

    # Lazy patch for MambaBase.get_kv_cache_spec. Importing this module during
    # register() triggers `from vllm.config import VllmConfig` while vllm.config
    # is still initializing.
    if module_name == "vllm.model_executor.layers.mamba.abstract" and not getattr(
        _custom_import, "_mamba_spec_patched", False
    ):
        _custom_import._mamba_spec_patched = True
        try:
            _apply_mamba_spec_patch(module)
            logging.getLogger("vllm_kunlun").info(
                "[KunlunPlugin] lazy-patched MambaBase.get_kv_cache_spec"
            )
        except Exception:
            _custom_import._mamba_spec_patched = False
            logging.getLogger("vllm_kunlun").exception(
                "[KunlunPlugin] lazy MambaBase.get_kv_cache_spec patch failed"
            )

    if module_name == "vllm.config.speculative" and not getattr(
        _custom_import, "_qwen35_mtp_spec_patched", False
    ):
        _custom_import._qwen35_mtp_spec_patched = True
        try:
            _apply_qwen35_mtp_speculative_patch(module)
            logging.getLogger("vllm_kunlun").info(
                "[KunlunPlugin] lazy-patched SpeculativeConfig for Qwen3.5 MTP"
            )
        except Exception:
            _custom_import._qwen35_mtp_spec_patched = False
            logging.getLogger("vllm_kunlun").exception(
                "[KunlunPlugin] lazy SpeculativeConfig MTP patch failed"
            )

    _apply_mtp_mamba_state_patch(module_name)

    return module


# =========================================================================
# Registration steps (each step is a self-contained function)
# =========================================================================

# Tracks which registration steps have completed successfully,
# so that repeated register() calls (triggered by vLLM's multi-phase
# plugin discovery) skip already-done work instead of re-executing.
_completed_steps: set[str] = set()


def _load_native_extension(logger: logging.Logger) -> None:
    """Load _kunlun C extension to register torch.ops._C.weak_ref_tensor."""
    if "native_ext" in _completed_steps:
        return
    _completed_steps.add("native_ext")  # only attempt once
    try:
        from . import _kunlun  # noqa: F401

        logger.info("[KunlunPlugin] _kunlun native extension loaded")
    except ImportError as e:
        logger.warning("[KunlunPlugin] Failed to load _kunlun: %s", e)


def _patch_schema_utils(logger: logging.Logger) -> None:
    """Import wrapper & patch schema utilities."""
    if "schema" in _completed_steps:
        return
    from .schema import direct_register_custom_op  # noqa: F401
    from .schema import patch_annotations_for_schema  # noqa: F401

    logger.info("[KunlunPlugin] schema utils loaded and patched")
    _completed_steps.add("schema")


def _patch_mamba_spec_if_loaded(logger: logging.Logger) -> None:
    """Patch MambaBase when the upstream module was already imported."""
    if "vllm.model_executor.layers.mamba.abstract" not in sys.modules:
        return
    if getattr(_custom_import, "_mamba_spec_patched", False):
        return

    _custom_import._mamba_spec_patched = True
    try:
        _apply_mamba_spec_patch(
            sys.modules["vllm.model_executor.layers.mamba.abstract"]
        )
        logger.info(
            "[KunlunPlugin] patched MambaBase.get_kv_cache_spec to allow qwen3_5_moe"
        )
    except Exception:
        _custom_import._mamba_spec_patched = False
        logger.exception("[KunlunPlugin] MambaBase.get_kv_cache_spec patch failed")


def _patch_qwen35_mtp_spec_if_loaded(logger: logging.Logger) -> None:
    """Patch SpeculativeConfig when vllm.config.speculative was already imported."""
    if "vllm.config.speculative" not in sys.modules:
        return
    if getattr(_custom_import, "_qwen35_mtp_spec_patched", False):
        return

    _custom_import._qwen35_mtp_spec_patched = True
    try:
        _apply_qwen35_mtp_speculative_patch(sys.modules["vllm.config.speculative"])
        logger.info("[KunlunPlugin] patched SpeculativeConfig for Qwen3.5 MTP")
    except Exception:
        _custom_import._qwen35_mtp_spec_patched = False
        logger.exception("[KunlunPlugin] SpeculativeConfig MTP patch failed")


def _patch_mtp_mamba_state_if_loaded(logger: logging.Logger) -> None:
    """Patch already loaded GPUModelRunner for MTP Mamba state handling."""
    try:
        from vllm_kunlun.v1.worker import mtp_mamba_state_patch

        mtp_mamba_state_patch.patch_loaded_modules()
    except Exception:
        logger.exception("[KunlunPlugin] eager MTP Mamba state patch failed")


def _install_import_hook(logger: logging.Logger) -> None:
    """Replace builtins.__import__ to redirect vLLM modules to Kunlun."""
    if "import_hook" in _completed_steps:
        return
    builtins.__import__ = _custom_import
    logger.info("[KunlunPlugin] import_hook() ok")
    _completed_steps.add("import_hook")


# =========================================================================
# Public API
# =========================================================================


def register():
    """Register the Kunlun platform.

    Called by vLLM plugin discovery before model loading.
    vLLM may invoke this multiple times during different discovery phases;
    each step tracks its own completion state via ``_completed_steps`` so
    already-succeeded work is skipped while previously-failed work (e.g.
    _patch_rotary_embedding blocked by circular import) is retried.
    """
    logger = _configure_kunlun_logger()

    first_call = "register_entered" not in _completed_steps
    if first_call:
        _completed_steps.add("register_entered")
        logger.info("[KunlunPlugin] register() pid=%s", os.getpid())

    _load_native_extension(logger)
    _patch_schema_utils(logger)  # fatal: raises on failure
    _patch_mamba_spec_if_loaded(logger)
    _patch_qwen35_mtp_spec_if_loaded(logger)
    _install_import_hook(logger)  # fatal: raises on failure
    _patch_mtp_mamba_state_if_loaded(logger)

    if first_call:
        logger.info("[KunlunPlugin] register() done")
    return "vllm_kunlun.platforms.kunlun.KunlunPlatform"


def register_model():
    """Register models for training and inference."""
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
