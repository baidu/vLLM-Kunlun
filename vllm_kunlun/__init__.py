"""vLLM Kunlun plugin registration entrypoints.

Keep platform discovery and the public registration wrappers here.  The
implementation lives in the ``registration`` package: startup stages in
``registration/bootstrap.py``, wholesale module replacements in
``registration/module_redirects.py``, and post-import patches in
``registration/compat_patches.py``.
"""

import logging
import os

from .registration import bootstrap
from .registration.import_hooks import dispatch_hooks, install_import_hook

# Platform discovery can call register() more than once in the same process,
# including re-entrantly while the first call is importing vLLM modules.
_REGISTER_STATE = "idle"
_REGISTER_ERROR: BaseException | None = None
_KUNLUN_PLATFORM = "vllm_kunlun.platforms.kunlun.KunlunPlatform"


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


def _run_startup_stages(logger: logging.Logger) -> None:
    """Run every startup stage in dependency order.

    Failure policy lives in the stages themselves: optional stages catch and
    log their own errors, while load-bearing stages let them propagate to
    ``register()``.
    """
    # 1. Block CUDA extension imports before any vLLM kernel module loads.
    bootstrap.stub_vllm_cuda_extensions()
    # 2. Register bare custom ops before OOT layer imports need them.
    bootstrap.register_custom_ops(logger)
    # 3. Optional speculative-decoding compatibility patches.
    bootstrap.load_spec_decode_compat(logger)
    # 4. Native operators backing the Python ops.
    bootstrap.load_native_extension(logger)
    # 5. Patch custom-op schema registration before any model loads.
    bootstrap.load_schema_helpers(logger)
    # 6. Install the import dispatcher, then patch the vLLM modules that
    #    were already imported before it was installed.
    install_import_hook()
    dispatch_hooks()
    logger.info("[KunlunPlugin] import hook installed")
    # 7. Add torch_xmlir's missing memory-info API.
    bootstrap.patch_memory_info(logger)


def register() -> str:
    """Register the Kunlun platform and its compatibility patches."""
    global _REGISTER_ERROR, _REGISTER_STATE
    if _REGISTER_STATE == "failed":
        raise RuntimeError(
            "Kunlun plugin registration previously failed; " "retry in a fresh process"
        ) from _REGISTER_ERROR
    if _REGISTER_STATE != "idle":
        logging.getLogger("vllm_kunlun").debug(
            "[KunlunPlugin] register() skipped; state=%s", _REGISTER_STATE
        )
        return _KUNLUN_PLATFORM
    _REGISTER_STATE = "registering"

    logger = _configure_kunlun_logger()
    logger.info("[KunlunPlugin] register() pid=%s", os.getpid())
    try:
        _run_startup_stages(logger)
    except Exception as error:
        if isinstance(error, bootstrap.CustomOpsRegistrationError):
            _REGISTER_ERROR = error
            _REGISTER_STATE = "failed"
        else:
            _REGISTER_STATE = "idle"
        logger.exception("[KunlunPlugin] register() failed")
        raise
    _REGISTER_STATE = "registered"
    logger.info("[KunlunPlugin] register() done")
    return _KUNLUN_PLATFORM


def register_model():
    """Register models for training and inference."""
    from .models import register_model as _register

    _register()


def register_reasoning_parser():
    """Register reasoning parsers for inference."""
    from .reasoning import register_reasoning_parser as _register

    _register()


def register_tool_parser():
    """Register tool parsers for inference."""
    from .tool_parsers import register_tool_parser as _register

    _register()
