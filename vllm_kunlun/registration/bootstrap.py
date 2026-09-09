"""Startup helpers for the Kunlun vLLM plugin.

``vllm_kunlun.register()`` calls these helpers in a fixed order during
platform discovery, before the import dispatcher in ``import_hooks`` takes
over.  Each public function is one self-contained startup stage:

1. ``stub_vllm_cuda_extensions()``: keep vLLM's CUDA extension imports from
   failing on a machine without CUDA.
2. ``register_custom_ops()``: register Kunlun operators with torch early.
3. ``load_spec_decode_compat()``: optional speculative-decoding patches.
4. ``register_weak_ref_tensor()``: alias the ``_C`` operator vLLM hardcodes.
5. ``load_schema_helpers()``: patch vLLM's custom-op schema registration.
6. ``patch_memory_info()``: fill in a torch API missing from torch_xmlir.

Failure policy differs by stage on purpose: operator registration and the
memory-info patch are load-bearing and re-raise, while the optional stages
only log, so a partial environment can still start.
"""

import importlib
import importlib.util
import logging
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

_CUDA_EXTENSION_MODULES = ("vllm._C", "vllm._moe_C")
_CUSTOM_OPS_PRIVATE_NAME = "_vllm_kunlun_custom_ops_registration"
_CUSTOM_OPS_CANONICAL_NAME = "vllm_kunlun.ops._custom_ops"
_CUSTOM_OPS_REGISTRATION_ERROR: BaseException | None = None
_SPEC_DECODE_COMPAT_MODULES = (
    "vllm_kunlun.v1.sample.spec_decode.dflash",
    "vllm_kunlun.v1.sample.spec_decode.eagle",
)
_MIN_XSPEEDGATE_VERSION = (1, 5, 0)
_WEAK_REF_TENSOR_LIBRARY = None


class CustomOpsRegistrationError(RuntimeError):
    """A custom-op registration failure that cannot be retried in-process."""


def stub_vllm_cuda_extensions() -> None:
    """Prevent vLLM CUDA extensions from being imported on Kunlun.

    vLLM imports its compiled CUDA extensions unconditionally.  Registering
    empty placeholder modules up front makes those imports succeed without
    loading any CUDA code; the kernels they would provide are supplied by
    Kunlun's own operators instead.
    """
    for name in _CUDA_EXTENSION_MODULES:
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)


def _load_custom_ops_module() -> ModuleType:
    """Load the bare custom-op module exactly once.

    The file is loaded by path under a private module name rather than via
    ``import vllm_kunlun.ops``, because importing the full ops package this
    early would pull in dependencies that are not ready yet during platform
    discovery.

    Running the file twice would repeat its torch custom-op registrations
    (an error), so if the canonical package import happened first, both
    names are pointed at that same module instead of executing it again.
    """
    global _CUSTOM_OPS_REGISTRATION_ERROR

    if _CUSTOM_OPS_REGISTRATION_ERROR is not None:
        raise CustomOpsRegistrationError(
            "Kunlun custom-op registration previously failed; "
            "retry in a fresh process"
        ) from _CUSTOM_OPS_REGISTRATION_ERROR

    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ops_file = os.path.join(package_root, "ops", "_custom_ops.py")

    if _CUSTOM_OPS_PRIVATE_NAME in sys.modules:
        return sys.modules[_CUSTOM_OPS_PRIVATE_NAME]

    canonical_module = sys.modules.get(_CUSTOM_OPS_CANONICAL_NAME)
    if canonical_module is not None:
        sys.modules[_CUSTOM_OPS_PRIVATE_NAME] = canonical_module
        return canonical_module

    spec = importlib.util.spec_from_file_location(
        _CUSTOM_OPS_PRIVATE_NAME,
        ops_file,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Kunlun ops from {ops_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CUSTOM_OPS_PRIVATE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        # Torch dispatcher registrations are not rolled back with the module.
        # Mark this stage as non-retryable instead of re-executing the file.
        _CUSTOM_OPS_REGISTRATION_ERROR = CustomOpsRegistrationError(
            "Kunlun custom-op registration partially completed and cannot "
            "be retried in this process"
        )
        sys.modules.pop(_CUSTOM_OPS_PRIVATE_NAME, None)
        raise _CUSTOM_OPS_REGISTRATION_ERROR from error
    return module


def register_custom_ops(logger: logging.Logger) -> None:
    """Register Kunlun custom operators during platform discovery."""
    _load_custom_ops_module()
    logger.info("[KunlunPlugin] vllm_kunlun custom ops registered")


def load_spec_decode_compat(logger: logging.Logger) -> None:
    """Load optional speculative-decoding compatibility modules.

    These modules patch vLLM's speculative-decoding paths on import.  They
    are optional because the vLLM features they target do not exist in every
    supported version; a missing module is logged at debug level and skipped.
    """
    for module_name in _SPEC_DECODE_COMPAT_MODULES:
        try:
            importlib.import_module(module_name)
            logger.info(
                "[KunlunPlugin] loaded speculative-decoding compatibility: %s",
                module_name,
            )
        except ImportError as error:
            logger.debug(
                "[KunlunPlugin] speculative-decoding compatibility unavailable: "
                "%s: %s",
                module_name,
                error,
            )


def register_weak_ref_tensor(logger: logging.Logger) -> None:
    """Provide ``torch.ops._C.weak_ref_tensor`` by aliasing xspeedgate_ops.

    vLLM's CUDA-graph capture calls ``torch.ops._C.weak_ref_tensor``, which
    normally comes from vLLM's own CUDA extension.  ``_C`` is empty on Kunlun
    because ``stub_vllm_cuda_extensions()`` blocks that extension, so the
    equivalent xspeedgate_ops operator is registered under the name vLLM
    expects.  Both build an alias tensor with ``from_blob`` and no deleter,
    and torch_xmlir maps Kunlun tensors to the CUDA dispatch key.
    """
    global _WEAK_REF_TENSOR_LIBRARY

    if _WEAK_REF_TENSOR_LIBRARY is not None:
        return

    import torch

    try:
        import xspeedgate_ops  # noqa: F401  registers the custom operators

        installed_version = version("xspeedgate_ops")
    except (ImportError, PackageNotFoundError) as error:
        raise RuntimeError(
            "Kunlun requires xspeedgate_ops>=1.5.0; install a compatible "
            "vendor wheel before loading the vLLM-Kunlun plugin"
        ) from error

    version_match = re.match(r"^(\d+)\.(\d+)\.(\d+)", installed_version)
    if version_match is None or tuple(map(int, version_match.groups())) < (
        _MIN_XSPEEDGATE_VERSION
    ):
        raise RuntimeError(
            "Kunlun requires xspeedgate_ops>=1.5.0; found " f"{installed_version!r}"
        )

    try:
        weak_ref_op = torch.ops.xspeedgate_ops.weak_ref_tensor.default
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(
            "Kunlun requires xspeedgate_ops>=1.5.0 with the " "weak_ref_tensor operator"
        ) from error

    # The Library must stay referenced: its destructor deregisters the op.
    library = torch.library.Library("_C", "FRAGMENT")
    library.define("weak_ref_tensor(Tensor input) -> Tensor")
    library.impl(
        "weak_ref_tensor",
        weak_ref_op,
        "CUDA",
    )
    _WEAK_REF_TENSOR_LIBRARY = library
    logger.info("[KunlunPlugin] registered _C::weak_ref_tensor via xspeedgate_ops")


def load_schema_helpers(logger: logging.Logger) -> None:
    """Patch vLLM's custom-op schema registration.

    Importing ``vllm_kunlun.schema`` applies the patch as a side effect and
    exposes ``direct_register_custom_op`` to the rest of the plugin.
    """
    from .. import schema  # noqa: F401

    logger.info("[KunlunPlugin] vLLM custom-op schema helpers loaded")


def _resolve_device_index(torch_module, device) -> int:
    """Resolve a supported device value to the CUDA-compatible index.

    Anything without a usable index (``None``, a ``torch.device`` created
    without one, or an unrecognized type) falls back to the current device,
    matching how torch itself treats such values.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, torch_module.device) and device.index is not None:
        return device.index
    return torch_module.cuda.current_device()


def _kunlun_get_memory_info(device=None) -> tuple[int, int]:
    """Return free and total memory in bytes for a Kunlun device."""
    import torch

    device_index = _resolve_device_index(torch, device)
    return torch.cuda.mem_get_info(device_index)


def patch_memory_info(logger: logging.Logger) -> None:
    """Install Kunlun's memory-info compatibility function on torch.

    torch_xmlir does not implement ``torch.accelerator.get_memory_info``,
    which vLLM's memory profiling calls.  Route it to
    ``torch.cuda.mem_get_info``, which torch_xmlir maps to the Kunlun
    runtime.
    """
    import torch

    torch.accelerator.get_memory_info = _kunlun_get_memory_info
    logger.info("[KunlunPlugin] patched torch.accelerator.get_memory_info")
