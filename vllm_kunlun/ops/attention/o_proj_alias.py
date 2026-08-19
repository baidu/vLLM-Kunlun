"""DeepSeek-V4 output projection alias wiring.

The V4 attention/MLA blocks call a helper named ``deep_gemm_fp8_o_proj``
inside several NVIDIA-specific submodules
(``ops.o_proj``, ``flashmla``, ``flashinfer_sparse``).  On Kunlun there is no
deep-gemm backend; instead we substitute
:func:`vllm_kunlun.ops.deepseek_v4_o_proj.deepseek_v4_bf16_o_proj`, which
performs inverse-RoPE + grouped BMM (+ optional FP8 weight dequant) using the
available XPU kernels, falling back to PyTorch reference automatically.
"""
import logging
from typing import List

from vllm_kunlun.runtime_utils import WarningOnce

LOGGER = logging.getLogger("vllm_kunlun.ops.attention.o_proj_alias")
_APPLIED_SENTINEL = "_dsv4_o_proj_wired"
_O_PROJ_FN_NAME = "deep_gemm_fp8_o_proj"
_EXPECTED_SOURCE = "vllm_kunlun.ops.deepseek_v4_o_proj"
_TARGET_MODULES = (
    "vllm.models.deepseek_v4.nvidia.ops.o_proj",
    "vllm.models.deepseek_v4.nvidia.flashmla",
    "vllm.models.deepseek_v4.nvidia.flashinfer_sparse",
)


def _predicate(mod: object) -> bool:
    fn = getattr(mod, _O_PROJ_FN_NAME, None)
    return bool(getattr(mod, _APPLIED_SENTINEL, False)) and fn is not None and getattr(
        fn, "__module__", ""
    ) == _EXPECTED_SOURCE


def _applier(mod: object) -> None:
    # Lazy import so the heavy vLLM model graph is only pulled in when needed.
    from ...ops.deepseek_v4_o_proj import deepseek_v4_bf16_o_proj

    original_fn = getattr(mod, _O_PROJ_FN_NAME, None)
    replacement = deepseek_v4_bf16_o_proj

    # Keep the old symbol discoverable under a private attribute so callers can
    # reach upstream behaviour without re-importing community modules.
    try:
        if original_fn is not None:
            setattr(replacement, "_dsv4_original_impl", original_fn)
    except Exception:
        pass

    setattr(mod, _O_PROJ_FN_NAME, replacement)
    setattr(mod, _APPLIED_SENTINEL, True)
    LOGGER.info("Wired DSV4 O-projection alias into %s", mod.__name__)
