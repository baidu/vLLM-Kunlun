"""Substitute ``deep_gemm_fp8_o_proj`` with the Kunlun BF16 implementation.

DeepSeek-V4's NVIDIA submodules call this helper for the attention output
projection; Kunlun has no deep-gemm backend, so the symbol is rebound to
:func:`vllm_kunlun.ops.deepseek_v4_o_proj.deepseek_v4_bf16_o_proj`
(inverse-RoPE + grouped BMM, with a PyTorch reference fallback).
"""
import logging

LOGGER = logging.getLogger("vllm_kunlun.ops.attention.o_proj_alias")
_APPLIED_SENTINEL = "_dsv4_o_proj_wired"
_O_PROJ_FN_NAME = "deep_gemm_fp8_o_proj"
_EXPECTED_SOURCE = "vllm_kunlun.ops.deepseek_v4_o_proj"


def _predicate(mod: object) -> bool:
    fn = getattr(mod, _O_PROJ_FN_NAME, None)
    return bool(getattr(mod, _APPLIED_SENTINEL, False)) and fn is not None and getattr(
        fn, "__module__", ""
    ) == _EXPECTED_SOURCE


def _applier(mod: object) -> None:
    from ...ops.deepseek_v4_o_proj import deepseek_v4_bf16_o_proj

    original_fn = getattr(mod, _O_PROJ_FN_NAME, None)
    if original_fn is not None:
        # Keep the original reachable for debugging.
        setattr(deepseek_v4_bf16_o_proj, "_dsv4_original_impl", original_fn)

    setattr(mod, _O_PROJ_FN_NAME, deepseek_v4_bf16_o_proj)
    setattr(mod, _APPLIED_SENTINEL, True)
    LOGGER.info("Wired DSV4 O-projection alias into %s", mod.__name__)
