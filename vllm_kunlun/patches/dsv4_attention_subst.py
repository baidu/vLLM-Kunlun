"""Miscellaneous Kunlun symbol substitutions targeting
``vllm.models.deepseek_v4.attention``.

Two unrelated but equally-trivial responsibilities live here because neither deserves
its own package-level directory:

* **indexer Q** swaps ``fused_indexer_q_rope_quant`` to point at the Kunlun-native FP8
  variant exported from :mod:`vllm_kunlun.ops.fp8`.
* **mm dtype** registers an ``aten::mm.dtype`` CUDA dispatch-table impl returning a casted
  torch.mm result, satisfying upstream dtype-fallback expectations on platforms lacking
  certain mixed-dtype GEMM kernels.
"""
import logging
from typing import List


LOGGER = logging.getLogger("vllm_kunlun.patches.dsv4_attention_subst")


_INDEXER_Q_APPLIED_ATTR = "_kunlun_indexer_q_patched"


def _indexer_q_predicate(mod) -> bool:
    fn = getattr(mod, "fused_indexer_q_rope_quant", None)
    expected_source = "vllm_kunlun.ops.fp8"
    fn_okay = fn is not None and getattr(fn, "__module__", "") == expected_source
    return bool(getattr(mod, _INDEXER_Q_APPLIED_ATTR, False)) and bool(fn_okay)


def _indexer_q_applier(mod) -> None:
    """Replace ``fused_indexer_q_rope_quant`` with Kunlun-native FP8 quantized variant."""
    from vllm_kunlun.ops.fp8 import fused_indexer_q_rope_quant_kunlun

    mod.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_kunlun
    setattr(mod, _INDEXER_Q_APPLIED_ATTR, True)
    LOGGER.info("[KunlunPlugin] patched V4 Indexer Q RoPE/FP8 quantization")


_MM_DTYPE_APPLIED_ATTR = "_kunlun_mm_dtype_library"


def _mm_dtype_predicate(mod) -> bool:
    return getattr(mod, _MM_DTYPE_APPLIED_ATTR, None) is not None


def _mm_dtype_applier(mod) -> None:
    """Register ``aten::mm.dtype`` CUDA IMPL returning casted-torch.mm output.

    Lifted verbatim from legacy root hook ``_v4_attention_mm_dtype_apply``.
    """
    torch = mod.torch
    library = torch.library.Library("aten", "IMPL", "CUDA")

    def _kunlun_mm_dtype(input, mat2, out_dtype):
        return torch.mm(input.to(out_dtype), mat2.to(out_dtype))

    library.impl("mm.dtype", _kunlun_mm_dtype)
    setattr(mod, _MM_DTYPE_APPLIED_ATTR, library)
    LOGGER.info("[KunlunPlugin] registered V4 aten::mm.dtype fallback")


def apply(master_enabled_check: bool = True) -> List[str]:
    """Legacy eager-install shim retained purely for backward compatibility."""
    del master_enabled_check
    return []
