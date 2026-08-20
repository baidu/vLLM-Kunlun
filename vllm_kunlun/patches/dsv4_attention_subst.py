"""Symbol substitutions for ``vllm.models.deepseek_v4.attention``.

* ``fused_indexer_q_rope_quant`` -> Kunlun-native FP8 variant.
* ``aten::mm.dtype`` -> casted torch.mm fallback for mixed-dtype GEMM.
"""
import logging


LOGGER = logging.getLogger("vllm_kunlun.patches.dsv4_attention_subst")


_INDEXER_Q_APPLIED_ATTR = "_kunlun_indexer_q_patched"


def _indexer_q_predicate(mod) -> bool:
    fn = getattr(mod, "fused_indexer_q_rope_quant", None)
    expected_source = "vllm_kunlun.ops.fp8"
    fn_okay = fn is not None and getattr(fn, "__module__", "") == expected_source
    return bool(getattr(mod, _INDEXER_Q_APPLIED_ATTR, False)) and bool(fn_okay)


def _indexer_q_applier(mod) -> None:
    """Point ``fused_indexer_q_rope_quant`` at the Kunlun FP8 implementation."""
    from vllm_kunlun.ops.fp8 import fused_indexer_q_rope_quant_kunlun

    mod.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_kunlun
    setattr(mod, _INDEXER_Q_APPLIED_ATTR, True)
    LOGGER.info("Patched V4 indexer Q RoPE/FP8 quantization")


_MM_DTYPE_APPLIED_ATTR = "_kunlun_mm_dtype_library"


def _mm_dtype_predicate(mod) -> bool:
    return getattr(mod, _MM_DTYPE_APPLIED_ATTR, None) is not None


def _mm_dtype_applier(mod) -> None:
    """Register an ``aten::mm.dtype`` impl that casts inputs then calls torch.mm."""
    torch = mod.torch
    library = torch.library.Library("aten", "IMPL", "CUDA")

    def _kunlun_mm_dtype(input, mat2, out_dtype):
        return torch.mm(input.to(out_dtype), mat2.to(out_dtype))

    library.impl("mm.dtype", _kunlun_mm_dtype)
    setattr(mod, _MM_DTYPE_APPLIED_ATTR, library)
    LOGGER.info("Registered V4 aten::mm.dtype fallback")
