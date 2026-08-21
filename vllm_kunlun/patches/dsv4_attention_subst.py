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


_MM_DTYPE_W_CACHE: "dict[tuple, object]" = {}


def _mm_dtype_applier(mod) -> None:
    """Register an ``aten::mm.dtype`` impl that casts inputs then calls torch.mm."""
    torch = mod.torch
    library = torch.library.Library("aten", "IMPL", "CUDA")

    def _kunlun_mm_dtype(input, mat2, out_dtype):
        """mm with an explicit output dtype: cast both sides, then torch.mm.

        The V4 callers pass ``weight.T`` (mat2._base is a Parameter), so the
        weight cast is cacheable and would remove a per-layer transpose+cast
        from every step. The fp32 copy is opt-in (KUNLUN_DSV4_MM_DTYPE_WCACHE=1,
        together with a higher --gpu-memory-utilization) because vLLM sizes the
        KV pool from post-profile free memory: a persistent cache directly
        shrinks the pool. Non-Parameter mat2 always casts per call.
        """
        import os
        base = getattr(mat2, "_base", None)
        if (
            os.environ.get("KUNLUN_DSV4_MM_DTYPE_WCACHE") == "1"
            and isinstance(base, torch.nn.Parameter)
        ):
            key = (mat2.data_ptr(), tuple(mat2.shape), out_dtype)
            w = _MM_DTYPE_W_CACHE.get(key)
            if w is None:
                w = mat2.to(out_dtype)
                _MM_DTYPE_W_CACHE[key] = w
                if len(_MM_DTYPE_W_CACHE) % 10 == 0:
                    LOGGER.info(
                        "[mm.dtype] wcache entries=%d bytes=%d",
                        len(_MM_DTYPE_W_CACHE),
                        sum(t.numel() * t.element_size() for t in _MM_DTYPE_W_CACHE.values()),
                    )
            return torch.mm(input.to(out_dtype), w)
        return torch.mm(input.to(out_dtype), mat2.to(out_dtype))

    library.impl("mm.dtype", _kunlun_mm_dtype)
    setattr(mod, _MM_DTYPE_APPLIED_ATTR, library)
    LOGGER.info("Registered V4 aten::mm.dtype fallback (weight cast cached)")
