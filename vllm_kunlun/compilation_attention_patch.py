"""Patch CompilationConfig._attention_ops for Kunlun unified attention custom op."""

from __future__ import annotations


def patch_compilation_attention_ops_for_kunlun() -> None:
    """Append Kunlun attention op names so V1 splitting/CUDA graphs treat them like stock ops."""
    try:
        from vllm.config.compilation import CompilationConfig
    except Exception:
        return

    if getattr(CompilationConfig, "_kunlun_attention_ops_patched", False):
        return

    ops = getattr(CompilationConfig, "_attention_ops", None)
    if not isinstance(ops, list):
        return

    dotted = "vllm.unified_attention_with_output_kunlun"
    colon = "vllm::unified_attention_with_output_kunlun"

    if dotted not in ops:
        ops.append(dotted)
    if colon not in ops:
        ops.append(colon)

    CompilationConfig._kunlun_attention_ops_patched = True
