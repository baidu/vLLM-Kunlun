"""DeepSeek V4 KV-cache compatibility patch."""

from typing import Callable, List

_PATCHED = "_kunlun_dsv4_cache_patch_applied"


def is_applied(mod: object) -> bool:
    return bool(getattr(mod, _PATCHED, False))


def apply(mod: object) -> None:
    if is_applied(mod):
        return

    from vllm_kunlun.ops.attention.cache_utils import (
        dequantize_and_gather_k_cache_pytorch,
    )
    from vllm_kunlun.ops.attention.sparse_index import (
        combine_topk_swa_indices_pytorch,
        compute_global_topk_indices_and_lens_pytorch,
    )

    def dequantize_and_gather_k_cache(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
        use_fnuz: bool = False,
    ):
        return dequantize_and_gather_k_cache_pytorch(
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            block_size,
            offset,
            use_fnuz=use_fnuz,
        )

    dequantize_and_gather_k_cache._kunlun_patched_v2 = True
    mod.dequantize_and_gather_k_cache = dequantize_and_gather_k_cache
    if hasattr(mod, "dequantize_and_gather_k_cache_triton"):
        mod.dequantize_and_gather_k_cache_triton = dequantize_and_gather_k_cache
    if hasattr(mod, "compute_global_topk_indices_and_lens"):
        mod.compute_global_topk_indices_and_lens = compute_global_topk_indices_and_lens_pytorch
    if hasattr(mod, "combine_topk_swa_indices"):
        mod.combine_topk_swa_indices = combine_topk_swa_indices_pytorch
    setattr(mod, _PATCHED, True)


def register(register_post_import_hook: Callable[..., None]) -> List[str]:
    targets = (
        "vllm.models.deepseek_v4.common.ops.cache_utils",
        "vllm.models.deepseek_v4.common.ops",
        "vllm.models.deepseek_v4.nvidia.flashmla",
    )
    for target in targets:
        register_post_import_hook(target, is_applied, apply)
    return list(targets)
