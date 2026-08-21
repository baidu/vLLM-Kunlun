"""Standalone (non-packed) KV cache tensors for DSV4.

The stock DSV4 planner (`_get_kv_cache_config_packed`) packs every
(page_size, slot) tier into one physical backing so per-layer views are
page-strided column slices. Consumers only receive those per-tier views and
block-id namespaces stay independent per group, so the packing is physically
— not semantically — load-bearing. Contiguous-only kernels then pay a
full-pool staging copy per layer per step.

This patch keeps the planner's capacity math (same buckets, same num_blocks)
but emits standalone per-slot tensors (no offset / block_stride): the stock
allocator then creates one contiguous tensor per tier via the exact code path
every non-packed vLLM model uses. Set KUNLUN_DSV4_KV_UNPACK=0 to restore the
page-packed layout.
"""
import logging

import torch

LOGGER = logging.getLogger("vllm_kunlun.patches.dsv4_kv_unpack")

_PLANNER_KEY = "_kunlun_kv_unpack_planner_applied"


def _enabled() -> bool:
    from vllm_kunlun.config.deepseek_v4 import FeatureFlags
    return bool(FeatureFlags().kv_cache_unpack)


_DTYPE_KEY = "_kunlun_indexer_int8_applied"


def _dtype_predicate(mod: object) -> bool:
    return bool(getattr(mod, _DTYPE_KEY, False))


def _dtype_applier(mod: object) -> None:
    """indexer cache dtype uint8 -> int8。

    XPU aten index_put_ 不支持 uint8 目的张量，内部做 uint8<->int8 整视图
    往返物化（每次 insert ~3 遍 31MB）；kunlun_ops 的 indexer kernel 家族
    签名本就是 int8_t*。所有消费方均为字节级 view，语义零变化。"""
    if getattr(mod, _DTYPE_KEY, False):
        return
    cache_cls = getattr(mod, "DeepseekV4IndexerCache", None)
    if cache_cls is None:
        return
    orig_init = cache_cls.__init__

    def __init__(self, *args, **kwargs):
        if kwargs.get("dtype") is torch.uint8:
            kwargs["dtype"] = torch.int8
        orig_init(self, *args, **kwargs)

    cache_cls.__init__ = __init__
    setattr(mod, _DTYPE_KEY, True)
    LOGGER.info("Patched DeepseekV4IndexerCache: dtype uint8 -> int8")


def _planner_predicate(mod: object) -> bool:
    return bool(getattr(mod, _PLANNER_KEY, False))


def _planner_applier(mod: object) -> None:
    if getattr(mod, _PLANNER_KEY, False):
        return
    orig = mod._get_kv_cache_config_packed
    KVCacheTensor = mod.KVCacheTensor

    def _get_kv_cache_config_unpacked(vllm_config, kv_cache_groups, available_memory):
        num_blocks, _packed_tensors = orig(vllm_config, kv_cache_groups,
                                           available_memory)
        if not _enabled() or num_blocks == 0:
            return num_blocks, _packed_tensors
        buckets = mod._bucket_layers_by_page_size(kv_cache_groups)
        tensors = []
        for page_size, slots in buckets.items():
            for slot in slots:
                tensors.append(
                    KVCacheTensor(size=page_size * num_blocks, shared_by=slot)
                )
        LOGGER.info(
            "[kv-unpack] planner: num_blocks=%d tensors=%d -> standalone "
            "contiguous (stock packed would alias one backing)", num_blocks,
            len(tensors),
        )
        return num_blocks, tensors

    mod._get_kv_cache_config_packed = _get_kv_cache_config_unpacked
    setattr(mod, _PLANNER_KEY, True)
    LOGGER.info("Patched _get_kv_cache_config_packed: per-slot standalone "
                "tensors instead of page-packed backing")
