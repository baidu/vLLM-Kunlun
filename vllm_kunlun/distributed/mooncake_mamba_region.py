"""Fix MambaSpec region registration in MooncakeConnector (no vllm edits).

Upstream ``MooncakeConnectorWorker.register_kv_caches`` unpacks MambaSpec
layers with ``conv, _ = cache_or_caches``, which assumes the old
``(conv_state, ssm_state)`` layout. The current ``gpu_model_runner`` instead
hands out a single contiguous ``[num_blocks, 1, 1, page_size_bytes]`` int8
page view per mamba layer -- the method's own ``dict[str, torch.Tensor]``
annotation already contradicts that unpacking.

Kimi-K3 is an MLA + KDA hybrid with two KV cache groups, and the KDA group's
recurrent state has to be transferred as well, otherwise the decode instance
starts from the wrong state.

Rather than duplicating the dozens of lines of upstream registration logic,
wrap each mamba layer tensor as ``(page, page)`` at the entry point so that
the upstream unpacking lands on the whole page. This keeps the patch minimal
and resilient to upstream churn.
"""

import logging

import torch

logger = logging.getLogger("vllm_kunlun")


def applied(mod) -> bool:
    cls = getattr(mod, "MooncakeConnectorWorker", None)
    if cls is None:
        return True
    return getattr(cls.register_kv_caches, "_kunlun_mamba_region_fix", False)


def apply(mod) -> None:
    from vllm.v1.kv_cache_interface import MambaSpec

    cls = getattr(mod, "MooncakeConnectorWorker", None)
    if cls is None:
        return
    orig = cls.register_kv_caches
    if getattr(orig, "_kunlun_mamba_region_fix", False):
        return

    def register_kv_caches(self, kv_caches, *args, **kwargs):
        fixed = {}
        wrapped = 0
        for name, cache in kv_caches.items():
            spec = getattr(self, "_layer_specs", {}).get(name)
            if isinstance(spec, MambaSpec) and torch.is_tensor(cache):
                # Upstream does `conv, _ = cache_or_caches`; hand it a
                # 2-tuple so the `conv` it picks up is the whole page view.
                fixed[name] = (cache, cache)
                wrapped += 1
            else:
                fixed[name] = cache
        logger.info(
            "[KunlunPlugin] mooncake mamba region fix: %d/%d layer(s) wrapped",
            wrapped,
            len(kv_caches),
        )
        return orig(self, fixed, *args, **kwargs)

    register_kv_caches._kunlun_mamba_region_fix = True
    cls.register_kv_caches = register_kv_caches
    logger.info(
        "[KunlunPlugin] patched MooncakeConnectorWorker.register_kv_caches "
        "(mamba region single-page layout)"
    )
