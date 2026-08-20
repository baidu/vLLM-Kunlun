"""Opt-in FULL-cudagraph switch for DeepSeek-V4 attention builders.

vLLM demotes ``cudagraph_mode=FULL`` to ``FULL_DECODE_ONLY`` unless every
metadata builder advertises ALWAYS support; behind ``KUNLUN_DSV4_FORCE_FULL_CG=1``
(off by default) this patch raises the four DSv4 builders to ALWAYS.
"""
import logging
import sys

LOGGER = logging.getLogger("vllm_kunlun.patches.dsv4_full_cg")

_V4_FULL_CG_BUILDERS: tuple[tuple[str, str, bool], ...] = (
    # (module, class, also_replace_get_cudagraph_support_method)
    ("vllm.v1.attention.backends.mla.indexer", "DeepseekV32IndexerMetadataBuilder", True),
    ("vllm.models.deepseek_v4.sparse_mla", "DeepseekV4FlashMLAMetadataBuilder", False),
    ("vllm.v1.attention.backends.mla.sparse_swa", "DeepseekSparseSWAMetadataBuilder", False),
    (
        "vllm_kunlun.v1.attention.backends.mla.flashmla_sparse",
        "FlashMLASparseMetadataBuilder",
        False,
    ),
)


def _predicate(mod: object) -> bool:
    # Always re-run: builders become importable at different times, and the
    # per-class guard below keeps each scan idempotent.
    return False


def _applier(mod: object) -> None:
    from vllm.v1.attention.backend import AttentionCGSupport

    patched = []
    for mod_name, cls_name, override_method in _V4_FULL_CG_BUILDERS:
        target_mod = sys.modules.get(mod_name)
        if target_mod is None:
            continue
        cls = getattr(target_mod, cls_name, None)
        if cls is None or getattr(cls, "_kunlun_full_cg_patched", False):
            continue
        cls._cudagraph_support = AttentionCGSupport.ALWAYS
        if override_method:
            cls.get_cudagraph_support = classmethod(
                lambda cls, vllm_config, kv_cache_spec: AttentionCGSupport.ALWAYS
            )
        cls._kunlun_full_cg_patched = True
        patched.append(cls_name)

    if patched:
        LOGGER.info(
            "KUNLUN_DSV4_FORCE_FULL_CG=1: cudagraph_support -> ALWAYS for %s",
            ", ".join(patched),
        )
