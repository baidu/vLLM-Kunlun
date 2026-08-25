"""Post-import patch registry in the upstream ``compat_patches`` style.

Each entry is ``(label, target, flag, applied, apply)`` with function-object
references.  ``_PLATFORM_HOOKS`` is unconditional (every model on Kunlun XPU);
``_DSV4_HOOKS`` is gated by a
:class:`vllm_kunlun.config.deepseek_v4.FeatureFlags` property (empty flag
name means always-on).  Feature modules that cannot be imported eagerly
(import rings, early platform discovery) are resolved at first dispatch via
:func:`_lazy_feature`, the equivalent of upstream's import-for-side-effect
style; the one early-importing target (``vllm._custom_ops``) uses the same
settle-guard-and-retry pattern as upstream's OOT registration hook.
Everything registers through :func:`populate_platform_hooks` /
:func:`populate_hooks` at platform-import time -- there is no second
registration window.
"""
import importlib
import logging
import os
import sys
from typing import Callable, Tuple

LOGGER = logging.getLogger("vllm_kunlun.patches.registry")

HookPair = Tuple[Callable[[object], bool], Callable[[object], None]]

def _module_busy(mod: object) -> bool:
    """True while *mod* is still executing its import (partial module)."""
    spec = getattr(mod, "__spec__", None)
    return spec is not None and getattr(spec, "_initializing", False)


def _lazy_feature(module_path: str, predicate_name: str, apply_name: str) -> HookPair:
    """Resolve a predicate/applier pair from *module_path* at first dispatch.

    The feature module is imported only when the dispatcher asks, keeping
    registration itself free of heavy imports (the fused-moe graph and the
    ops package are not safe to enter during platform discovery).  While the
    fused-moe import graph has not settled the pair answers "not yet"
    without importing, so the boot-time import ring produces quiet retries
    instead of error-storm logs.
    """

    def applied(mod: object) -> bool:
        if not _fused_moe_graph_settled():
            return False
        feature = importlib.import_module(module_path)
        return bool(getattr(feature, predicate_name)(mod))

    def do_apply(mod: object) -> None:
        if not _fused_moe_graph_settled():
            return
        feature = importlib.import_module(module_path)
        getattr(feature, apply_name)(mod)

    return applied, do_apply


# ---------------------------------------------------------------------------
# Platform-level patches (unconditional; every model on Kunlun XPU).
# ---------------------------------------------------------------------------

def _fp8_kernels_applied(mod: object) -> bool:
    from vllm.platforms import PlatformEnum

    fp8_kernels = getattr(mod, "_POSSIBLE_FP8_KERNELS", None)
    fp8_block_kernels = getattr(mod, "_POSSIBLE_FP8_BLOCK_KERNELS", None)
    return (
        fp8_kernels is not None
        and fp8_block_kernels is not None
        and PlatformEnum.OOT in fp8_kernels
        and PlatformEnum.OOT in fp8_block_kernels
    )


def _fused_moe_graph_settled() -> bool:
    """True once vllm finished importing the fused-moe method module.

    Importing vllm_kunlun.ops (directly or via quantization/fused-moe layers)
    pulls the upstream fused-moe graph, which is circular through
    vllm._aiter_ops during early boot; entering it mid-cycle fails and
    retries forever.
    """
    ufmm = sys.modules.get(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
    )
    if ufmm is None:
        return False
    spec = getattr(ufmm, "__spec__", None)
    return spec is None or not getattr(spec, "_initializing", False)


def _fp8_kernels_apply(mod: object) -> None:
    if not _fused_moe_graph_settled():
        return
    try:
        import vllm_kunlun.quantization.kernels  # noqa: F401  (registers on import)
    except ImportError:
        return  # mid-cycle through another edge; a later dispatch retries


def _fp8_moe_method_applied(mod: object) -> bool:
    cls = getattr(mod, "Fp8MoEMethod", None)
    return cls is None or getattr(cls, "_kunlun_fp8_moe", False)


def _fp8_moe_method_apply(mod: object) -> None:
    if not hasattr(mod, "Fp8MoEMethod"):
        return
    if not _fused_moe_graph_settled():
        return
    try:
        layer_mod = importlib.import_module("vllm_kunlun.ops.fused_moe.layer")
    except ImportError:
        return  # mid-cycle through another edge; a later dispatch retries
    KunlunFp8MoEMethod = getattr(layer_mod, "KunlunFp8MoEMethod", None)
    if KunlunFp8MoEMethod is None:
        LOGGER.warning("KunlunFp8MoEMethod not available yet (deferred)")
        return
    mod.Fp8MoEMethod = KunlunFp8MoEMethod
    LOGGER.info("Substituted KunlunFp8MoEMethod for upstream Fp8MoEMethod")


# These warmups invoke Intel/CUDA Triton sparse-MLA kernels; Kunlun's sparse
# attention path is patched separately and must remain enabled.
_SPARSE_MLA_WARMUPS = (
    "sparse_mla_triton_warmup_if_needed",
    "flashinfer_sparse_mla_decode_autotune_warmup",
    "deepseek_v4_sparse_mla_attention_warmup",
)


def _sparse_mla_warmup_applied(mod: object) -> bool:
    return all(
        getattr(getattr(mod, name, None), "_kunlun_patched", False)
        for name in _SPARSE_MLA_WARMUPS
    )


def _sparse_mla_warmup_apply(mod: object) -> None:
    def _noop(*args, **kwargs):
        LOGGER.info("Skipping non-Kunlun sparse MLA warmup")

    _noop._kunlun_patched = True
    for name in _SPARSE_MLA_WARMUPS:
        setattr(mod, name, _noop)


def _worker_kv_bind_applied(mod: object) -> bool:
    fn = getattr(mod, "bind_kv_cache", None)
    return fn is not None and getattr(fn, "__module__", "") == __name__


def _worker_kv_bind_apply(mod: object) -> None:
    def bind_kv_cache_kunlun(
        kv_caches,
        forward_context,
        runner_kv_caches,
        num_attn_module=1,
    ):
        assert len(runner_kv_caches) == 0
        index2name = mod.defaultdict(list)
        for layer_name in kv_caches:
            layer_index = mod.extract_layer_index(layer_name, num_attn_module)
            index2name[layer_index].append(layer_name)

        for layer_index in sorted(index2name):
            for layer_name in index2name[layer_index]:
                runner_kv_caches.append(kv_caches[layer_name])

        for layer_name, kv_cache in kv_caches.items():
            forward_context[layer_name].kv_cache = kv_cache

    mod.bind_kv_cache = bind_kv_cache_kunlun
    LOGGER.info("Patched OOT multi-cache binding")


_PLATFORM_HOOKS = (
    # (label, target, flag, applied, apply)
    ("platform.fp8_kernels", "vllm.model_executor.kernels.linear", "",
     _fp8_kernels_applied, _fp8_kernels_apply),
    ("platform.fp8_moe_method",
     "vllm.model_executor.layers.quantization.fp8", "",
     _fp8_moe_method_applied, _fp8_moe_method_apply),
    ("platform.sparse_mla_warmup",
     "vllm.model_executor.warmup.kernel_warmup", "",
     _sparse_mla_warmup_applied, _sparse_mla_warmup_apply),
    ("platform.worker_kv_bind", "vllm.v1.worker.utils", "",
     _worker_kv_bind_applied, _worker_kv_bind_apply),
    ("platform.op_inventory", "vllm.v1.worker.gpu_model_runner", "",
     *_lazy_feature("vllm_kunlun.adapter_utils",
                    "_op_inventory_applied", "_op_inventory_apply")),
)


# ---------------------------------------------------------------------------
# DSV4 patches with inline logic (upstream "direct" style).
# ---------------------------------------------------------------------------

def _mla_layout_applied(mod: object) -> bool:
    return bool(getattr(mod, "_kunlun_dsv4_layout_patched", False))


def _mla_layout_apply(mod: object) -> None:
    """Flip ``use_fp8_ds_mla_layout=False`` on every DSv4 attention subclass.

    Upstream asserts FP8 packed KV-cache pages; Kunlun ships BF16 plain-row
    pages, so every subclass must be flipped before backend dispatch selects
    the bf16 path.
    """
    import inspect

    for _name, _cls in list(mod.__dict__.items()):
        if not inspect.isclass(_cls):
            continue
        if _cls.__module__ != mod.__name__:
            continue
        if not hasattr(_cls, "use_fp8_ds_mla_layout"):
            continue
        try:
            _cls.use_fp8_ds_mla_layout = False
        except Exception:  # noqa: BLE001
            pass
    setattr(mod, "_kunlun_dsv4_layout_patched", True)
    LOGGER.info(
        "Flipped use_fp8_ds_mla_layout=False on %s subclasses", mod.__name__
    )


_INDEXER_Q_APPLIED_ATTR = "_kunlun_indexer_q_patched"


def _indexer_q_applied(mod: object) -> bool:
    fn = getattr(mod, "fused_indexer_q_rope_quant", None)
    fn_okay = fn is not None and getattr(fn, "__module__", "") == "vllm_kunlun.ops.fp8"
    return bool(getattr(mod, _INDEXER_Q_APPLIED_ATTR, False)) and bool(fn_okay)


def _indexer_q_apply(mod: object) -> None:
    """Point ``fused_indexer_q_rope_quant`` at the Kunlun FP8 implementation."""
    from vllm_kunlun.ops.fp8 import fused_indexer_q_rope_quant_kunlun

    mod.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_kunlun
    setattr(mod, _INDEXER_Q_APPLIED_ATTR, True)
    LOGGER.info("Patched V4 indexer Q RoPE/FP8 quantization")


_MM_DTYPE_APPLIED_ATTR = "_kunlun_mm_dtype_library"

_MM_DTYPE_W_CACHE: "dict[tuple, object]" = {}


def _mm_dtype_applied(mod: object) -> bool:
    return getattr(mod, _MM_DTYPE_APPLIED_ATTR, None) is not None


def _mm_dtype_apply(mod: object) -> None:
    """Register an ``aten::mm.dtype`` impl that casts inputs then calls torch.mm.

    The V4 callers pass ``weight.T`` (mat2._base is a Parameter), so the
    weight cast is cacheable and would remove a per-layer transpose+cast
    from every step. The fp32 copy is opt-in (KUNLUN_DSV4_MM_DTYPE_WCACHE=1,
    together with a higher --gpu-memory-utilization) because vLLM sizes the
    KV pool from post-profile free memory: a persistent cache directly
    shrinks the pool. Non-Parameter mat2 always casts per call.
    """
    torch = mod.torch
    library = torch.library.Library("aten", "IMPL", "CUDA")

    def _kunlun_mm_dtype(input, mat2, out_dtype):
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
                        sum(
                            t.numel() * t.element_size()
                            for t in _MM_DTYPE_W_CACHE.values()
                        ),
                    )
            return torch.mm(input.to(out_dtype), w)
        return torch.mm(input.to(out_dtype), mat2.to(out_dtype))

    library.impl("mm.dtype", _kunlun_mm_dtype)
    setattr(mod, _MM_DTYPE_APPLIED_ATTR, library)
    LOGGER.info("Registered V4 aten::mm.dtype fallback (weight cast cached)")


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


def _full_cg_applied(mod: object) -> bool:
    # Always re-run: builders become importable at different times, and the
    # per-class guard inside the applier keeps each scan idempotent.
    return False


def _full_cg_apply(mod: object) -> None:
    """Raise the four DSv4 metadata builders to ALWAYS cudagraph support.

    vLLM demotes ``cudagraph_mode=FULL`` to ``FULL_DECODE_ONLY`` unless every
    builder advertises ALWAYS support; behind KUNLUN_DSV4_FORCE_FULL_CG=1
    (off by default) this patch raises them.
    """
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


_DSV4_CACHE_PATCHED = "_kunlun_dsv4_cache_patch_applied"


# --- platform policy: wrap KunlunPlatform.check_and_update_config --------

_PLATFORM_POLICY_DONE = "__kunlun_dsv4_platform_policy__"


def _platform_policy_applied(mod: object) -> bool:
    return bool(getattr(mod, _PLATFORM_POLICY_DONE, False))


def _platform_policy_apply(mod: object) -> None:
    """Wrap ``KunlunPlatform.check_and_update_config`` once per process.

    Fires on the gpu_worker import (vLLM core settled, platform resolution
    finished); the policy module re-checks its own idempotence sentinel on
    the platform class.
    """
    from vllm_kunlun.models import deepseek_v4_policy

    deepseek_v4_policy.apply()
    setattr(mod, _PLATFORM_POLICY_DONE, True)


# --- MoE hash/softplus routing: torch.ops shim + direct router -----------
# vllm._custom_ops imports early, before the fused-moe graph settles, so the
# applier guards and retries on a later dispatch (upstream's OOT pattern).
# The predicates are pure getattr on the target and need no feature import.

def _moe_hash_enabled() -> bool:
    from vllm_kunlun.config.deepseek_v4 import FeatureFlags

    flags = FeatureFlags()
    return bool(flags.hash_topk_fused or flags.activation_routing_accel)


def _moe_hash_custom_applied(mod: object) -> bool:
    if not _moe_hash_enabled():
        return True
    return getattr(getattr(mod, "topk_hash_softplus_sqrt", None), "_dsv4_wired", False)


def _moe_hash_custom_apply(mod: object) -> None:
    if not _moe_hash_enabled():
        return
    if not _fused_moe_graph_settled():
        return
    try:
        from vllm_kunlun.ops.fused_moe import moe_hash_router
    except ImportError:
        return  # mid-cycle through another edge; a later dispatch retries
    moe_hash_router._install_custom_ops_shim(mod)


def _moe_hash_router_applied(mod: object) -> bool:
    if not _moe_hash_enabled():
        return True
    return getattr(getattr(mod, "vllm_topk_softplus_sqrt", None), "_dsv4_wired", False)


def _moe_hash_router_apply(mod: object) -> None:
    if not _moe_hash_enabled():
        return
    if not _fused_moe_graph_settled():
        return
    try:
        from vllm_kunlun.ops.fused_moe import moe_hash_router
    except ImportError:
        return  # mid-cycle through another edge; a later dispatch retries
    moe_hash_router._install_direct_router(mod)


# --- optional diagnostics (KUNLUN_DSV4_DEBUG=1) ---------------------------

def _debug_enabled() -> bool:
    return os.getenv("KUNLUN_DSV4_DEBUG", "0") == "1"


def _logits_debug_applied(mod: object) -> bool:
    if not _debug_enabled():
        return True
    fn = getattr(getattr(mod, "LogitsProcessor", None), "_gather_logits", None)
    return bool(getattr(fn, "_kunlun_dsv4_debug", False))


def _logits_debug_apply(mod: object) -> None:
    from vllm_kunlun.patches import dsv4_logits_debug

    dsv4_logits_debug.apply(mod)


def _runner_debug_applied(mod: object) -> bool:
    if not _debug_enabled():
        return True
    fn = getattr(getattr(mod, "GPUModelRunner", None), "execute_model", None)
    return bool(getattr(fn, "_kunlun_dsv4_debug", False))


def _runner_debug_apply(mod: object) -> None:
    from vllm_kunlun.patches import dsv4_runner_debug

    dsv4_runner_debug.apply(mod)


def _cache_applied(mod: object) -> bool:
    return bool(getattr(mod, _DSV4_CACHE_PATCHED, False))


def _cache_apply(mod: object) -> None:
    """Bind the pytorch KV-cache gather/topk helpers over the Triton ones."""
    if _cache_applied(mod):
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
    setattr(mod, _DSV4_CACHE_PATCHED, True)


# ---------------------------------------------------------------------------
# DSV4 hook tables.  Entries whose feature module owns the logic use
# _lazy_feature(); the rest carry inline pairs defined above.
# ---------------------------------------------------------------------------

_DSV4_HOOKS = (
    # (label, target, flag, applied, apply)
    ("dsv4.norms.model", "vllm.models.deepseek_v4.nvidia.model",
     "rmsnorm_shortcut",
     *_lazy_feature("vllm_kunlun.ops.attention.norms",
                    "_model_predicate", "_model_apply")),
    ("dsv4.norms.attention", "vllm.models.deepseek_v4.attention",
     "rmsnorm_shortcut",
     *_lazy_feature("vllm_kunlun.ops.attention.norms",
                    "_attn_predicate", "_attn_apply")),
    ("dsv4.oproj", "vllm.models.deepseek_v4.nvidia.ops.o_proj",
     "oproj_native",
     *_lazy_feature("vllm_kunlun.ops.attention.o_proj_alias",
                    "_predicate", "_applier")),
    ("dsv4.oproj.flashmla", "vllm.models.deepseek_v4.nvidia.flashmla",
     "oproj_native",
     *_lazy_feature("vllm_kunlun.ops.attention.o_proj_alias",
                    "_predicate", "_applier")),
    ("dsv4.oproj.flashinfer", "vllm.models.deepseek_v4.nvidia.flashinfer_sparse",
     "oproj_native",
     *_lazy_feature("vllm_kunlun.ops.attention.o_proj_alias",
                    "_predicate", "_applier")),
    ("dsv4.mhc.tile", "vllm.model_executor.kernels.mhc.tilelang",
     "mhc_tilelang_native",
     *_lazy_feature("vllm_kunlun.ops.fused_moe.mhc_hyperconnection",
                    "_tilelang_predicate", "_tilelang_applier")),
    ("dsv4.mhc.model", "vllm.models.deepseek_v4.nvidia.model",
     "mhc_model_native",
     *_lazy_feature("vllm_kunlun.ops.fused_moe.mhc_hyperconnection",
                    "_model_applied", "_model_applier")),
    ("dsv4.flashmla.metadata.ops", "vllm.v1.attention.ops.flashmla",
     "flashmla_metadata_native",
     *_lazy_feature("vllm_kunlun.ops.attention.flashmla_bridge",
                    "_flashmla_metadata_predicate", "_flashmla_metadata_applier")),
    ("dsv4.flashmla.metadata.sparse_backend",
     "vllm.v1.attention.backends.mla.flashmla_sparse",
     "flashmla_metadata_native",
     *_lazy_feature("vllm_kunlun.ops.attention.flashmla_bridge",
                    "_flashmla_metadata_predicate", "_flashmla_metadata_applier")),
    ("dsv4.flashmla.metadata.sparse_swa",
     "vllm.v1.attention.backends.mla.sparse_swa",
     "flashmla_metadata_native",
     *_lazy_feature("vllm_kunlun.ops.attention.flashmla_bridge",
                    "_flashmla_metadata_predicate", "_flashmla_metadata_applier")),
    ("dsv4.flashmla.metadata.nvidia", "vllm.models.deepseek_v4.nvidia.flashmla",
     "flashmla_metadata_native",
     *_lazy_feature("vllm_kunlun.ops.attention.flashmla_bridge",
                    "_flashmla_metadata_predicate", "_flashmla_metadata_applier")),
    ("dsv4.flashmla.padded_heads", "vllm.models.deepseek_v4.nvidia.flashmla",
     "flashmla_padded_heads_native",
     *_lazy_feature("vllm_kunlun.ops.attention.flashmla_bridge",
                    "_flashmla_padded_heads_predicate", "_flashmla_padded_heads_applier")),
    ("dsv4.mla.layout", "vllm.models.deepseek_v4.attention",
     "mla_layout_native", _mla_layout_applied, _mla_layout_apply),
    ("dsv4.indexer.q", "vllm.models.deepseek_v4.attention",
     "indexer_q_native", _indexer_q_applied, _indexer_q_apply),
    ("dsv4.mm.dtype", "vllm.models.deepseek_v4.attention",
     "mm_dtype_native", _mm_dtype_applied, _mm_dtype_apply),
    ("dsv4.indexer.sparse", "vllm.model_executor.layers.sparse_attn_indexer",
     "indexer_decode_native",
     *_lazy_feature("vllm_kunlun.ops.attention.indexer_decode",
                    "_applied", "_install_kunlun_indexer")),
    # Attach decode host lens at metadata build time so the indexer forward
    # never blocks on a D2H sync; unconditional like the kv_insert aliases.
    ("dsv4.indexer.host_lens", "vllm.v1.attention.backends.mla.indexer",
     "",
     *_lazy_feature("vllm_kunlun.ops.attention.indexer_decode",
                    "_builder_host_lens_applied", "_install_indexer_builder_host_lens")),
    # Standalone contiguous KV cache tensors instead of the page-packed
    # backing; planner capacity math is unchanged.
    ("dsv4.kv_unpack.planner", "vllm.v1.core.kv_cache_utils",
     "kv_cache_unpack",
     *_lazy_feature("vllm_kunlun.patches.dsv4_kv_unpack",
                    "_planner_predicate", "_planner_applier")),
    # indexer cache dtype int8: XPU index_put_ lacks native uint8 dst support
    # and round-trips the whole cache view through int8 (see feature module).
    ("dsv4.indexer.int8_cache", "vllm.models.deepseek_v4.attention",
     "indexer_cache_int8",
     *_lazy_feature("vllm_kunlun.patches.dsv4_kv_unpack",
                    "_dtype_predicate", "_dtype_applier")),
    # Drop the indexer cache's page alignment padding so the cache tensor is
    # truly contiguous (prerequisite for the native indexer store op).
    ("dsv4.indexer.unpad", "vllm.models.deepseek_v4.attention",
     "indexer_cache_int8",
     *_lazy_feature("vllm_kunlun.patches.dsv4_kv_unpack",
                    "_unpad_predicate", "_unpad_applier")),
    # Community code calls these via torch.ops._C directly; the alias binding
    # is mandatory whenever DSV4 runs, so no feature flag gates it.
    ("dsv4.kv_insert.attention", "vllm.models.deepseek_v4.attention",
     "",
     *_lazy_feature("vllm_kunlun.ops.attention.kv_insert_paths",
                    "_alias_predicate", "_alias_applier")),
    ("dsv4.kv_insert.model", "vllm.models.deepseek_v4.nvidia.model",
     "",
     *_lazy_feature("vllm_kunlun.ops.attention.kv_insert_paths",
                    "_alias_predicate", "_alias_applier")),
    ("dsv4.full_cg.sparse_mla", "vllm.models.deepseek_v4.sparse_mla",
     "force_full_cg", _full_cg_applied, _full_cg_apply),
    ("dsv4.full_cg.sparse_swa", "vllm.v1.attention.backends.mla.sparse_swa",
     "force_full_cg", _full_cg_applied, _full_cg_apply),
    ("dsv4.full_cg.attn_utils", "vllm.v1.worker.gpu.attn_utils",
     "force_full_cg", _full_cg_applied, _full_cg_apply),
    # Pytorch KV-cache gather/topk helper bindings (was patches/dsv4_cache).
    ("dsv4.cache.cache_utils", "vllm.models.deepseek_v4.common.ops.cache_utils",
     "", _cache_applied, _cache_apply),
    ("dsv4.cache.ops", "vllm.models.deepseek_v4.common.ops",
     "", _cache_applied, _cache_apply),
    ("dsv4.cache.flashmla", "vllm.models.deepseek_v4.nvidia.flashmla",
     "", _cache_applied, _cache_apply),
    # Compressed-KV pipeline (gate is an OR of two flags, kept inside the
    # feature module's predicates -- sticky True when both are off).
    ("dsv4.compressor.slot.compressor_utils",
     "vllm.v1.attention.backends.mla.compressor_utils", "",
     *_lazy_feature("vllm_kunlun.ops.attention.compressor",
                    "_slot_mapping_applied", "_install_slot_mapping")),
    ("dsv4.compressor.slot.indexer",
     "vllm.v1.attention.backends.mla.indexer", "",
     *_lazy_feature("vllm_kunlun.ops.attention.compressor",
                    "_slot_mapping_applied", "_install_slot_mapping")),
    ("dsv4.compressor.save_partial_states",
     "vllm.models.deepseek_v4.common.ops.save_partial_states", "",
     *_lazy_feature("vllm_kunlun.ops.attention.compressor",
                    "_sps_applied", "_install_save_partial_states")),
    ("dsv4.compressor.vect",
     "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache", "",
     *_lazy_feature("vllm_kunlun.ops.attention.compressor",
                    "_vect_applied", "_install_compress_norm_rope_store_triton")),
    # Sparse-MLA metadata shims (indexer prefill, c128a, SWA kernels).
    ("dsv4.flashmla.indexer_prefill",
     "vllm.v1.attention.backends.mla.indexer", "flashmla_sparse_backend",
     *_lazy_feature("vllm_kunlun.ops.attention.flashmla_bridge",
                    "_indexer_prefill_applied", "_install_indexer_prefill_kernel")),
    ("dsv4.flashmla.c128a", "vllm.models.deepseek_v4.sparse_mla",
     "flashmla_sparse_backend",
     *_lazy_feature("vllm_kunlun.ops.attention.flashmla_bridge",
                    "_c128a_applied", "_install_c128a_metadata")),
    ("dsv4.flashmla.swa_kernel",
     "vllm.v1.attention.backends.mla.sparse_swa", "flashmla_sparse_backend",
     *_lazy_feature("vllm_kunlun.ops.attention.flashmla_bridge",
                    "_swa_kernel_applied", "_install_swa_kernel")),
    ("dsv4.flashmla.swa_prefill_lens",
     "vllm.v1.attention.backends.mla.sparse_swa", "flashmla_sparse_backend",
     *_lazy_feature("vllm_kunlun.ops.attention.flashmla_bridge",
                    "_swa_prefill_lens_applied", "_install_prefill_gather_lenses")),
    # MoE hash/softplus routing (inline pairs: settle-guarded deferred import).
    ("dsv4.moe_hash.custom_ops", "vllm._custom_ops", "",
     _moe_hash_custom_applied, _moe_hash_custom_apply),
    ("dsv4.moe_hash.router",
     "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router", "",
     _moe_hash_router_applied, _moe_hash_router_apply),
    # Platform config policy wrap (fires at gpu_worker import, like the old
    # fire-once lazy entry -- but with a real predicate instead of relying
    # on the memory-pool hook's chain to carry it).
    ("dsv4.platform_policy", "vllm.v1.worker.gpu_worker", "",
     _platform_policy_applied, _platform_policy_apply),
    # Optional low-overhead diagnostics.
    ("dsv4.logits_debug", "vllm.model_executor.layers.logits_processor", "",
     _logits_debug_applied, _logits_debug_apply),
    ("dsv4.runner_debug", "vllm.v1.worker.gpu_model_runner", "",
     _runner_debug_applied, _runner_debug_apply),
)


# Per-label consecutive applier failure counters, throttled like the
# dispatcher's own failure logs (first attempt + every 200th).
_apply_failures: dict = {}


def _flag_on(flag_name: str) -> bool:
    """Resolve one FeatureFlags property; empty name means always-on."""
    if not flag_name:
        return True
    from vllm_kunlun.config.deepseek_v4 import FeatureFlags
    return bool(getattr(FeatureFlags(), flag_name))


def _try_apply(label: str, applier: Callable[[object], None], mod: object) -> None:
    """Run one feature applier, isolating and throttling its failures."""
    try:
        applier(mod)
    except Exception:  # noqa: BLE001
        _apply_failures[label] = n = _apply_failures.get(label, 0) + 1
        if n == 1 or n % 200 == 0:
            LOGGER.exception("patch failed: %s (attempt %d)", label, n)
        return
    LOGGER.debug("Applied %s to %s", label, getattr(mod, "__name__", mod))


def _register_table(
    table: tuple,
    register_post_import_hook: Callable[..., None],
) -> None:
    """Register one aggregated hook per target with the dispatcher.

    This layer honors the dispatcher's one-registration-per-target rule:
    entries sharing a target are composed into a single predicate/applier
    pair here, so only feature modules and eager adapters registering later
    stack through the dispatcher's chain-merge.  The aggregated predicate
    reports True while the target is mid-import or every gated feature is
    either applied or flag-off (sticky); the applier skips flag-off entries
    and isolates per-feature failures.
    """
    groups: dict = {}
    for label, target, flag_name, applied, apply in table:
        groups.setdefault(target, []).append((label, flag_name, applied, apply))

    for target, entries in groups.items():
        def _applied(mod, _entries=entries):
            if _module_busy(mod):
                return True
            for _label, flag, predicate, _apply in _entries:
                if _flag_on(flag) and not predicate(mod):
                    return False
            return True

        def _apply(mod, _entries=entries):
            for label, flag, _predicate, applier in _entries:
                if _flag_on(flag):
                    _try_apply(label, applier, mod)

        register_post_import_hook(target, _applied, _apply)


def populate_platform_hooks(register_post_import_hook: Callable[..., None]) -> None:
    """Register every unconditional platform-level patch with the dispatcher."""
    _register_table(_PLATFORM_HOOKS, register_post_import_hook)


def populate_hooks(register_post_import_hook: Callable[..., None]) -> None:
    """Register every DSV4 hook with the dispatcher (idempotent per process)."""
    _register_table(_DSV4_HOOKS, register_post_import_hook)
