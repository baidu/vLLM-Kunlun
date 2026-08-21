"""Static registry of Kunlun post-import patches.

Two tables live here:

* ``_PLATFORM_PATCHES`` -- unconditional platform-level patches (quant kernel
  registries, warmup skips, KV-cache binding) that must apply for every model.
* ``_STATIC_PATCHES`` -- DeepSeek-V4 adapters, each gated by a
  :class:`vllm_kunlun.config.deepseek_v4.FeatureFlags` property (an empty flag
  name means always-on).

Both are wired into the root package's post-import dispatcher via
``populate_platform_hooks()`` / ``populate_hooks()``.
"""
import importlib
import logging
import os
from typing import Callable, List, Tuple

LOGGER = logging.getLogger("vllm_kunlun.patches.registry")
HookSpec = Tuple[str, str, Callable[[object], bool], Callable[[object], None]]
# Targets that will be patched lazily through the plugin dispatcher once their host modules are loaded.
_LAZY_HOOKS: List[Tuple[str, str, Callable[[object], bool], Callable[[object], None]]] = []

_LAZY_DESCRIPTOR = "__dsv4_lazy_applied_sentinel__"

_STATIC_PATCHES = (
    ("dsv4.norms.model", "vllm.models.deepseek_v4.nvidia.model",
     "vllm_kunlun.ops.attention.norms", "_model_predicate", "_model_apply",
     "rmsnorm_shortcut"),
    ("dsv4.norms.attention", "vllm.models.deepseek_v4.attention",
     "vllm_kunlun.ops.attention.norms", "_attn_predicate", "_attn_apply",
     "rmsnorm_shortcut"),
    ("dsv4.oproj", "vllm.models.deepseek_v4.nvidia.ops.o_proj",
     "vllm_kunlun.ops.attention.o_proj_alias", "_predicate", "_applier",
     "oproj_native"),
    ("dsv4.oproj.flashmla", "vllm.models.deepseek_v4.nvidia.flashmla",
     "vllm_kunlun.ops.attention.o_proj_alias", "_predicate", "_applier",
     "oproj_native"),
    ("dsv4.oproj.flashinfer", "vllm.models.deepseek_v4.nvidia.flashinfer_sparse",
     "vllm_kunlun.ops.attention.o_proj_alias", "_predicate", "_applier",
     "oproj_native"),
    ("dsv4.mhc.tile",
     "vllm.model_executor.kernels.mhc.tilelang",
     "vllm_kunlun.ops.fused_moe.mhc_hyperconnection",
     "_tilelang_predicate", "_tilelang_applier",
     "mhc_tilelang_native"),

    ("dsv4.mhc.model",
     "vllm.models.deepseek_v4.nvidia.model",
     "vllm_kunlun.ops.fused_moe.mhc_hyperconnection",
     "_model_applied", "_model_applier",
     "mhc_model_native"),

    ("dsv4.flashmla.metadata.ops", "vllm.v1.attention.ops.flashmla",
     "vllm_kunlun.ops.attention.flashmla_bridge",
     "_flashmla_metadata_predicate", "_flashmla_metadata_applier",
     "flashmla_metadata_native"),

    ("dsv4.flashmla.metadata.sparse_backend",
     "vllm.v1.attention.backends.mla.flashmla_sparse",
     "vllm_kunlun.ops.attention.flashmla_bridge",
     "_flashmla_metadata_predicate", "_flashmla_metadata_applier",
     "flashmla_metadata_native"),

    ("dsv4.flashmla.metadata.sparse_swa",
     "vllm.v1.attention.backends.mla.sparse_swa",
     "vllm_kunlun.ops.attention.flashmla_bridge",
     "_flashmla_metadata_predicate", "_flashmla_metadata_applier",
     "flashmla_metadata_native"),

    ("dsv4.flashmla.metadata.nvidia",
     "vllm.models.deepseek_v4.nvidia.flashmla",
     "vllm_kunlun.ops.attention.flashmla_bridge",
     "_flashmla_metadata_predicate", "_flashmla_metadata_applier",
     "flashmla_metadata_native"),

    ("dsv4.flashmla.padded_heads",
     "vllm.models.deepseek_v4.nvidia.flashmla",
     "vllm_kunlun.ops.attention.flashmla_bridge",
     "_flashmla_padded_heads_predicate", "_flashmla_padded_heads_applier",
     "flashmla_padded_heads_native"),

    ("dsv4.mla.layout", "vllm.models.deepseek_v4.attention",
     "vllm_kunlun.patches.dsv4_mla_layout",
     "_layout_predicate", "_layout_applier",
     "mla_layout_native"),

    ("dsv4.indexer.q", "vllm.models.deepseek_v4.attention",
     "vllm_kunlun.patches.dsv4_attention_subst",
     "_indexer_q_predicate", "_indexer_q_applier",
     "indexer_q_native"),

    ("dsv4.mm.dtype", "vllm.models.deepseek_v4.attention",
     "vllm_kunlun.patches.dsv4_attention_subst",
     "_mm_dtype_predicate", "_mm_dtype_applier",
     "mm_dtype_native"),

    ("dsv4.indexer.sparse", "vllm.model_executor.layers.sparse_attn_indexer",
     "vllm_kunlun.ops.attention.indexer_decode", "_applied", "_install_kunlun_indexer",
     "indexer_decode_native"),
    # Attach decode host lens at metadata build time so the indexer forward
    # never blocks on a D2H sync; unconditional like the kv_insert aliases.
    ("dsv4.indexer.host_lens", "vllm.v1.attention.backends.mla.indexer",
     "vllm_kunlun.ops.attention.indexer_decode",
     "_builder_host_lens_applied", "_install_indexer_builder_host_lens",
     ""),
    # Standalone contiguous KV cache tensors instead of the page-packed
    # backing; planner capacity math is unchanged.
    ("dsv4.kv_unpack.planner", "vllm.v1.core.kv_cache_utils",
     "vllm_kunlun.patches.dsv4_kv_unpack", "_planner_predicate", "_planner_applier",
     "kv_cache_unpack"),
    # indexer cache dtype int8: XPU index_put_ lacks native uint8 dst support and
    # round-trips the whole cache view through int8 (see patch docstring).
    ("dsv4.indexer.int8_cache", "vllm.models.deepseek_v4.attention",
     "vllm_kunlun.patches.dsv4_kv_unpack", "_dtype_predicate", "_dtype_applier",
     "indexer_cache_int8"),
    # Drop the indexer cache's page alignment padding so the cache tensor is
    # truly contiguous (prerequisite for the native indexer store op).
    ("dsv4.indexer.unpad", "vllm.models.deepseek_v4.attention",
     "vllm_kunlun.patches.dsv4_kv_unpack", "_unpad_predicate", "_unpad_applier",
     "indexer_cache_int8"),
    # Community code calls these via torch.ops._C directly; the alias binding
    # is mandatory whenever DSV4 runs, so no feature flag gates it.
    ("dsv4.kv_insert.attention", "vllm.models.deepseek_v4.attention",
     "vllm_kunlun.ops.attention.kv_insert_paths", "_alias_predicate", "_alias_applier",
     ""),
    ("dsv4.kv_insert.model", "vllm.models.deepseek_v4.nvidia.model",
     "vllm_kunlun.ops.attention.kv_insert_paths", "_alias_predicate", "_alias_applier",
     ""),
    ("dsv4.full_cg.sparse_mla", "vllm.models.deepseek_v4.sparse_mla",
     "vllm_kunlun.patches.dsv4_full_cg", "_predicate", "_applier",
     "force_full_cg"),
    ("dsv4.full_cg.sparse_swa", "vllm.v1.attention.backends.mla.sparse_swa",
     "vllm_kunlun.patches.dsv4_full_cg", "_predicate", "_applier",
     "force_full_cg"),
    ("dsv4.full_cg.attn_utils", "vllm.v1.worker.gpu.attn_utils",
     "vllm_kunlun.patches.dsv4_full_cg", "_predicate", "_applier",
     "force_full_cg"),
)


def _module_busy(mod: object) -> bool:
    """True while *mod* is still executing its import (partial module)."""
    spec = getattr(mod, "__spec__", None)
    return spec is not None and getattr(spec, "_initializing", False)


def _register_static_patches(register_post_import_hook: Callable[..., None]) -> None:
    for label, target, module_path, predicate_name, apply_name, flag_name in _STATIC_PATCHES:
        def _enabled(_flag_name=flag_name):
            if not _flag_name:
                return True
            from vllm_kunlun.config.deepseek_v4 import FeatureFlags
            return bool(getattr(FeatureFlags(), _flag_name))

        def _applied(mod, _module_path=module_path, _predicate_name=predicate_name,
                     _enabled=_enabled):
            if _module_busy(mod) or not _enabled():
                return True
            feature = importlib.import_module(_module_path)
            return bool(getattr(feature, _predicate_name)(mod))

        def _apply(mod, _label=label, _module_path=module_path,
                   _apply_name=apply_name, _enabled=_enabled):
            if not _enabled():
                return
            feature = importlib.import_module(_module_path)
            getattr(feature, _apply_name)(mod)
            LOGGER.debug("Applied %s to %s", _label, getattr(mod, "__name__", mod))

        register_post_import_hook(target, _applied, _apply)


def _register_lazy(
    target_module_path: str,
    applied_test: Callable[[object], bool],
    applier: Callable[[object], None],
    *,
    label: str | None = None,
) -> None:
    """Record a hook that should fire when *target_module_path* appears in sys.modules."""
    effective_label = label or target_module_path
    _LAZY_HOOKS.append((effective_label, target_module_path, applied_test, applier))


def populate_hooks(register_post_import_hook: Callable[..., None]) -> List[str]:
    """Register every DSV4 hook known so far and eagerly install cheap ones.

    Args:
        register_post_import_hook: callback provided by root package used to queue
            patches against imported community modules.
    Returns:
        Labels of adapters successfully registered on this invocation.
    """
    labels_installed: List[str] = []

    # platform_policy must stay lazy: importing KunlunPlatform eagerly would
    # re-enter platform resolution before current_platform is set.
    _register_lazy(
        "vllm.v1.worker.gpu_worker",
        lambda m: True,  # fire once, platform_policy.apply() is idempotent
        lambda m: __import__("vllm_kunlun.models.deepseek_v4_policy", fromlist=["apply"]).apply(),
        label="dsv4.platform_policy",
    )
    if os.getenv("KUNLUN_DSV4_DEBUG", "0") == "1":
        _register_lazy(
            "vllm.model_executor.layers.logits_processor",
            lambda m: getattr(m.LogitsProcessor._gather_logits, "_kunlun_dsv4_debug", False),
            lambda m: __import__(
                "vllm_kunlun.patches.dsv4_logits_debug", fromlist=["apply"]
            ).apply(m),
            label="dsv4.logits_debug",
        )
        _register_lazy(
            "vllm.v1.worker.gpu_model_runner",
            lambda m: getattr(m.GPUModelRunner.execute_model, "_kunlun_dsv4_debug", False),
            lambda m: __import__(
                "vllm_kunlun.patches.dsv4_runner_debug", fromlist=["apply"]
            ).apply(m),
            label="dsv4.runner_debug",
        )

    _register_static_patches(register_post_import_hook)

    try:
        from vllm_kunlun.ops.fused_moe.mhc_hyperconnection import apply as _mhc_apply
        _mhc_apply()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("DSV4 MHC hyperconnection installation failed (%s)", exc)

    try:
        from vllm_kunlun.ops.attention.flashmla_bridge import apply as _flashmla_bridge_apply
        _flashmla_bridge_apply()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("DSV4 FlashMLA metadata bridge registration failed (%s)", exc)

    try:
        from vllm_kunlun.ops.attention.compressor import apply as _compressor_apply
        _compressor_apply()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("DSV4 compressor adapter registration failed (%s)", exc)

    try:
        from vllm_kunlun.ops.fused_moe.moe_hash_router import apply as _moe_hash_apply
        _moe_hash_apply()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("DSV4 MoE hash/softplus router registration failed (%s)", exc)

    try:
        from vllm_kunlun.quantization.deepseek_v4 import apply as _moe_int8_apply
        _moe_int8_apply()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("DSV4 INT8-W8A8-MoE bridge registration failed (%s)", exc)

    try:
        from vllm_kunlun.patches.dsv4_cache import register as _cache_register
        labels_installed.extend(_cache_register(register_post_import_hook))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("DSV4 cache patch registration failed (%s)", exc)

    # New patches should be declared statically in _STATIC_PATCHES;
    # _LAZY_HOOKS is reserved for hooks that must not import eagerly.
    for descriptor_label, target, pred_fn, applier_fn in list(_LAZY_HOOKS):

        def _applier(mod, inner_app=applier_fn, marker=descriptor_label):
            try:
                inner_app(mod)
                setattr(mod, _LAZY_DESCRIPTOR, marker)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Lazy DSV4 adapter failed for %r", marker)

        register_post_import_hook(target, lambda mod, p=pred_fn: p(mod), _applier)
        labels_installed.append(f"lazy:{target}")

    return labels_installed


# Platform-level patches (unconditional; apply to every model on Kunlun XPU).
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


def _fp8_kernels_apply(mod: object) -> None:
    import vllm_kunlun.quantization.kernels  # noqa: F401  (registers on import)


def _fp8_moe_method_applied(mod: object) -> bool:
    cls = getattr(mod, "Fp8MoEMethod", None)
    return cls is None or getattr(cls, "_kunlun_fp8_moe", False)


def _fp8_moe_method_apply(mod: object) -> None:
    if not hasattr(mod, "Fp8MoEMethod"):
        return
    layer_mod = importlib.import_module("vllm_kunlun.ops.fused_moe.layer")
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


_PLATFORM_PATCHES = (
    ("platform.fp8_kernels", "vllm.model_executor.kernels.linear",
     "vllm_kunlun.patches.registry", "_fp8_kernels_applied", "_fp8_kernels_apply"),
    ("platform.fp8_moe_method", "vllm.model_executor.layers.quantization.fp8",
     "vllm_kunlun.patches.registry", "_fp8_moe_method_applied", "_fp8_moe_method_apply"),
    ("platform.sparse_mla_warmup", "vllm.model_executor.warmup.kernel_warmup",
     "vllm_kunlun.patches.registry", "_sparse_mla_warmup_applied", "_sparse_mla_warmup_apply"),
    ("platform.worker_kv_bind", "vllm.v1.worker.utils",
     "vllm_kunlun.patches.registry", "_worker_kv_bind_applied", "_worker_kv_bind_apply"),
    ("platform.op_inventory", "vllm.v1.worker.gpu_model_runner",
     "vllm_kunlun.adapter_utils", "_op_inventory_applied", "_op_inventory_apply"),
)


def populate_platform_hooks(register_post_import_hook: Callable[..., None]) -> None:
    """Register every unconditional platform-level patch with the dispatcher."""
    for label, target, module_path, predicate_name, apply_name in _PLATFORM_PATCHES:
        def _applied(mod, _module_path=module_path, _predicate_name=predicate_name):
            if _module_busy(mod):
                return True
            feature = importlib.import_module(_module_path)
            return bool(getattr(feature, _predicate_name)(mod))

        def _apply(mod, _label=label, _module_path=module_path, _apply_name=apply_name):
            feature = importlib.import_module(_module_path)
            getattr(feature, _apply_name)(mod)
            LOGGER.debug("Applied %s to %s", _label, getattr(mod, "__name__", mod))

        register_post_import_hook(target, _applied, _apply)
