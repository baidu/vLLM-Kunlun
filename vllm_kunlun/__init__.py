"""vllm kunlun init"""

import builtins
import importlib
import logging
import os
import sys

OLD_IMPORT_HOOK = builtins.__import__


def _configure_kunlun_logger() -> logging.Logger:
    """Reuse vLLM's handler for the vllm_kunlun logger tree."""
    from vllm.logger import init_logger as init_vllm_logger

    vllm_logger = init_vllm_logger("vllm")
    kunlun_logger = logging.getLogger("vllm_kunlun")

    if not kunlun_logger.handlers:
        for handler in vllm_logger.handlers:
            kunlun_logger.addHandler(handler)

    kunlun_logger.setLevel(vllm_logger.getEffectiveLevel())
    kunlun_logger.propagate = False
    return kunlun_logger


# ---------------------------------------------------------------------------
# Wiring inventory — single source of truth for "what did each hook actually
# bind?" (Question 1 from wangtianyu15, 2026-08-08.)
# Format: op_name -> source (one of):
#   "xspeedgate_ops_module"     getattr(xs, op) — never used in practice, py mod
#                              is just a shim
#   "torch.ops.xspeedgate_ops"  registered to dispatcher + xray wrapper
#   "kunlun_ops_module"         getattr(kunlun_ops, op) — NOT in torch.ops
#   "torch_fallback"            no native — the call falls through to a torch
#                              reference inside the hook
#   "skip"                      intentionally not attempted
# Triggered to log from _op_inventory_final_apply (after gpu_model_runner is
# loaded, so dispatcher hooks have all fired by then).
# ---------------------------------------------------------------------------
_WIRED_INVENTORY: dict[str, str] = {}


def record_wired(op_name: str, source: str) -> None:
    """Record that an op got bound via `source`. Called from any hook.

    Logs immediately so the binding is visible even if a later summary
    hook races with the install hook.
    """
    _WIRED_INVENTORY[op_name] = source
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] wired: %-40s via %s", op_name, source)


def _log_wired_inventory() -> None:
    """Dump one INFO summary + per-op lines."""
    if not _WIRED_INVENTORY:
        return
    logger = logging.getLogger("vllm_kunlun")
    by_src: dict[str, list[str]] = {}
    for op, src in _WIRED_INVENTORY.items():
        by_src.setdefault(src, []).append(op)
    summary = ", ".join(
        f"{src}({len(ops)})" for src, ops in sorted(by_src.items())
    )
    logger.info("[KunlunPlugin] wired inventory: %s", summary)
    for src in sorted(by_src):
        for op in sorted(by_src[src]):
            logger.info("[KunlunPlugin]   %-40s <- %s", op, src)


# Re-entry sentinel for the post-import hooks dispatcher. Some hooks
# trigger their own imports (e.g. importing ``vllm_kunlun.v1.worker.utils``
# to apply the KVBlockZeroer patch), which would re-enter
# ``_custom_import`` recursively. A single dispatcher-level guard is
# sufficient because all hooks are idempotent and we only need one to
# run per real import event.
_POST_IMPORT_DISPATCH_IN_PROGRESS = {"v": False}


_MODULE_MAPPINGS = {
    "vllm.compilation.wrapper": "vllm_kunlun.compilation.wrapper",
    "vllm.model_executor.model_loader.bitsandbytes_loader": "vllm_kunlun.models.model_loader.bitsandbytes_loader",
    "vllm.v1.sample.ops.topk_topp_sampler": "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
    "vllm.v1.sample.ops.logprobs": "vllm_kunlun.v1.sample.ops.logprobs",
    "vllm.v1.sample.rejection_sampler": "vllm_kunlun.v1.sample.rejection_sampler",
    "vllm.attention.ops.merge_attn_states": "vllm_kunlun.ops.attention.merge_attn_states",
    "vllm.v1.worker.mamba_utils": "vllm_kunlun.v1.worker.mamba_utils",
}


# ---------------------------------------------------------------------------
# Post-import hook registry
# ---------------------------------------------------------------------------
# Each entry: (target_module_name, applied_predicate, apply_callable).
#
#   target_module_name  upstream module that must be loaded for this hook
#                       to be applicable. The hook only runs after this
#                       module appears in ``sys.modules``.
#   applied_predicate   ``fn(module) -> bool``. Return True if the patch
#                       has already been applied (cheap, side-effect free).
#                       Used both for idempotency and to short-circuit
#                       once the hook has succeeded.
#   apply_callable      ``fn(module) -> None``. Performs the actual
#                       patch. Must set its own "applied" sentinel so
#                       ``applied_predicate`` returns True afterwards.
#
# To add a new hook: write the apply function (in a dedicated module if
# non-trivial; inline lambda for one-liners), then append a tuple here.
# ---------------------------------------------------------------------------
_POST_IMPORT_HOOKS: list = []


def _register_post_import_hook(target, applied, apply):
    _POST_IMPORT_HOOKS.append((target, applied, apply))


def _dispatch_post_import_hooks():
    """Run every registered post-import hook whose target is loaded.

    Re-entrant safe: importing the kunlun replacement module from within
    a hook re-triggers ``_custom_import`` -> this dispatcher; the
    in-progress sentinel short-circuits the inner call.
    """
    if _POST_IMPORT_DISPATCH_IN_PROGRESS["v"]:
        return
    _POST_IMPORT_DISPATCH_IN_PROGRESS["v"] = True
    try:
        for target, applied, apply in _POST_IMPORT_HOOKS:
            mod = sys.modules.get(target)
            if mod is None:
                continue
            spec = getattr(mod, "__spec__", None)
            if spec is not None and getattr(spec, "_initializing", False):
                continue
            try:
                if applied(mod):
                    continue
                apply(mod)
            except Exception:
                logging.getLogger("vllm_kunlun").exception(
                    "[KunlunPlugin] post-import hook failed for target=%s", target
                )
    finally:
        _POST_IMPORT_DISPATCH_IN_PROGRESS["v"] = False


# --- hook 1: KVBlockZeroer in vllm.v1.worker.utils ------------------------
# Importing the kunlun replacement module triggers an in-place class
# patch (``_kunlun_patched`` flag set on KVBlockZeroer). See
# ``vllm_kunlun/v1/worker/utils.py`` for the actual patch body.
def _kvblockzeroer_applied(mod):
    cls = getattr(mod, "KVBlockZeroer", None)
    return cls is None or getattr(cls, "_kunlun_patched", False)


def _kvblockzeroer_apply(mod):
    if not hasattr(mod, "KVBlockZeroer"):
        return  # upstream module loaded before its class body executed
    import vllm_kunlun.v1.worker.utils  # noqa: F401  (self-applies on import)


_register_post_import_hook(
    "vllm.v1.worker.utils", _kvblockzeroer_applied, _kvblockzeroer_apply
)


# --- hook 2: qwen3_vl HAS_TRITON ------------------------------------------
# Triton kernel ``_bilinear_pos_embed_kernel`` is unsupported on Kunlun XPU.
# Force the module to fall back to native pos-embed interpolation.
def _qwen3vl_applied(mod):
    return not getattr(mod, "HAS_TRITON", False)


def _qwen3vl_apply(mod):
    mod.HAS_TRITON = False
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] qwen3_vl HAS_TRITON forced to False"
    )


_register_post_import_hook(
    "vllm.model_executor.models.qwen3_vl", _qwen3vl_applied, _qwen3vl_apply
)


# --- hook 3: BlockTable.compute_slot_mapping ------------------------------
# Replace the upstream Triton kernel with a torch-native version.
def _block_table_applied(mod):
    cls = getattr(mod, "BlockTable", None)
    return cls is None or getattr(cls, "_kunlun_slot_patched", False)


def _block_table_apply(mod):
    import vllm_kunlun.v1.worker.block_table  # noqa: F401  (self-applies on import)


_register_post_import_hook(
    "vllm.v1.worker.block_table", _block_table_applied, _block_table_apply
)


# --- hook 4: apply_grammar_bitmask in vllm.v1.structured_output.utils -----
# Replace the upstream xgrammar auto backend with torch_native on Kunlun XPU.
def _grammar_bitmask_applied(mod):
    fn = getattr(mod, "apply_grammar_bitmask", None)
    return fn is not None and getattr(fn, "_kunlun_patched", False)


def _grammar_bitmask_apply(mod):
    if not hasattr(mod, "apply_grammar_bitmask"):
        return
    import vllm_kunlun.v1.structured_output.utils  # noqa: F401


_register_post_import_hook(
    "vllm.v1.structured_output.utils", _grammar_bitmask_applied, _grammar_bitmask_apply
)


# --- hook 5: OOT FP8 linear kernel registries ------------------------------
def _fp8_kernels_applied(mod):
    from vllm.platforms import PlatformEnum

    fp8_kernels = getattr(mod, "_POSSIBLE_FP8_KERNELS", None)
    fp8_block_kernels = getattr(mod, "_POSSIBLE_FP8_BLOCK_KERNELS", None)
    return (
        fp8_kernels is not None
        and fp8_block_kernels is not None
        and PlatformEnum.OOT in fp8_kernels
        and PlatformEnum.OOT in fp8_block_kernels
    )


def _fp8_kernels_apply(mod):
    if not hasattr(mod, "_POSSIBLE_FP8_KERNELS"):
        return
    import vllm_kunlun.quantization.kernels  # noqa: F401


_register_post_import_hook(
    "vllm.model_executor.kernels.linear", _fp8_kernels_applied, _fp8_kernels_apply
)


# --- hook 5b: replace upstream Fp8MoEMethod with Kunlun correctness-fallback ---
# Upstream Fp8MoEMethod has no Kunlun-compatible kernel path; we substitute
# KunlunFp8MoEMethod which dequantizes selected FP8 experts to BF16.
def _fp8_moe_method_applied(mod):
    cls = getattr(mod, "Fp8MoEMethod", None)
    return cls is None or getattr(cls, "_kunlun_fp8_moe", False)


def _fp8_moe_method_apply(mod):
    if not hasattr(mod, "Fp8MoEMethod"):
        return
    import importlib
    layer_mod = importlib.import_module("vllm_kunlun.ops.fused_moe.layer")
    KunlunFp8MoEMethod = getattr(layer_mod, "KunlunFp8MoEMethod", None)
    if KunlunFp8MoEMethod is None:
        logging.getLogger("vllm_kunlun").warning(
            "KunlunFp8MoEMethod not available yet (deferred)")
        return
    mod.Fp8MoEMethod = KunlunFp8MoEMethod


_register_post_import_hook(
    "vllm.model_executor.layers.quantization.fp8",
    _fp8_moe_method_applied,
    _fp8_moe_method_apply,
)


# --- hook 6: Worker._maybe_get_memory_pool_context -----------------------
# vllm 0.25.1 _maybe_get_memory_pool_context() gates on is_cuda_alike() /
# is_xpu(). KunlunPlatform is OOT so neither returns True, causing it to
# fall through to get_mem_allocator_instance() which raises RuntimeError.
# Patch the method to return nullcontext() for Kunlun.
def _memory_pool_applied(mod):
    cls = getattr(mod, "Worker", None)
    return cls is None or getattr(cls, "_kunlun_memory_pool_patched", False)


def _memory_pool_apply(mod):
    from contextlib import nullcontext as _nullcontext

    _orig = mod.Worker._maybe_get_memory_pool_context

    def _patched(self, tag: str):
        from vllm.platforms import current_platform

        if type(current_platform).__name__ == "KunlunPlatform":
            return _nullcontext()
        return _orig(self, tag)

    mod.Worker._maybe_get_memory_pool_context = _patched
    mod.Worker._kunlun_memory_pool_patched = True
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched Worker._maybe_get_memory_pool_context"
    )


_register_post_import_hook(
    "vllm.v1.worker.gpu_worker", _memory_pool_applied, _memory_pool_apply
)


# --- hook 6: skip qwen_triton_warmup on Kunlun XPU ---
def _qwen_triton_warmup_applied(mod):
    fn = getattr(mod, "qwen_triton_warmup", None)
    return fn is not None and getattr(fn, "_kunlun_patched", False)


def _qwen_triton_warmup_apply(mod):
    def _noop(*args, **kwargs):
        import logging

        logging.getLogger("vllm_kunlun").info(
            "[KunlunPlugin] Skipping qwen_triton_warmup"
        )

    _noop._kunlun_patched = True
    mod.qwen_triton_warmup = _noop
    import logging

    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched kernel_warmup.qwen_triton_warmup -> no-op"
    )


_register_post_import_hook(
    "vllm.model_executor.warmup.kernel_warmup",
    _qwen_triton_warmup_applied,
    _qwen_triton_warmup_apply,
)


# The generic warmup invokes Intel/CUDA Triton sparse MLA kernels. Kunlun's
# actual sparse attention path is patched separately and must remain enabled.
_SPARSE_MLA_WARMUPS = (
    "sparse_mla_triton_warmup_if_needed",
    "flashinfer_sparse_mla_decode_autotune_warmup",
    "deepseek_v4_sparse_mla_attention_warmup",
)


def _sparse_mla_warmup_applied(mod):
    return all(
        getattr(getattr(mod, name, None), "_kunlun_patched", False)
        for name in _SPARSE_MLA_WARMUPS
    )


def _sparse_mla_warmup_apply(mod):
    def _noop(*args, **kwargs):
        import logging

        logging.getLogger("vllm_kunlun").info(
            "[KunlunPlugin] Skipping non-Kunlun sparse MLA warmup"
        )

    _noop._kunlun_patched = True
    for name in _SPARSE_MLA_WARMUPS:
        setattr(mod, name, _noop)


_register_post_import_hook(
    "vllm.model_executor.warmup.kernel_warmup",
    _sparse_mla_warmup_applied,
    _sparse_mla_warmup_apply,
)


def _v4_attention_alias_applied(mod):
    return getattr(mod, "_kunlun_v4_kv_insert_patched", False)


def _v4_attention_alias_apply(mod):
    """Bind Kunlun-native KV-insert paths onto ``torch.ops._C``.

    Algorithmic bodies live in ``vllm_kunlun.ops.attention.dsv4_kv_insert_paths``;
    this root-level wrapper just performs symbol registration against
    community-side dispatcher namespaces plus sets the idempotency
    sentinel consumed by sibling hooks checking whether this target was
    already wired up during an earlier post-import dispatch round.
    """
    import torch
    from vllm_kunlun.ops.attention.dsv4_kv_insert_paths import (
        fp8_quant_insert,
        bf16_full_cache_insert,
    )
    setattr(
        torch.ops._C,
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        fp8_quant_insert,
    )
    setattr(
        torch.ops._C,
        "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert",
        bf16_full_cache_insert,
    )
    mod._kunlun_v4_kv_insert_patched = True


_register_post_import_hook(
    "vllm.models.deepseek_v4.attention",
    _v4_attention_alias_applied,
    _v4_attention_alias_apply,
)
_register_post_import_hook(
    "vllm.models.deepseek_v4.nvidia.model",
    _v4_attention_alias_applied,
    _v4_attention_alias_apply,
)


# === DSv4 FULL cudagraph: raise attention-backend cudagraph support to ALWAYS ===
#
# WHY. `init_attn_backend()` (vllm/v1/worker/gpu/attn_utils.py) takes the MINIMUM
# `get_cudagraph_support()` over every metadata builder in the forward pass, and
# `CompilationConfig.resolve_cudagraph_mode_and_sizes()` demotes
# `cudagraph_mode=FULL` -> `FULL_DECODE_ONLY` whenever that minimum is not ALWAYS.
# All four builders DSv4 uses advertise UNIFORM_BATCH, so FULL is silently demoted.
#
# WHAT. Behind `KUNLUN_DSV4_FORCE_FULL_CG=1`, raise those four builders to ALWAYS.
# `_cudagraph_support` is enough for three of them (the base
# `get_cudagraph_support` just returns it); the indexer builder overrides the
# method itself (indexer.py:244), so that one needs the method replaced too.
#
# SCOPE. Off by default. When on, only the four DSv4 builder classes below are
# touched -- no other model's backend changes, and no vllm source file is edited.

_V4_FULL_CG_BUILDERS: tuple[tuple[str, str, bool], ...] = (
    # (module, class, also_override_get_cudagraph_support_method)
    ("vllm.v1.attention.backends.mla.indexer", "DeepseekV32IndexerMetadataBuilder", True),
    ("vllm.models.deepseek_v4.sparse_mla", "DeepseekV4FlashMLAMetadataBuilder", False),
    ("vllm.v1.attention.backends.mla.sparse_swa", "DeepseekSparseSWAMetadataBuilder", False),
    (
        "vllm_kunlun.v1.attention.backends.mla.flashmla_sparse",
        "FlashMLASparseMetadataBuilder",
        False,
    ),
)


def _v4_full_cg_enabled() -> bool:
    return os.environ.get("KUNLUN_DSV4_FORCE_FULL_CG", "0") in ("1", "true", "True", "yes")


def _v4_full_cg_applied(mod):
    # Always re-run: the hook fires on several modules and each call patches only
    # the classes that are already importable, so later calls pick up the rest.
    return False


def _v4_full_cg_apply(mod):
    if not _v4_full_cg_enabled():
        return
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
        logging.getLogger("vllm_kunlun").info(
            "[KunlunPlugin] KUNLUN_DSV4_FORCE_FULL_CG=1: cudagraph_support -> ALWAYS for %s",
            ", ".join(patched),
        )


for _full_cg_target in (
    "vllm.models.deepseek_v4.sparse_mla",
    "vllm.v1.attention.backends.mla.sparse_swa",
    "vllm.v1.worker.gpu.attn_utils",
):
    _register_post_import_hook(
        _full_cg_target,
        _v4_full_cg_applied,
        _v4_full_cg_apply,
    )


def _v4_o_proj_applied(mod):
    # The installed name is `deep_gemm_fp8_o_proj` (that is what the community
    # modules import), but the replacement comes from
    # vllm_kunlun.ops.deepseek_v4_o_proj -- check against that module, otherwise
    # the predicate never matches and the hook re-fires on every import
    # statement executed in the process.
    fn = getattr(mod, "deep_gemm_fp8_o_proj", None)
    return fn is not None and getattr(fn, "__module__", "") == (
        "vllm_kunlun.ops.deepseek_v4_o_proj"
    )


def _v4_o_proj_apply(mod):
    from vllm_kunlun.ops.deepseek_v4_o_proj import deepseek_v4_bf16_o_proj

    mod.deep_gemm_fp8_o_proj = deepseek_v4_bf16_o_proj
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched V4 output projection"
    )


for _v4_o_proj_module in (
    "vllm.models.deepseek_v4.nvidia.ops.o_proj",
    "vllm.models.deepseek_v4.nvidia.flashmla",
    "vllm.models.deepseek_v4.nvidia.flashinfer_sparse",
):
    _register_post_import_hook(
        _v4_o_proj_module,
        _v4_o_proj_applied,
        _v4_o_proj_apply,
    )


def _hash_topk_applied(mod):
    return getattr(
        getattr(mod, "topk_hash_softplus_sqrt", None),
        "_kunlun_hash_fallback",
        False,
    )


def _preload_mapped(full_name):
    """Load the kunlun replacement for ``full_name`` into sys.modules."""
    if full_name in sys.modules:
        return
    target_module = _MODULE_MAPPINGS[full_name]
    module = importlib.import_module(target_module)
    sys.modules[full_name] = module
    sys.modules[target_module] = module


def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    try:
        if level == 0:
            # Case 1: `from vllm.x.y import Z` / `import vllm.x.y`
            # Here module_name is the full dotted path of the mapped module.
            if module_name in _MODULE_MAPPINGS:
                _preload_mapped(module_name)

            # Case 2: `from vllm.x import y` where y itself is a mapped submodule.
            # CPython calls __import__("vllm.x", fromlist=("y",)); module_name
            # does not include "y", so we must check each fromlist entry.
            if fromlist:
                for name in fromlist:
                    full = f"{module_name}.{name}"
                    if full in _MODULE_MAPPINGS:
                        _preload_mapped(full)
    except Exception:
        logging.getLogger("vllm_kunlun").debug(
            "[KunlunPlugin] _custom_import preload skipped for %s", module_name,
            exc_info=True,
        )

    result = OLD_IMPORT_HOOK(
        module_name, globals=globals, locals=locals, fromlist=fromlist, level=level
    )

    # Run all registered post-import hooks. Each hook checks its own
    # target module presence and idempotency flag; the dispatcher itself
    # has a re-entry guard so hook-triggered imports do not recurse.
    _dispatch_post_import_hooks()

    return result


def import_hook():
    """Apply import hook for VLLM Kunlun"""
    builtins.__import__ = _custom_import


def _worker_kv_bind_applied(mod):
    fn = getattr(mod, "bind_kv_cache", None)
    return fn is not None and getattr(fn, "__module__", "") == __name__


def _worker_kv_bind_apply(mod):
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
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched OOT multi-cache binding"
    )


_register_post_import_hook(
    "vllm.v1.worker.utils",
    _worker_kv_bind_applied,
    _worker_kv_bind_apply,
)


# ---------------------------------------------------------------------------
# V4 compressor ops: save_partial_states + compress_norm_rope_store
# These Triton kernels are not available on Kunlun; replace with PyTorch.
# ---------------------------------------------------------------------------

def _masked_paged_write(cache, dest, write_ok, data, col_start=0):
    """Scatter ``data`` into paged ``cache`` at flat slots ``dest``, skipping
    rows where ``write_ok`` is False -- without any host-side read.

    ``cache`` is [num_blocks, block_size, W]; ``dest`` [T] holds flat slot ids
    (block * block_size + offset); ``data`` is [T, D] written to columns
    ``col_start : col_start + D``.

    A masked scatter normally wants a compacted index list, but compaction
    (``nonzero`` / boolean masking) has a data-dependent output shape and needs
    a host-side count, which is unavailable during CUDA-graph capture. So every
    row is written unconditionally and the skipped rows are redirected onto
    block 0 with a zero payload: vLLM's BlockPool reserves block 0 as the
    ``null_block``, never hands it to a request, and keeps it all-zero, so
    those stores are no-ops and can never collide with a real destination.

    The mask is applied in fp32 and only then cast to the cache dtype, because
    xdnn's capture-mode kernels (``aten_capture/eager_customized``) do not
    implement ``where`` or ``index_select`` for uint8 -- and the DSv4 indexer's
    compressed cache is uint8. ``index_put_`` and dtype casts are implemented.
    """
    import torch

    D = data.shape[1]
    block_size = cache.shape[1]
    data = data.float()
    dest_f = torch.where(write_ok, dest, torch.zeros_like(dest))
    data_f = torch.where(
        write_ok.unsqueeze(-1), data, torch.zeros_like(data)
    ).to(cache.dtype)
    cache[:, :, col_start:col_start + D].index_put_(
        (dest_f // block_size, dest_f % block_size), data_f, accumulate=False
    )

def register():
    """Register the Kunlun platform"""

    logger = _configure_kunlun_logger()
    logger.info("[KunlunPlugin] register() pid=%s", os.getpid())

    # --- block vllm's NVIDIA prebuilt _C / _moe_C from being loaded ---
    # These are imported (via top-level ``import vllm._C`` in
    # ``vllm.platforms.cuda`` / inside ``Platform.import_kernels``) by
    # multiple vllm code paths. On Kunlun XPU they are useless and would
    # pre-register CUDA kernels that clash with the Kunlun
    # ``@custom_op`` / ``@impl(..., "CUDA")`` registrations on
    # PyTorch 2.9+. Stub them out NOW, before any other vllm import
    # has a chance to load them.
    import types as _types

    for _stub in ("vllm._C", "vllm._moe_C"):
        if _stub not in sys.modules:
            sys.modules[_stub] = _types.ModuleType(_stub)

    # --- eagerly register Kunlun custom ops ---
    # We load ``vllm_kunlun/ops/_custom_ops.py`` DIRECTLY via
    # ``spec_from_file_location`` under a private module name, instead of
    # ``import vllm_kunlun.ops`` which would trigger
    # ``vllm_kunlun/ops/__init__.py`` and transitively import
    # ``vllm_kunlun.ops.fused_moe.layer`` →
    # ``vllm.model_executor.layers.fused_moe.config`` →
    # ``vllm.model_executor.layers.quantization.utils.quant_utils`` →
    # ``vllm._custom_ops``. The last step calls
    # ``current_platform.import_kernels()`` while the platform plugin is
    # still mid-registration, which is fragile and was observed to leave
    # the worker process without any custom ops registered.
    #
    # Loading just the bare file registers all 54 Kunlun ops to
    # ``torch.ops._C`` / ``torch.ops._moe_C`` and avoids touching any
    # other vllm internals.
    try:
        import importlib.util as _ilu
        import os as _os

        _ops_file = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            "ops",
            "_custom_ops.py",
        )
        _private = "_vllm_kunlun_custom_ops_registration"
        if "vllm_kunlun.ops._custom_ops" not in sys.modules and _private not in sys.modules:
            _spec = _ilu.spec_from_file_location(_private, _ops_file)
            _mod = _ilu.module_from_spec(_spec)
            sys.modules[_private] = _mod
            _spec.loader.exec_module(_mod)
        logger.info("[KunlunPlugin] vllm_kunlun custom ops registered")
    except Exception:
        logger.exception("[KunlunPlugin] custom ops registration failed")
        raise

    # --- load native extension to register torch.ops._C.weak_ref_tensor ---
    try:
        from . import _kunlun  # noqa: F401

        logger.info("[KunlunPlugin] _kunlun native extension loaded")
    except ImportError as e:
        logger.warning("[KunlunPlugin] Failed to load _kunlun: %s", e)

    # --- import wrapper & patch utils ---
    try:
        from .schema import direct_register_custom_op  # noqa: F401
        from .schema import patch_annotations_for_schema  # noqa: F401

        logger.info("[KunlunPlugin] vllm_utils_wrapper loaded and patched")
    except Exception:
        logger.exception("[KunlunPlugin] wrapper import/patch failed")
        raise

    # --- import hook ---
    try:
        import_hook()
        _dispatch_post_import_hooks()
        logger.info("[KunlunPlugin] import_hook() ok")
    except Exception:
        logger.exception("[KunlunPlugin] import_hook() failed")
        raise

    # --- patch torch.accelerator.get_memory_info for Kunlun XPU ---
    # vllm 0.25.1 uses torch.accelerator.get_memory_info() which does not exist
    # in torch_xmlir 2.9. Patch it to use torch.cuda.mem_get_info which works on XPU.
    try:
        import torch as _torch

        def _kunlun_get_memory_info(device=None):
            if device is None:
                idx = _torch.cuda.current_device()
            elif isinstance(device, _torch.device):
                idx = (
                    device.index
                    if device.index is not None
                    else _torch.cuda.current_device()
                )
            elif isinstance(device, int):
                idx = device
            else:
                idx = _torch.cuda.current_device()
            return _torch.cuda.mem_get_info(idx)

        _torch.accelerator.get_memory_info = _kunlun_get_memory_info
        logger.info("[KunlunPlugin] patched torch.accelerator.get_memory_info")
    except Exception:
        logger.exception(
            "[KunlunPlugin] failed to patch torch.accelerator.get_memory_info"
        )
        raise

    # --- register reasoning parser override (lazy, to avoid circular import) ---
    try:
        from vllm.reasoning import ReasoningParserManager

        # Override the lazy registration path with our custom parser.
        # This happens before vllm's default lazy registration (which is
        # triggered when vllm.reasoning module is imported), so our path
        # takes precedence.
        # Custom parser for Qwen3.5 support
        ReasoningParserManager.register_lazy_module(
            name="qwen3",
            module_path="vllm_kunlun.reasoning.qwen3_reasoning_parser",
            class_name="Qwen3ReasoningParser",
        )
        logger.info("[KunlunPlugin] registered Qwen3ReasoningParser override (lazy)")
    except Exception:
        logger.exception("[KunlunPlugin] Qwen3ReasoningParser registration failed")
        # Non-fatal: continue without the override

    _log_op_inventory(logger)
    logger.info("[KunlunPlugin] register() done")
    return "vllm_kunlun.platforms.kunlun.KunlunPlatform"


def register_model():
    """Register models for training and inference"""
    from .models import register_model as _reg

    _reg()


def register_reasoning_parser():
    """Register reasoning parsers for inference."""
    from .reasoning import register_reasoning_parser as _reg_reasoning_parser

    _reg_reasoning_parser()


def register_tool_parser():
    """Register tool parsers for inference."""
    from .entrypoints.openai.tool_parsers import (
        register_tool_parser as _reg_tool_parser,
    )

    _reg_tool_parser()


def _log_op_inventory(logger, tag="early"):
    """Log the in-process op inventory.

    Called twice on purpose. ~20 xspeedgate ops register LAZILY, only once
    the quant modules are imported, so the register()-time snapshot ("early")
    under-reports and e.g. shows sparse_attn_fwd absent when it is in fact
    callable later. Trust the "final" line.

    A stale source tree on PYTHONPATH can also shadow the installed package
    while its .so still gets dlopened, so print which .so files are mapped.
    """
    try:
        import torch

        def names(ns):
            return {n.split("::", 1)[1]
                    for n in torch._C._dispatch_get_all_op_names()
                    if n.startswith(ns + "::")}

        xsg, kl = names("xspeedgate_ops"), names("_C")
        try:
            import xspeedgate_ops
            where = xspeedgate_ops.__file__
        except Exception:  # noqa: BLE001
            where = "(not importable)"
        logger.info("[KunlunPlugin] op inventory (%s): xspeedgate_ops=%d _C=%d from %s",
                    tag, len(xsg), len(kl), where)
        watch = ("sparse_attn_fwd", "act_sqrt_softplus", "dequantize_fp8_blocks",
                 "moe_pre_small", "compressed_attention", "mqa_logits_paged",
                 "moe_hash_topk_fused", "topk_per_row")
        logger.info("[KunlunPlugin] key ops (%s): %s", tag,
                    " ".join("%s=%s" % (w, "Y" if w in xsg else "n") for w in watch))
        with open("/proc/self/maps") as f:
            libs = sorted({ln.split()[-1] for ln in f
                           if "xspeedgate" in ln or "kunlun_ops" in ln})
        for lib in libs:
            logger.info("[KunlunPlugin] mapped %s", lib)
    except Exception as e:  # noqa: BLE001
        logger.warning("[KunlunPlugin] op inventory probe failed: %r", e)


def _op_inventory_final_applied(mod):
    return getattr(mod, "_kunlun_op_inventory_logged", False)


def _op_inventory_final_apply(mod):
    mod._kunlun_op_inventory_logged = True
    _log_op_inventory(logging.getLogger("vllm_kunlun"), tag="final")
    _log_wired_inventory()


# gpu_model_runner is imported during engine init, i.e. after the quantization
# modules that trigger the lazy op registration.
_register_post_import_hook(
    "vllm.v1.worker.gpu_model_runner",
    _op_inventory_final_applied,
    _op_inventory_final_apply,
)


# ---- DeepSeek-V4 OOT adapters -----------------------------------------------
# The per-feature monkey-patches for DeepSeek-V4-on-Kunlum live in the adapter
# package below rather than being hard-coded here. ``apply_all`` is the same
# explicit adapter facade used by the other model-specific integrations.
try:
    from .patches.deepseek_v4 import apply_all as _apply_v4_adapters

    _apply_v4_adapters(_register_post_import_hook)
except Exception:
    logging.getLogger("vllm_kunlun").exception(
        "[KunlunPlugin] DSV4 adapter pack failed to load"
    )

