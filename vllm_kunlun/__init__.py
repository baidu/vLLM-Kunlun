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


def _deepseek_v4_cache_applied(mod):
    fn = getattr(mod, "dequantize_and_gather_k_cache", None)
    return getattr(fn, "_kunlun_patched_v2", False)


def _deepseek_v4_cache_apply(mod):
    # [kunlun-hook v2] dsv4 dequant fallback
    #
    # Kunlun's Triton cannot bitcast float8 (data-type of size 8 <-> 16),
    # so the upstream triton dequant-gather kernel raises a CompilationError
    # at first call. We replace the public dispatcher in every caller-side
    # namespace with a PyTorch fallback that handles both plain bf16 and
    # packed UE8M0 fp8 cache pages. This also short-circuits the cutedsl
    # branch (which pulls in CUDA CuTe DSL).
    from vllm_kunlun.ops.deepseek_v4_cache import (
        dequantize_and_gather_k_cache_pytorch,
    )

    def _kunlun_dequant_and_gather(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset,
        use_fnuz: bool = False,
    ):
        return dequantize_and_gather_k_cache_pytorch(
            out, k_cache, seq_lens, gather_lens, block_table, block_size, offset,
            use_fnuz=use_fnuz,
        )

    _kunlun_dequant_and_gather._kunlun_patched_v2 = True
    mod.dequantize_and_gather_k_cache = _kunlun_dequant_and_gather
    # Also override the triton entry so any lingering direct callers
    # (or a re-import via cache_utils.dequantize_and_gather_k_cache_triton)
    # route through the PyTorch fallback.
    if hasattr(mod, "dequantize_and_gather_k_cache_triton"):
        mod.dequantize_and_gather_k_cache_triton = _kunlun_dequant_and_gather

    # Kunlun Triton also cannot execute _compute_global_topk_indices_and_lens
    # and _combine_topk_swa_indices (CUDA_ERROR_NOT_SUPPORTED). Replace the
    # public helpers in this namespace with PyTorch equivalents.
    from vllm_kunlun.ops.deepseek_v4_sparse_index import (
        compute_global_topk_indices_and_lens_pytorch as _kunlun_compute_global,
        combine_topk_swa_indices_pytorch as _kunlun_combine_topk_swa,
    )
    if hasattr(mod, "compute_global_topk_indices_and_lens"):
        mod.compute_global_topk_indices_and_lens = _kunlun_compute_global
    if hasattr(mod, "combine_topk_swa_indices"):
        mod.combine_topk_swa_indices = _kunlun_combine_topk_swa


# The dequant name is bound into three namespaces at import time:
#   - vllm.models.deepseek_v4.common.ops.cache_utils   (definition)
#   - vllm.models.deepseek_v4.common.ops               (package re-export)
#   - vllm.models.deepseek_v4.nvidia.flashmla          (from-import)
# Python from-import creates a fresh local binding, so we must patch each
# module's namespace independently after it is imported.
for _dsv4_cache_mod in (
    "vllm.models.deepseek_v4.common.ops.cache_utils",
    "vllm.models.deepseek_v4.common.ops",
    "vllm.models.deepseek_v4.nvidia.flashmla",
):
    _register_post_import_hook(
        _dsv4_cache_mod,
        _deepseek_v4_cache_applied,
        _deepseek_v4_cache_apply,
    )


def _dsv4_layout_applied(mod):
    return getattr(mod, "_kunlun_dsv4_layout_patched", False)


def _dsv4_layout_apply(mod):
    # [kunlun-hook] dsv4 use_fp8_ds_mla_layout False
    #
    # Upstream default is ``use_fp8_ds_mla_layout: ClassVar[bool] = True``
    # on DeepseekV4Attention, which asserts fp8 kv-cache. Kunlun ships bf16
    # kv-cache; flip the ClassVar on every DSv4 attention subclass in the
    # module so backend dispatch selects the bf16 plain-row page format.
    import inspect

    for _name, _cls in list(mod.__dict__.items()):
        if not inspect.isclass(_cls):
            continue
        if _cls.__module__ != mod.__name__:
            continue
        if "use_fp8_ds_mla_layout" in vars(_cls) or hasattr(_cls, "use_fp8_ds_mla_layout"):
            try:
                _cls.use_fp8_ds_mla_layout = False
            except Exception:
                pass
    mod._kunlun_dsv4_layout_patched = True


_register_post_import_hook(
    "vllm.models.deepseek_v4.attention",
    _dsv4_layout_applied,
    _dsv4_layout_apply,
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




def _flashmla_metadata_applied(mod):
    fn = getattr(mod, "get_mla_metadata", None)
    return fn is not None and getattr(fn, "_kunlun_patched", False)


def _flashmla_metadata_apply(mod):
    import torch
    from vllm_kunlun.ops.attention.flashmla import get_mla_metadata as _kunlun_get

    def get_mla_metadata(cache_seqlens=None, num_heads_per_head_k=1, num_heads_k=1):
        if cache_seqlens is None:
            empty = torch.empty(0, dtype=torch.int32)
            return empty, empty
        return _kunlun_get(cache_seqlens, num_heads_per_head_k, num_heads_k)

    get_mla_metadata._kunlun_patched = True
    mod.get_mla_metadata = get_mla_metadata

    # Also patch flash_mla_with_kvcache and flash_mla_sparse_fwd with Kunlun implementations
    from vllm_kunlun.ops.attention.flashmla import (
        flash_mla_with_kvcache as _kunlun_flash_mla_with_kvcache,
        flash_mla_sparse_prefill as _kunlun_flash_mla_sparse_fwd,
    )
    mod.flash_mla_with_kvcache = _kunlun_flash_mla_with_kvcache
    mod.flash_mla_sparse_fwd = _kunlun_flash_mla_sparse_fwd


for _flashmla_metadata_module in (
    "vllm.v1.attention.ops.flashmla",
    "vllm.v1.attention.backends.mla.flashmla_sparse",
    "vllm.v1.attention.backends.mla.sparse_swa",
    "vllm.models.deepseek_v4.nvidia.flashmla",
):
    _register_post_import_hook(
        _flashmla_metadata_module,
        _flashmla_metadata_applied,
        _flashmla_metadata_apply,
    )


def _v4_attention_alias_applied(mod):
    return getattr(mod, "_kunlun_v4_kv_insert_patched", False)


def _v4_attention_alias_apply(mod):
    import torch
    import kunlun_ops

    def _insert(q, kv, cache, slot_mapping, positions, cos_sin, padded_heads, eps, block_size):
        # Community vLLM call convention:
        #   q: [N, H, 512], kv: [N, 512], cache: [num_blocks, block_stride],
        #   slot_mapping: [M], positions: [N], cos_sin: [max_pos, 64]
        #   padded_heads: int, eps: float, block_size: int
        # kunlun_ops convention:
        #   (kv, cos_sin_cache, position_ids, slot_mapping, q, k_cache, cache_block_size, eps, scale)
        return kunlun_ops.fused_deepseek_v4_qnorm_rope_kv_insert(
            kv,
            cos_sin,
            positions,
            slot_mapping,
            q,
            cache,
            block_size,
            eps,
            None,
        )

    _bf16_insert_warned = [False]

    def _bf16_insert(q, kv, swa_kv_cache_3d, slot_mapping, positions, cos_sin, eps, block_size):
        # BF16 full-cache path: fused Q RMSNorm(no-weight) + GPT-J RoPE + KV
        # RoPE + paged bf16 cache write — same native kernel as the FP8 _insert
        # path (fused_deepseek_v4_qnorm_rope_kv_insert) with bf16 k_cache dtype.
        # Falls back to torch on failure (log-once).
        try:
            cache_2d = swa_kv_cache_3d.view(swa_kv_cache_3d.shape[0], -1)
            kunlun_ops.fused_deepseek_v4_qnorm_rope_kv_insert(
                kv, cos_sin, positions.long(), slot_mapping.long(),
                q, cache_2d, block_size, eps, None,
            )
            return q
        except Exception as e:  # noqa: BLE001
            if not _bf16_insert_warned[0]:
                import logging as _l
                _l.getLogger("vllm_kunlun").warning(
                    "native bf16 fused_deepseek_v4_qnorm_rope_kv_insert "
                    "failed (%s); using torch fallback", e)
                _bf16_insert_warned[0] = True

        # Torch fallback: Q RMSNorm (no weight) + GPT-J RoPE + KV RoPE + cache write.
        num_tokens = q.shape[0]
        rope_dim = 64
        q_float = q.float()
        rms = torch.rsqrt(q_float.square().mean(dim=-1, keepdim=True) + eps)
        q.copy_((q_float * rms).to(q.dtype))

        cos_sin_selected = cos_sin[positions]
        cos_val = cos_sin_selected[:, :32]
        sin_val = cos_sin_selected[:, 32:]

        def apply_gptj_rope(x_rope):
            x1 = x_rope[..., ::2]
            x2 = x_rope[..., 1::2]
            if x_rope.ndim == 3:
                cos_b = cos_val.unsqueeze(1)
                sin_b = sin_val.unsqueeze(1)
            else:
                cos_b = cos_val
                sin_b = sin_val
            out1 = x1.float() * cos_b.float() - x2.float() * sin_b.float()
            out2 = x1.float() * sin_b.float() + x2.float() * cos_b.float()
            return torch.stack([out1, out2], dim=-1).flatten(-2).to(x_rope.dtype)

        q[..., -rope_dim:] = apply_gptj_rope(q[..., -rope_dim:])
        kv_roped = kv.clone()
        kv_roped[..., -rope_dim:] = apply_gptj_rope(kv[..., -rope_dim:])

        slots = slot_mapping.to(torch.long)
        valid = slots >= 0
        if bool(valid.any()):
            v_slots = slots[valid]
            swa_kv_cache_3d[v_slots // block_size, v_slots % block_size, :] = (
                kv_roped[valid].to(swa_kv_cache_3d.dtype)
            )
        return q

    setattr(
        torch.ops._C,
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        _insert,
    )
    setattr(
        torch.ops._C,
        "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert",
        _bf16_insert,
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


# --- hook: DeepSeek V4 mHC TileLang replacement --------------------------
def _mhc_tilelang_applied(mod):
    return getattr(mod, "_kunlun_mhc_patched", False)


def _mhc_tilelang_apply(mod):
    from vllm_kunlun.ops.hyper_connection import (
        mhc_fused_post_pre_tilelang,
        mhc_post_tilelang,
        mhc_pre_tilelang,
    )

    mod.mhc_pre_tilelang = mhc_pre_tilelang
    mod.mhc_post_tilelang = mhc_post_tilelang
    mod.mhc_fused_post_pre_tilelang = mhc_fused_post_pre_tilelang
    mod._kunlun_mhc_patched = True
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched V4 mHC TileLang helpers"
    )


_register_post_import_hook(
    "vllm.model_executor.kernels.mhc.tilelang",
    _mhc_tilelang_applied,
    _mhc_tilelang_apply,
)


def _v4_model_mhc_applied(mod):
    fn = getattr(mod, "mhc_pre_tilelang", None)
    mhc_applied = fn is not None and getattr(fn, "__module__", "") == (
        "vllm_kunlun.ops.hyper_connection"
    )
    patched_classes = (
        getattr(mod, "DeepseekV4DecoderLayer", None),
        getattr(mod, "DeepseekV4ForCausalLM", None),
    )
    instances_patched = all(
        cls is not None
        and getattr(cls.__init__, "_kunlun_rmsnorm_patched", False)
        for cls in patched_classes
    )
    moe_cls = getattr(mod, "DeepseekV4MoE", None)
    moe_forward = getattr(moe_cls, "forward", None)
    moe_uses_fused_apply = "apply_weights" in getattr(
        getattr(moe_forward, "__code__", None), "co_names", ()
    )
    moe_patched = not moe_uses_fused_apply or getattr(
        moe_forward, "_kunlun_mhc_router_patched", False
    )
    head_fn = getattr(mod, "hc_head_fused_kernel_tilelang", None)
    head_patched = head_fn is not None and getattr(head_fn, "__module__", "") == (
        "vllm_kunlun.ops.hyper_connection"
    )
    return (
        mhc_applied
        and head_patched
        and instances_patched
        and moe_patched
        and getattr(mod, "_kunlun_v4_kv_insert_patched", False)
    )


def _v4_model_mhc_apply(mod):
    _v4_attention_alias_apply(mod)
    from vllm_kunlun.ops.hyper_connection import (
        hc_head_fused_kernel_kunlun,
        mhc_fused_post_pre_tilelang,
        mhc_post_tilelang,
        mhc_pre_tilelang,
    )

    mod.mhc_pre_tilelang = mhc_pre_tilelang
    mod.mhc_post_tilelang = mhc_post_tilelang
    mod.mhc_fused_post_pre_tilelang = mhc_fused_post_pre_tilelang
    mod.hc_head_fused_kernel_tilelang = hc_head_fused_kernel_kunlun

    if hasattr(mod, "RMSNorm"):
        from vllm_kunlun.ops.layernorm import (
            KunlunRMSNorm,
            fused_add_rms_norm_kunlun,
            rms_norm_kunlun,
        )

        native_fn = mod.RMSNorm.forward_native
        native_globals = getattr(native_fn, "__globals__", None)
        if native_globals is not None:
            native_globals["rms_norm"] = rms_norm_kunlun
            native_globals["fused_add_rms_norm"] = fused_add_rms_norm_kunlun

        from functools import wraps

        def _patch_rmsnorm_instances(root):
            for submodule in root.modules():
                if submodule.__class__.__name__ == "RMSNorm":
                    submodule._forward_method = KunlunRMSNorm.forward_oot.__get__(
                        submodule, submodule.__class__
                    )

        for class_name in (
            "DeepseekV4DecoderLayer",
            "DeepseekV4ForCausalLM",
        ):
            model_cls = getattr(mod, class_name, None)
            if model_cls is None or getattr(
                model_cls.__init__, "_kunlun_rmsnorm_patched", False
            ):
                continue

            original_init = model_cls.__init__

            @wraps(original_init)
            def _kunlun_model_init(
                self,
                *args,
                __original_init=original_init,
                **kwargs,
            ):
                __original_init(self, *args, **kwargs)
                _patch_rmsnorm_instances(self)

            _kunlun_model_init._kunlun_rmsnorm_patched = True
            model_cls.__init__ = _kunlun_model_init

    moe_cls = getattr(mod, "DeepseekV4MoE", None)
    moe_forward = getattr(moe_cls, "forward", None)
    moe_uses_fused_apply = "apply_weights" in getattr(
        getattr(moe_forward, "__code__", None), "co_names", ()
    )
    if moe_uses_fused_apply and not getattr(
        moe_forward, "_kunlun_mhc_router_patched", False
    ):

        def _kunlun_v4_moe_forward(self, hidden_states):
            router_input = (
                hidden_states[0]
                if isinstance(hidden_states, (list, tuple))
                else hidden_states
            )
            router_logits = self.gate(router_input)
            return self.moe.apply_weights(
                x=router_input,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=True,
                num_expert_group=self.n_group,
                topk_group=self.topk_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                enable_eplb=False,
            )

        _kunlun_v4_moe_forward._kunlun_mhc_router_patched = True
        moe_cls.forward = _kunlun_v4_moe_forward

    if all(
        getattr(mod, name, None) is not None
        for name in ("DeepseekV4DecoderLayer", "DeepseekV4ForCausalLM")
    ):
        logging.getLogger("vllm_kunlun").info(
            "[KunlunPlugin] patched DeepSeek V4 mHC and RMSNorm bindings"
        )


_register_post_import_hook(
    "vllm.models.deepseek_v4.nvidia.model",
    _v4_model_mhc_applied,
    _v4_model_mhc_apply,
)


def _v4_attention_rmsnorm_applied(mod):
    fn = getattr(mod, "fused_q_kv_rmsnorm", None)
    return fn is not None and getattr(fn, "__module__", "") == (
        "vllm_kunlun.ops.layernorm"
    )


def _v4_attention_rmsnorm_apply(mod):
    from vllm_kunlun.ops.layernorm import fused_q_kv_rmsnorm_kunlun

    mod.fused_q_kv_rmsnorm = fused_q_kv_rmsnorm_kunlun
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched V4 fused Q/KV RMSNorm"
    )


_register_post_import_hook(
    "vllm.models.deepseek_v4.attention",
    _v4_attention_rmsnorm_applied,
    _v4_attention_rmsnorm_apply,
)


def _v4_indexer_q_applied(mod):
    fn = getattr(mod, "fused_indexer_q_rope_quant", None)
    return fn is not None and getattr(fn, "__module__", "") == (
        "vllm_kunlun.ops.fp8"
    )


def _v4_indexer_q_apply(mod):
    from vllm_kunlun.ops.fp8 import fused_indexer_q_rope_quant_kunlun

    mod.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_kunlun
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched V4 Indexer Q RoPE/FP8 quantization"
    )


_register_post_import_hook(
    "vllm.models.deepseek_v4.attention",
    _v4_indexer_q_applied,
    _v4_indexer_q_apply,
)



def _v4_attention_mm_dtype_applied(mod):
    return getattr(mod, "_kunlun_mm_dtype_library", None) is not None


def _v4_attention_mm_dtype_apply(mod):
    torch = mod.torch
    library = torch.library.Library("aten", "IMPL", "CUDA")

    def _kunlun_mm_dtype(input, mat2, out_dtype):
        return torch.mm(input.to(out_dtype), mat2.to(out_dtype))

    library.impl("mm.dtype", _kunlun_mm_dtype)
    mod._kunlun_mm_dtype_library = library
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] registered V4 aten::mm.dtype fallback"
    )


_register_post_import_hook(
    "vllm.models.deepseek_v4.attention",
    _v4_attention_mm_dtype_applied,
    _v4_attention_mm_dtype_apply,
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
        if _private not in sys.modules:
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



# --- hook: DeepSeek V4 FlashMLA padded heads (Kunlun does not need NVIDIA h_q alignment) ---
def _flashmla_padded_heads_applied(mod):
    cls = getattr(mod, "DeepseekV4FlashMLAAttention", None)
    if cls is None:
        return False
    return getattr(cls, "_kunlun_no_pad", False)


def _flashmla_padded_heads_apply(mod):
    cls = getattr(mod, "DeepseekV4FlashMLAAttention", None)
    if cls is None:
        return

    @classmethod
    def _kunlun_get_padded_num_q_heads(cls_, num_heads: int) -> int:
        return num_heads

    cls.get_padded_num_q_heads = _kunlun_get_padded_num_q_heads
    cls._kunlun_no_pad = True
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched FlashMLA get_padded_num_q_heads (no padding)"
    )


_register_post_import_hook(
    "vllm.models.deepseek_v4.nvidia.flashmla",
    _flashmla_padded_heads_applied,
    _flashmla_padded_heads_apply,
)


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


# gpu_model_runner is imported during engine init, i.e. after the quantization
# modules that trigger the lazy op registration.
_register_post_import_hook(
    "vllm.v1.worker.gpu_model_runner",
    _op_inventory_final_applied,
    _op_inventory_final_apply,
)


# ---- DeepSeek-V4 OOT adapters -----------------------------------------------
# The per-feature monkey-patches for DeepSeek-V4-on-Kunlum live in the adapter
# package below rather than being hard-coded here. Calling install_all() keeps
# the original plugin bootstrap code unchanged while still enabling incremental
# migration of V4-specific hooks.
try:
    from .adapters.dsv4.installer import (
        install_all as _install_v4_adapters,
    )

    _install_v4_adapters(_register_post_import_hook)
except Exception:
    logging.getLogger("vllm_kunlun").exception(
        "[KunlunPlugin] DSV4 adapter pack failed to load"
    )

