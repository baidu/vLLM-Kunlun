"""vllm kunlun init"""

import builtins
import importlib
import logging
import os
import sys

from vllm.logger import init_logger as init_vllm_logger

OLD_IMPORT_HOOK = builtins.__import__


def _configure_kunlun_logger() -> logging.Logger:
    """Reuse vLLM's handler for the vllm_kunlun logger tree."""
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
    "vllm.v1.attention.ops.merge_attn_states": "vllm_kunlun.v1.attention.ops.merge_attn_states",
    "vllm.v1.attention.ops.triton_merge_attn_states": "vllm_kunlun.v1.attention.ops.triton_merge_attn_states",
    "vllm.v1.worker.mamba_utils": "vllm_kunlun.v1.worker.mamba_utils",
    "vllm.v1.attention.backends.mla.flashattn_mla": "vllm_kunlun.v1.attention.backends.mla.flashattn_mla",
    "vllm.models.kimi_k3.nvidia.mla": "vllm_kunlun.models.kimi_k3.nvidia.mla",
    "vllm.models.kimi_k3.nvidia.ops.attn_res": "vllm_kunlun.models.kimi_k3.nvidia.ops.attn_res",
    "vllm.models.kimi_k3.nvidia.kda": "vllm_kunlun.models.kimi_k3.nvidia.kda",
    # "vllm.v1.worker.gpu_model_runner": "vllm_kunlun.v1.worker.gpu_model_runner",
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


# --- hook 4b: Kimi-K3 KDA torch fallbacks ---------------------------------
# Every KDA kernel (causal_conv1d_*, chunk_kda_*, fused_recurrent_kda*,
# gather_initial_states, rms_norm_gated) is triton-only and triton cannot load
# binaries on P800. One torch implementation replaces one kernel; see
# vllm_kunlun/ops/kda.py for why the kunlun gated-delta-rule kernel cannot be
# used instead. Each module gets its own hook: importing an upstream module from
# inside a hook would run that package's relative imports before _MODULE_MAPPINGS
# can redirect them.
def _kda_applied(mod):
    return getattr(mod, "_kunlun_kda_patched", False)


def _kda_model_apply(mod):
    from vllm_kunlun.ops.kda import patch_kda_model

    patch_kda_model(mod)


def _kda_ops_apply(mod):
    from vllm_kunlun.ops.kda import patch_kda_ops

    patch_kda_ops(mod)


def _kda_norm_apply(mod):
    from vllm_kunlun.ops.kda import (
        patch_rms_norm_gated,
        register_oot_rms_norm_gated,
    )

    patch_rms_norm_gated(mod)
    register_oot_rms_norm_gated(mod)


_register_post_import_hook(
    "vllm.models.kimi_k3.nvidia.kda", _kda_applied, _kda_model_apply
)
_register_post_import_hook(
    "vllm.models.kimi_k3.nvidia.ops.third_party.kda", _kda_applied, _kda_ops_apply
)
_register_post_import_hook(
    "vllm.third_party.flash_linear_attention.ops.kda", _kda_applied, _kda_norm_apply
)


# --- hook 4c: SiluAndMul / SituAndMul forward_native --------------------
# Kimi-K3 and other models instantiate upstream SiluAndMul / SituAndMul
# directly, so the class methods must be patched at their definition site
# rather than relying on a `from vllm_kunlun.ops.activation import ...`
# side-effect. Import the kunlun module once the upstream one is loaded;
# it applies the monkey-patch idempotently.
def _activation_applied(mod):
    silu = getattr(mod, "SiluAndMul", None)
    situ = getattr(mod, "SituAndMul", None)
    silu_ok = silu is None or getattr(silu, "_kunlun_silu_and_mul_patched", False)
    situ_ok = situ is None or getattr(situ, "_kunlun_situ_and_mul_patched", False)
    return silu_ok and situ_ok


def _activation_apply(mod):
    import vllm_kunlun.ops.activation 


_register_post_import_hook(
    "vllm.model_executor.layers.activation",
    _activation_applied,
    _activation_apply,
)


# --- hook 5: Worker._maybe_get_memory_pool_context -----------------------
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


# --- hook: MessageQueue remote-bind port-conflict retry (P800) ------------
# On P800 the single get_open_port()+zmq bind in MessageQueue.__init__ can hit
# an already-in-use port. Wrap __init__ to retry (re-rolls the port each time),
# instead of intrusively editing vllm's shm_broadcast.py.
def _shm_mq_applied(mod):
    cls = getattr(mod, "MessageQueue", None)
    return cls is None or getattr(cls, "_kunlun_bind_retry", False)


def _shm_mq_apply(mod):
    import zmq

    cls = getattr(mod, "MessageQueue", None)
    if cls is None:
        return
    _orig_init = cls.__init__
    _MAX = 30

    def _init_with_retry(self, *args, **kwargs):
        for attempt in range(_MAX):
            try:
                return _orig_init(self, *args, **kwargs)
            except zmq.error.ZMQError as e:
                if attempt == _MAX - 1:
                    raise
                logging.getLogger("vllm_kunlun").warning(
                    "[KunlunPlugin] MessageQueue bind port conflict, "
                    "retry %d/%d: %s",
                    attempt + 1,
                    _MAX,
                    e,
                )

    cls.__init__ = _init_with_retry
    cls._kunlun_bind_retry = True
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched MessageQueue.__init__ with bind-retry"
    )


_register_post_import_hook(
    "vllm.distributed.device_communicators.shm_broadcast",
    _shm_mq_applied,
    _shm_mq_apply,
)


# --- hook: kda_metadata mamba state-indices Triton -> torch (P800) --------
# _mamba_get_block_table_tensor uses the Triton kernel
# _get_aligned_state_indices_kernel, which does not run on P800. Replace it
# with a torch-native gather: for each request take num_state_slots block ids
# from block_table starting at the block holding the current mamba state
# (col = max((seq_len - 1)//block_size, 0)).
def _kda_state_indices_applied(mod):
    fn = getattr(mod, "_mamba_get_block_table_tensor", None)
    return fn is None or getattr(fn, "_kunlun_torch_native", False)


def _kda_state_indices_apply(mod):
    import torch

    def _mamba_get_block_table_tensor(
        block_table, seq_lens, kv_cache_spec, mamba_cache_mode
    ):
        if mamba_cache_mode in ("all", "none"):
            return block_table
        num_state_slots = 1 + kv_cache_spec.num_speculative_blocks
        block_size = kv_cache_spec.block_size
        first = torch.clamp((seq_lens.long() - 1) // block_size, min=0)
        slots = torch.arange(num_state_slots, device=block_table.device)
        cols = (first[:, None] + slots[None, :]).clamp_(
            max=block_table.shape[1] - 1
        )
        return torch.gather(block_table, 1, cols.long())

    _mamba_get_block_table_tensor._kunlun_torch_native = True
    mod._mamba_get_block_table_tensor = _mamba_get_block_table_tensor
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched kda_metadata._mamba_get_block_table_tensor -> torch"
    )


_register_post_import_hook(
    "vllm.models.kimi_k3.nvidia.kda_metadata",
    _kda_state_indices_applied,
    _kda_state_indices_apply,
)


# --- hook: precompute non_spec_query_start_loc_cpu at KDA metadata build -----
# kunlun_ops.causal_conv1d_fwd takes query_start_loc as a python list, so
# vllm_kunlun.ops.kda.causal_conv1d_fn needs a host copy every KDA layer.
# KimiK3KDAMetadataBuilder.build already derives non_spec_query_start_loc_cpu
# with CPU-only ops, but discards it: GDNAttentionMetadata declares no such
# field and the constructor call never passes it. The consumer therefore falls
# back to query_start_loc.tolist() -- one device sync per KDA layer in the hot
# path, and the place where an unrelated async XPU kernel fault surfaces as
# "wait for noc idle timeout".
# Re-derive it here from CPU tensors only, keyed off the same branch condition
# the builder uses (num_spec_decodes == 0), and validate the length against the
# device tensor before attaching. On mismatch we leave it unset so ops/kda.py
# keeps its fallback.
def _kda_qsl_cpu_applied(mod):
    cls = getattr(mod, "KimiK3KDAMetadataBuilder", None)
    return cls is None or getattr(cls.build, "_kunlun_qsl_cpu", False)


def _kda_qsl_cpu_apply(mod):
    import torch

    cls = getattr(mod, "KimiK3KDAMetadataBuilder", None)
    if cls is None:
        return
    _orig_build = cls.build

    def _derive(md, common_attn_metadata, num_decode_draft_tokens_cpu, dev):
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        if md.num_spec_decodes == 0:
            # Builder reuses query_start_loc verbatim for the whole batch.
            out = qsl_cpu.to(torch.int32)
        elif num_decode_draft_tokens_cpu is None:
            return None
        else:
            # Spec path: cumsum over the active (non-spec, non-empty) requests.
            spec_mask = num_decode_draft_tokens_cpu >= 0
            query_lens_cpu = qsl_cpu.diff()
            active = (~spec_mask) & (query_lens_cpu > 0)
            lens = query_lens_cpu[active]
            out = torch.zeros(lens.numel() + 1, dtype=torch.int32)
            torch.cumsum(lens, dim=0, out=out[1:])
        return out if out.numel() == dev.numel() else None

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        md = _orig_build(
            self,
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        dev = getattr(md, "non_spec_query_start_loc", None)
        if dev is not None and getattr(md, "non_spec_query_start_loc_cpu", None) is None:
            cpu = _derive(md, common_attn_metadata, num_decode_draft_tokens_cpu, dev)
            if cpu is not None:
                md.non_spec_query_start_loc_cpu = cpu
        return md

    build._kunlun_qsl_cpu = True
    cls.build = build
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched KimiK3KDAMetadataBuilder.build to attach "
        "non_spec_query_start_loc_cpu"
    )


_register_post_import_hook(
    "vllm.models.kimi_k3.nvidia.kda_metadata",
    _kda_qsl_cpu_applied,
    _kda_qsl_cpu_apply,
)


# --- hook: precompute prefill.query_start_loc_cpu at metadata build ---------
# MLACommonPrefillMetadata carries only query_start_loc (device); the Kunlun MLA
# prefill backend needs a CPU copy for kunlun_ops.attention's context_seq_lod_cpu.
# Instead of a per-layer .cpu() in the hot path, attach it once per build here,
# reusing the CPU query_start_loc the builder already has (pure CPU slice, no
# extra D2H). Runs on host each step for both eager and cudagraph modes
# (metadata build is never inside a captured graph).
def _mla_qsl_cpu_applied(mod):
    cls = getattr(mod, "MLACommonMetadataBuilder", None)
    return cls is None or getattr(cls.build, "_kunlun_qsl_cpu", False)


def _mla_qsl_cpu_apply(mod):
    cls = getattr(mod, "MLACommonMetadataBuilder", None)
    if cls is None:
        return
    _orig_build = cls.build

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = _orig_build(self, common_prefix_len, common_attn_metadata, fast_build)
        prefill = getattr(md, "prefill", None)
        if prefill is not None and getattr(prefill, "query_start_loc_cpu", None) is None:
            full_cpu = common_attn_metadata.query_start_loc_cpu
            # prefill slice starts at reqs_start = num_decodes; recover it from
            # the (num_prefills + 1)-length prefill query_start_loc.
            reqs_start = full_cpu.numel() - prefill.query_start_loc.numel()
            prefill.query_start_loc_cpu = (
                full_cpu[reqs_start:] - full_cpu[reqs_start]
            )
        # Precompute the chunked-context key/value cu_seq_lens CPU copy once per
        # build (shared across all layers/chunks) so run_prefill_context_chunk
        # does not do a per-layer .cpu() for context_kvlen_lod_cpu.
        chunked = getattr(prefill, "chunked_context", None) if prefill else None
        if (
            chunked is not None
            and getattr(chunked, "cu_seq_lens", None) is not None
            and getattr(chunked, "cu_seq_lens_cpu", None) is None
        ):
            chunked.cu_seq_lens_cpu = chunked.cu_seq_lens.cpu()
        return md

    build._kunlun_qsl_cpu = True
    cls.build = build
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched MLACommonMetadataBuilder.build to attach "
        "prefill.query_start_loc_cpu"
    )


_register_post_import_hook(
    "vllm.model_executor.layers.attention.mla_attention",
    _mla_qsl_cpu_applied,
    _mla_qsl_cpu_apply,
)


# --- hook: MLA key concat -> xspeedgate_ops.concat_k_nope_k_pe ------------
# Upstream MLACommonBaseImpl._concat_k_nope_k_pe allocates an empty key and
# fills it with two strided slice-copies (mla_attention.py:2247-2258). On K3
# only the chunked-context path reaches it -- the new-token concat already
# lives inside fused_kimi_k3_mla_key_concat_kv_cache_insert -- and it runs
# once per context chunk per MLA layer, with the token dim growing as context
# accumulates (measured 8192 -> 57344 over 7 chunks).
#
# The XPU op has hard layout constraints, so every call is screened and falls
# back to upstream when they do not hold.
_CONCAT_K_DTYPES = ("float16", "bfloat16", "float32")


def _concat_k_supported(k_nope, k_pe) -> bool:
    if k_nope.dtype is not k_pe.dtype:
        return False
    if str(k_nope.dtype).rsplit(".", 1)[-1] not in _CONCAT_K_DTYPES:
        return False
    if k_nope.dim() != 3 or k_pe.dim() != 3:
        return False
    # k_pe is broadcast over heads, so it must be [num_tokens, 1, pe_dim].
    if k_pe.shape[0] != k_nope.shape[0] or k_pe.shape[1] != 1:
        return False
    if k_nope.stride(-1) != 1 or k_pe.stride(-1) != 1:
        return False
    if k_nope.device != k_pe.device:
        return False
    # nope_dim, pe_dim and the output row must each be 64-byte aligned.
    width = k_nope.element_size()
    nope_dim, pe_dim = k_nope.shape[-1], k_pe.shape[-1]
    return all(
        (dim * width) % 64 == 0 for dim in (nope_dim, pe_dim, nope_dim + pe_dim)
    )


def _mla_concat_k_applied(mod):
    cls = getattr(mod, "MLACommonBaseImpl", None)
    return cls is None or getattr(
        cls._concat_k_nope_k_pe, "_kunlun_concat_k", False
    )


def _mla_concat_k_apply(mod):
    import torch

    cls = getattr(mod, "MLACommonBaseImpl", None)
    if cls is None:
        return

    upstream = cls._concat_k_nope_k_pe
    log = logging.getLogger("vllm_kunlun")

    # torch.ops.xspeedgate_ops.* only resolves once the extension module has
    # been imported, and this hook can fire before anything else pulls it in.
    import xspeedgate_ops  # noqa: F401

    op = getattr(torch.ops.xspeedgate_ops, "concat_k_nope_k_pe", None)
    if op is None:
        # Older xspeedgate build. Mark the untouched method as handled so the
        # dispatcher (which re-runs on every import) stops retrying.
        upstream._kunlun_concat_k = True
        log.info(
            "[KunlunPlugin] xspeedgate_ops.concat_k_nope_k_pe not available, "
            "keeping upstream _concat_k_nope_k_pe"
        )
        return

    def _concat_k_nope_k_pe(self, k_nope, k_pe):
        if _concat_k_supported(k_nope, k_pe):
            return op(k_nope, k_pe)
        return upstream(self, k_nope, k_pe)

    _concat_k_nope_k_pe._kunlun_concat_k = True
    cls._concat_k_nope_k_pe = _concat_k_nope_k_pe
    log.info(
        "[KunlunPlugin] MLACommonBaseImpl._concat_k_nope_k_pe -> xspeedgate_ops"
    )


_register_post_import_hook(
    "vllm.model_executor.layers.attention.mla_attention",
    _mla_concat_k_applied,
    _mla_concat_k_apply,
)


# --- hook: skip fa4_cutedsl_warmup on Kunlun XPU --------------------------
# FA4 CuTeDSL MLA-prefill warmup is NV-only. Because our FlashAttnPrefillBackend
# override reports get_name()=="FLASH_ATTN", fa4_cutedsl_warmup() does not early
# return and imports upstream vllm...mla.prefill.flash_attn, whose top-level
# `from fa_utils import compile_flash_attn_varlen_func_from_specs` fails on this
# vllm build (symbol absent) -> ImportError kills every worker. No-op it.
def _fa4_warmup_applied(mod):
    fn = getattr(mod, "fa4_cutedsl_warmup", None)
    return fn is not None and getattr(fn, "_kunlun_patched", False)


def _fa4_warmup_apply(mod):
    if not hasattr(mod, "fa4_cutedsl_warmup"):
        return

    def _noop(*args, **kwargs):
        logging.getLogger("vllm_kunlun").info(
            "[KunlunPlugin] Skipping fa4_cutedsl_warmup (NV-only)"
        )

    _noop._kunlun_patched = True
    mod.fa4_cutedsl_warmup = _noop
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched kernel_warmup.fa4_cutedsl_warmup -> no-op"
    )


_register_post_import_hook(
    "vllm.model_executor.warmup.kernel_warmup",
    _fa4_warmup_applied,
    _fa4_warmup_apply,
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
        pass

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


# --- hook: MooncakeConnector mamba region registration ---------------------
# upstream unpacks MambaSpec caches as (conv_state, ssm_state); the current
# gpu_model_runner hands out a single contiguous page view per mamba layer.
# See vllm_kunlun/distributed/mooncake_mamba_region.py for the reasoning.
def _mooncake_mamba_region_applied(mod):
    from vllm_kunlun.distributed.mooncake_mamba_region import applied as _applied

    return _applied(mod)


def _mooncake_mamba_region_apply(mod):
    from vllm_kunlun.distributed.mooncake_mamba_region import apply as _apply

    _apply(mod)


_register_post_import_hook(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector",
    _mooncake_mamba_region_applied,
    _mooncake_mamba_region_apply,
)


# --- hook: keep CUDA-style device selection under data parallelism ---------
# KunlunPlatform is PlatformEnum.OOT, so vLLM shards the device list per DP
# rank while the worker still applies the CUDA DP local-rank offset. See
# vllm_kunlun/distributed/dp_device_assign.py for the failure it causes.
def _dp_device_assign_applied(mod):
    from vllm_kunlun.distributed.dp_device_assign import applied as _applied

    return _applied(mod)


def _dp_device_assign_apply(mod):
    from vllm_kunlun.distributed.dp_device_assign import apply as _apply

    _apply(mod)


_register_post_import_hook(
    "vllm.v1.engine.utils",
    _dp_device_assign_applied,
    _dp_device_assign_apply,
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

    # --- override MLA prefill backend selection for P800 ---
    # get_mla_prefill_backend() only knows about FLASH_ATTN on non-Blackwell
    # devices, and its class (vllm...prefill.flash_attn.FlashAttnPrefillBackend)
    # needs flash_attn_varlen_func which is unavailable on Kunlun -> ImportError
    # -> "No valid MLA prefill backend found". Override FLASH_ATTN to our
    # kunlun_ops-backed implementation so selection + import succeed.
    try:
        from vllm.v1.attention.backends.mla.prefill.registry import (
            MLAPrefillBackendEnum,
            register_mla_prefill_backend,
        )

        register_mla_prefill_backend(
            MLAPrefillBackendEnum.FLASH_ATTN,
            "vllm_kunlun.v1.attention.backends.mla.prefill.flash_attn."
            "FlashAttnPrefillBackend",
        )
        logger.info(
            "[KunlunPlugin] registered Kunlun FlashAttnPrefillBackend override "
            "for MLAPrefillBackendEnum.FLASH_ATTN"
        )
    except Exception:
        logger.exception(
            "[KunlunPlugin] failed to register MLA prefill backend override"
        )
        raise

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
