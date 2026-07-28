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
    from vllm_kunlun.ops.deepseek_v4_topk import (
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
    from vllm_kunlun.ops.fused_moe.layer import KunlunFp8MoEMethod
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


# Kunlun cannot execute the CUDA Triton compressed-slot metadata kernel. This
# fallback mirrors compressor_utils' integer mapping and only moves metadata.
def _compressed_slot_mapping_applied(mod):
    fn = getattr(mod, "get_compressed_slot_mapping", None)
    return fn is not None and getattr(fn, "_kunlun_patched", False)


def _compressed_slot_mapping_apply(mod):
    import torch

    def _get_compressed_slot_mapping(
        num_tokens, query_start_loc, seq_lens, block_table, block_size,
        compress_ratio, out=None,
    ):
        if out is None:
            result = torch.full(
                (num_tokens,), -1, dtype=torch.int64,
                device=query_start_loc.device,
            )
        else:
            out.fill_(-1)
            result = out[:num_tokens]

        starts = query_start_loc.cpu().tolist()
        lengths = seq_lens.cpu().tolist()
        tables = block_table.cpu()
        result_cpu = torch.full((num_tokens,), -1, dtype=torch.int64)
        for req_idx, query_start in enumerate(starts[:-1]):
            query_end = starts[req_idx + 1]
            query_len = query_end - query_start
            start_pos = lengths[req_idx] - query_len
            for offset in range(query_len):
                pos = start_pos + offset
                if (pos + 1) % compress_ratio != 0:
                    continue
                compressed_pos = pos // compress_ratio
                block_id = compressed_pos // block_size
                result_cpu[query_start + offset] = (
                    tables[req_idx, block_id].item() * block_size
                    + compressed_pos % block_size
                )
        result.copy_(result_cpu.to(result.device))
        return out if out is not None else result

    _get_compressed_slot_mapping._kunlun_patched = True
    mod.get_compressed_slot_mapping = _get_compressed_slot_mapping


_register_post_import_hook(
    "vllm.v1.attention.backends.mla.compressor_utils",
    _compressed_slot_mapping_applied,
    _compressed_slot_mapping_apply,
)
_register_post_import_hook(
    "vllm.v1.attention.backends.mla.indexer",
    _compressed_slot_mapping_applied,
    _compressed_slot_mapping_apply,
)


def _indexer_prefill_kernel_applied(mod):
    kernel = getattr(mod, "_build_prefill_chunk_metadata_kernel", None)
    return getattr(kernel, "_kunlun_patched", False)


def _indexer_prefill_kernel_apply(mod):
    import torch

    class _CpuLaunchable:
        _kunlun_patched = True

        def __getitem__(self, grid):
            def launch(
                query_start_loc,
                uncompressed_seq_lens,
                cu_compressed_seq_lens,
                row_start_cu_compressed_seq_lens,
                token_to_seq,
                cu_seq_len_ks,
                cu_seq_len_ke,
                query_slice_start,
                query_slice_stop,
                dcp_rank,
                dcp_world,
                dcp_interleave,
                *,
                BLOCK_SIZE,
                COMPRESS_RATIO,
            ):
                starts = query_start_loc.cpu().tolist()
                uncompressed = uncompressed_seq_lens.cpu().tolist()
                compressed_cu = cu_compressed_seq_lens.cpu().tolist()
                local_cu = row_start_cu_compressed_seq_lens.cpu().tolist()
                token_count = token_to_seq.numel()
                token_cpu = torch.empty(token_count, dtype=torch.int32)
                ks_cpu = torch.empty(cu_seq_len_ks.numel(), dtype=torch.int32)
                ke_cpu = torch.empty(cu_seq_len_ke.numel(), dtype=torch.int32)
                for req_idx in range(len(uncompressed)):
                    query_start = starts[req_idx]
                    query_end = starts[req_idx + 1]
                    query_len = query_end - query_start
                    start_pos = uncompressed[req_idx] - query_len
                    row_start = local_cu[req_idx]
                    for offset in range(query_len):
                        absolute = query_start + offset
                        if query_slice_start <= absolute < query_slice_stop:
                            out_pos = absolute - query_slice_start
                            context = (start_pos + 1 + offset) // COMPRESS_RATIO
                            if dcp_world > 1:
                                base = (context // dcp_interleave // dcp_world) * dcp_interleave
                                remainder = context - base * dcp_world
                                context = base + min(
                                    max(remainder - dcp_rank * dcp_interleave, 0),
                                    dcp_interleave,
                                )
                            ks_cpu[out_pos] = row_start
                            ke_cpu[out_pos] = row_start + context
                    seq_start = compressed_cu[req_idx]
                    seq_end = compressed_cu[req_idx + 1]
                    token_cpu[seq_start:seq_end] = req_idx
                token_to_seq.copy_(token_cpu.to(token_to_seq.device))
                cu_seq_len_ks.copy_(ks_cpu.to(cu_seq_len_ks.device))
                cu_seq_len_ke.copy_(ke_cpu.to(cu_seq_len_ke.device))

            return launch

    mod._build_prefill_chunk_metadata_kernel = _CpuLaunchable()


_register_post_import_hook(
    "vllm.v1.attention.backends.mla.indexer",
    _indexer_prefill_kernel_applied,
    _indexer_prefill_kernel_apply,
)


def _c128a_metadata_applied(mod):
    fn = getattr(mod, "build_c128a_topk_metadata", None)
    return fn is not None and getattr(fn, "_kunlun_patched", False)


def _c128a_metadata_apply(mod):
    import torch

    def _build_c128a_topk_metadata(
        positions,
        compress_ratio,
        num_decode_tokens,
        token_to_req_indices,
        block_table,
        block_size,
        slot_mapping,
        global_decode_buffer,
        decode_lens_buffer,
        prefill_buffer,
        max_compressed_tokens=8192,
    ):
        num_tokens = positions.shape[0]
        num_prefill_tokens = num_tokens - num_decode_tokens
        global_decode = global_decode_buffer[:num_decode_tokens]
        decode_lens = decode_lens_buffer[:num_decode_tokens]
        prefill_local = prefill_buffer[:num_prefill_tokens]
        if num_tokens == 0:
            return global_decode, decode_lens, prefill_local

        global_decode_buffer[:num_decode_tokens].fill_(-1)
        decode_lens_buffer[:num_decode_tokens].zero_()
        prefill_buffer[:num_prefill_tokens].fill_(-1)
        positions_cpu = positions.cpu().tolist()
        req_cpu = token_to_req_indices.cpu().tolist()
        table_cpu = block_table.cpu()
        slots_cpu = slot_mapping.cpu().tolist()
        for token_idx, position in enumerate(positions_cpu):
            num_compressed = min(
                (position + 1) // compress_ratio, max_compressed_tokens
            )
            if token_idx < num_decode_tokens:
                if slots_cpu[token_idx] < 0:
                    continue
                req_idx = req_cpu[token_idx]
                values = [
                    table_cpu[req_idx, offset // block_size].item() * block_size
                    + offset % block_size
                    for offset in range(num_compressed)
                ]
                if values:
                    global_decode_buffer[token_idx, :num_compressed].copy_(
                        torch.tensor(
                            values,
                            dtype=global_decode_buffer.dtype,
                            device=global_decode_buffer.device,
                        )
                    )
                    decode_lens_buffer[token_idx] = num_compressed
            else:
                row = token_idx - num_decode_tokens
                if num_compressed:
                    prefill_buffer[row, :num_compressed].copy_(
                        torch.arange(
                            num_compressed,
                            dtype=prefill_buffer.dtype,
                            device=prefill_buffer.device,
                        )
                    )
        return global_decode, decode_lens, prefill_local

    _build_c128a_topk_metadata._kunlun_patched = True
    mod.build_c128a_topk_metadata = _build_c128a_topk_metadata


_register_post_import_hook(
    "vllm.models.deepseek_v4.sparse_mla",
    _c128a_metadata_applied,
    _c128a_metadata_apply,
)


def _swa_kernel_applied(mod):
    kernel = getattr(mod, "_compute_swa_indices_and_lens_kernel", None)
    return getattr(kernel, "_kunlun_patched", False)


def _swa_kernel_apply(mod):
    import torch

    class _CpuLaunchable:
        _kunlun_patched = True

        def __getitem__(self, grid):
            def launch(
                swa_indices,
                swa_indices_stride,
                swa_lens,
                window_size,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                block_table_stride,
                block_size,
                token_offset,
                *,
                TRITON_BLOCK_SIZE,
            ):
                starts = query_start_loc.cpu().tolist()
                lengths = seq_lens.cpu().tolist()
                reqs = token_to_req_indices.cpu().tolist()
                valid = is_valid_token.cpu().tolist()
                tables = block_table.cpu()
                rows = swa_indices.shape[0]
                width = swa_indices.shape[1]
                swa_indices.fill_(-1)
                lens_cpu = torch.zeros(rows, dtype=torch.int32)
                for pid in range(rows):
                    token_idx = pid + token_offset
                    if token_idx >= len(valid) or not valid[token_idx]:
                        continue
                    req_idx = reqs[token_idx]
                    query_len = starts[req_idx + 1] - starts[req_idx]
                    prefix_len = lengths[req_idx] - query_len
                    pos = prefix_len + token_idx - starts[req_idx]
                    start_pos = max(pos - window_size + 1, 0)
                    end_pos = pos + 1
                    swa_len = end_pos - start_pos
                    lens_cpu[pid] = swa_len
                    values = [
                        tables[req_idx, p // block_size].item() * block_size
                        + p % block_size
                        for p in range(start_pos, end_pos)
                    ]
                    if values:
                        swa_indices[pid, :len(values)].copy_(
                            torch.tensor(
                                values,
                                dtype=swa_indices.dtype,
                                device=swa_indices.device,
                            )
                        )
                swa_lens[:rows].copy_(lens_cpu.to(swa_lens.device))

            return launch

    mod._compute_swa_indices_and_lens_kernel = _CpuLaunchable()


_register_post_import_hook(
    "vllm.v1.attention.backends.mla.sparse_swa",
    _swa_kernel_applied,
    _swa_kernel_apply,
)


def _prefill_metadata_kernel_applied(mod):
    kernel = getattr(mod, "_compute_prefill_metadata_kernel", None)
    return getattr(kernel, "_kunlun_patched", False)


def _prefill_metadata_kernel_apply(mod):
    import torch

    class _CpuLaunchable:
        _kunlun_patched = True

        def __getitem__(self, grid):
            def launch(
                prefill_gather_lens,
                seq_lens,
                query_start_loc,
                num_prefills,
                num_decodes,
                window_size,
                *,
                BLOCK_SIZE,
            ):
                lengths = seq_lens.cpu().tolist()
                starts = query_start_loc.cpu().tolist()
                values = []
                for offset in range(num_prefills):
                    req_idx = num_decodes + offset
                    query_len = starts[req_idx + 1] - starts[req_idx]
                    prefix_len = lengths[req_idx] - query_len
                    values.append(
                        query_len + min(prefix_len, window_size - 1)
                    )
                prefill_gather_lens.copy_(
                    torch.tensor(
                        values,
                        dtype=prefill_gather_lens.dtype,
                        device=prefill_gather_lens.device,
                    )
                )

            return launch

    mod._compute_prefill_metadata_kernel = _CpuLaunchable()


_register_post_import_hook(
    "vllm.v1.attention.backends.mla.sparse_swa",
    _prefill_metadata_kernel_applied,
    _prefill_metadata_kernel_apply,
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
    from vllm_kunlun.ops.mhc import (
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
        "vllm_kunlun.ops.mhc"
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
        "vllm_kunlun.ops.mhc"
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
    from vllm_kunlun.ops.mhc import (
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


def _sparse_indexer_oot_applied(mod):
    from vllm.model_executor.custom_op import op_registry_oot

    cls = op_registry_oot.get("SparseAttnIndexer")
    return cls is not None and cls.__module__ == __name__


def _sparse_indexer_oot_apply(mod):
    from vllm.model_executor.custom_op import CustomOp

    @CustomOp.register_oot(name="SparseAttnIndexer")
    class KunlunSparseAttnIndexer(mod.SparseAttnIndexer):
        def forward_oot(self, hidden_states, q_quant, k, weights):
            """Vectorized PyTorch top-K selection for Lightning Indexer on Kunlun.

            Computes Q*K scores for each token against all compressed KV slots
            in the indexer's paged K cache, then selects top-K positions.

            Args:
                hidden_states: [num_tokens, hidden_size] bf16 (unused for score,
                    but shape gives num_tokens)
                q_quant: [num_tokens, n_heads=64, head_dim=128] float8_e4m3fn
                    (from fused_indexer_q_rope_quant_kunlun; stored as int8 on Kunlun)
                k: None (K already written to cache by compressor)
                weights: [num_tokens, n_heads=64] float32
                    (includes q_scale * softmax_scale * head_scale folded in)
            """
            import torch
            from vllm.forward_context import get_forward_context

            num_tokens = hidden_states.shape[0]
            topk_tokens = self.topk_tokens
            device = hidden_states.device

            # Pre-fill buffer with -1 (sentinel for "no token selected")
            self.topk_indices_buffer[:num_tokens] = -1

            # Access per-layer indexer metadata from forward context
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None or not isinstance(attn_metadata, dict):
                # Warmup/profiling: no real metadata available
                return self.topk_indices_buffer

            k_cache_prefix_name = self.k_cache.prefix
            indexer_meta = attn_metadata.get(k_cache_prefix_name)
            if indexer_meta is None:
                return self.topk_indices_buffer

            # Get K cache: [num_blocks, block_size, head_dim_bytes=132] uint8
            # Layout per slot: [128 bytes INT8 K values] + [4 bytes FP32 scale]
            kv_cache = self.k_cache.kv_cache
            if kv_cache.numel() == 0:
                return self.topk_indices_buffer

            num_blocks_cache = kv_cache.shape[0]
            block_size_cache = kv_cache.shape[1]
            head_dim_bytes = kv_cache.shape[-1]  # 132
            head_dim = self.head_dim  # 128

            # ---- Decode path ----
            has_decode = indexer_meta.num_decode_tokens > 0
            has_prefill = indexer_meta.num_prefills > 0
            num_decode_tokens = indexer_meta.num_decode_tokens

            if has_decode and indexer_meta.decode is not None:
                decode_meta = indexer_meta.decode
                # decode_meta.seq_lens: [batch] or [batch, next_n] int32
                # decode_meta.block_table: [batch, max_blocks] int32
                seq_lens = decode_meta.seq_lens
                block_table = decode_meta.block_table

                if seq_lens.dim() == 2:
                    # MTP: [B, next_n] -> use last column as effective seq_len
                    effective_seq_lens = seq_lens[:, -1]
                else:
                    effective_seq_lens = seq_lens

                batch_size = effective_seq_lens.shape[0]

                # Q: dequantize fp8 to float for score computation
                # q_quant is stored as int8 (Kunlun cannot cast fp8->bf16)
                # Shape: [num_tokens, 64, 128]
                q_decode = q_quant[:num_decode_tokens]
                w_decode = weights[:num_decode_tokens]  # [num_decode_tokens, 64]

                # Dequant Q: treat as raw int8 values (scale already in weights)
                q_float = q_decode.view(torch.int8).float()  # [T, 64, 128]

                # Compute weighted Q sum across heads:
                # q_summed[t, d] = sum_h(q[t,h,d] * w[t,h])
                # This reduces multi-head to single-head for MQA-style K cache
                q_weighted = q_float * w_decode.unsqueeze(-1)  # [T, 64, 128]
                q_summed = q_weighted.sum(dim=1)  # [T, 128]

                # Per-token top-K selection against paged K cache
                for t in range(num_decode_tokens):
                    if t >= batch_size:
                        break
                    seq_len = int(effective_seq_lens[t].item())
                    if seq_len <= 0:
                        continue

                    # Gather K from paged cache for this request
                    num_blocks_needed = (seq_len + block_size_cache - 1) // block_size_cache
                    bt = block_table[t, :num_blocks_needed].long()

                    # Collect all valid slots
                    slots_per_block = min(block_size_cache, seq_len)
                    all_k_int8 = []
                    all_k_scale = []
                    total_gathered = 0

                    for blk_idx in range(num_blocks_needed):
                        blk_id = int(bt[blk_idx].item())
                        remaining = seq_len - total_gathered
                        n_slots = min(block_size_cache, remaining)
                        if n_slots <= 0:
                            break
                        # kv_cache[blk_id, :n_slots, :] -> [n_slots, 132]
                        blk_data = kv_cache[blk_id, :n_slots, :]
                        k_data = blk_data[:, :head_dim]  # [n_slots, 128] uint8 = int8
                        k_scale_raw = blk_data[:, head_dim:head_dim+4].contiguous()
                        # Interpret 4 bytes as float32 scale
                        k_scale_f32 = k_scale_raw.view(torch.float32)  # [n_slots, 1]
                        all_k_int8.append(k_data)
                        all_k_scale.append(k_scale_f32)
                        total_gathered += n_slots

                    if not all_k_int8:
                        continue

                    # Concatenate all gathered K: [seq_len_actual, 128]
                    k_int8_cat = torch.cat(all_k_int8, dim=0)
                    k_scale_cat = torch.cat(all_k_scale, dim=0)  # [seq_len_actual, 1]

                    # Dequant K: k_float = k_int8 * scale
                    k_float = k_int8_cat.view(torch.int8).float() * k_scale_cat  # [S, 128]

                    # Score: q_summed[t] @ k_float^T -> [S]
                    scores = torch.matmul(q_summed[t:t+1], k_float.T).squeeze(0)  # [S]

                    # Top-K selection
                    actual_topk = min(topk_tokens, scores.shape[0])
                    if actual_topk > 0:
                        _, top_idx = scores.topk(actual_topk, dim=0)
                        self.topk_indices_buffer[t, :actual_topk] = top_idx.to(torch.int32)

            if has_prefill and indexer_meta.prefill is not None:
                prefill_meta = indexer_meta.prefill
                # Prefill: iterate over chunks
                # Each chunk has cu_seqlen_ks, cu_seqlen_ke, block_table, token_start/end
                q_prefill = q_quant[num_decode_tokens:num_tokens]
                w_prefill = weights[num_decode_tokens:num_tokens]

                q_float_p = q_prefill.view(torch.int8).float()  # [P, 64, 128]
                q_weighted_p = q_float_p * w_prefill.unsqueeze(-1)  # [P, 64, 128]
                q_summed_p = q_weighted_p.sum(dim=1)  # [P, 128]

                for chunk in prefill_meta.chunks:
                    token_start = chunk.token_start - num_decode_tokens
                    token_end = chunk.token_end - num_decode_tokens
                    cu_seqlen_ks = chunk.cu_seqlen_ks  # [num_seqs_in_chunk + 1]
                    cu_seqlen_ke = chunk.cu_seqlen_ke  # [num_seqs_in_chunk + 1]

                    if chunk.local_total_seq_lens == 0:
                        continue

                    # Gather K from paged cache
                    # block_table: [num_seqs_in_chunk, max_blocks]
                    chunk_bt = chunk.block_table
                    local_cu = chunk.local_cu_seq_lens  # [num_seqs + 1]

                    if local_cu is None:
                        continue

                    num_seqs = local_cu.shape[0] - 1

                    # Gather all K for this chunk into contiguous buffer
                    # Then per-token topk
                    for seq_idx in range(num_seqs):
                        seq_k_start = int(local_cu[seq_idx].item())
                        seq_k_end = int(local_cu[seq_idx + 1].item())
                        seq_k_len = seq_k_end - seq_k_start

                        if seq_k_len <= 0:
                            continue

                        # Gather K from paged cache for this sequence
                        num_blocks_needed = (seq_k_len + block_size_cache - 1) // block_size_cache
                        bt_row = chunk_bt[seq_idx, :num_blocks_needed].long()

                        all_k_int8 = []
                        all_k_scale = []
                        total_gathered = 0

                        for blk_idx in range(num_blocks_needed):
                            blk_id = int(bt_row[blk_idx].item())
                            remaining = seq_k_len - total_gathered
                            n_slots = min(block_size_cache, remaining)
                            if n_slots <= 0:
                                break
                            blk_data = kv_cache[blk_id, :n_slots, :]
                            k_data = blk_data[:, :head_dim]
                            k_scale_raw = blk_data[:, head_dim:head_dim+4].contiguous()
                            k_scale_f32 = k_scale_raw.view(torch.float32)
                            all_k_int8.append(k_data)
                            all_k_scale.append(k_scale_f32)
                            total_gathered += n_slots

                        if not all_k_int8:
                            continue

                        k_int8_cat = torch.cat(all_k_int8, dim=0)
                        k_scale_cat = torch.cat(all_k_scale, dim=0)
                        k_float = k_int8_cat.view(torch.int8).float() * k_scale_cat

                        # Find which tokens in this chunk belong to this sequence
                        # cu_seqlen_ks/ke define Q boundaries per sequence
                        q_start = int(cu_seqlen_ks[seq_idx].item())
                        q_end = int(cu_seqlen_ke[seq_idx].item())

                        for qt in range(q_start, q_end):
                            local_qt = token_start + qt
                            if local_qt < 0 or local_qt >= q_summed_p.shape[0]:
                                continue
                            # This token can attend to K[0:qt_pos_in_seq] (causal)
                            # qt_pos = qt - q_start gives relative position
                            qt_pos = qt - q_start
                            # In prefill, compressed seq_k_len is the compressed history
                            # All tokens see the full compressed history
                            causal_len = seq_k_len
                            if causal_len <= 0:
                                continue

                            scores = torch.matmul(
                                q_summed_p[local_qt:local_qt+1],
                                k_float[:causal_len].T
                            ).squeeze(0)

                            actual_topk = min(topk_tokens, scores.shape[0])
                            if actual_topk > 0:
                                _, top_idx = scores.topk(actual_topk, dim=0)
                                buf_idx = num_decode_tokens + local_qt
                                self.topk_indices_buffer[buf_idx, :actual_topk] = top_idx.to(torch.int32)

            return self.topk_indices_buffer

    mod.KunlunSparseAttnIndexer = KunlunSparseAttnIndexer
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] registered V4 SparseAttnIndexer OOT path"
    )


_register_post_import_hook(
    "vllm.model_executor.layers.sparse_attn_indexer",
    _sparse_indexer_oot_applied,
    _sparse_indexer_oot_apply,
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
    fn = getattr(mod, "deep_gemm_fp8_o_proj", None)
    return fn is not None and getattr(fn, "__module__", "") == (
        "vllm_kunlun.ops.deep_gemm"
    )


def _v4_o_proj_apply(mod):
    from vllm_kunlun.ops.deep_gemm import deepseek_v4_bf16_o_proj

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


def _kunlun_sqrt_softplus_scores(gating_output):
    """sqrt(softplus(x)) via the on-device XSpeedGate act_sqrt_softplus op.

    Numerically equivalent to ``torch.sqrt(F.softplus(gating_output.float()))``
    (the community reference), but fused into a single XPU kernel launch on the
    router hot path. Falls back to the torch expression if the op is missing or
    raises (e.g. unexpected dtype/device)."""
    import torch
    x = gating_output.float()
    op = getattr(torch.ops.xspeedgate_ops, "act_sqrt_softplus", None)
    if op is not None:
        try:
            return op(x)
        except Exception:
            pass
    import torch.nn.functional as F
    return torch.sqrt(F.softplus(x))


def _kunlun_topk_softplus_sqrt_accel(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize=False,
    e_score_correction_bias=None,
    input_tokens=None,
    hash_indices_table=None,
    routed_scaling_factor=1.0,
):
    """Kunlun-accelerated drop-in for ``_topk_softplus_sqrt_torch``.

    Same signature and semantics as the community router fallback; only the
    sqrt(softplus) score computation is offloaded to ``act_sqrt_softplus``."""
    import torch

    scores = _kunlun_sqrt_softplus_scores(gating_output)
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.float()
    else:
        scores_for_choice = scores
    topk = topk_weights.shape[-1]
    if hash_indices_table is not None and input_tokens is not None:
        expert_ids = hash_indices_table[input_tokens.long()]
        topk_indices.copy_(expert_ids)
        weights = scores.gather(1, expert_ids.long())
    else:
        _, indices = torch.topk(scores_for_choice, k=topk, dim=-1)
        topk_indices.copy_(indices)
        weights = scores.gather(1, indices)
    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True).clamp(min=1e-20))
    topk_weights.copy_(weights * routed_scaling_factor)
    return topk_weights, topk_indices


_kunlun_topk_softplus_sqrt_accel._kunlun_hash_fallback = True
_kunlun_topk_softplus_sqrt_accel._kunlun_sqrt_softplus_accel = True


def _hash_topk_apply(mod):
    def _kunlun_topk_hash_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        input_tokens=None,
        hash_indices_table=None,
    ):
        # Late binding: prefer the accelerated router fn if the router module
        # has already been patched, else fall back to the community torch impl.
        try:
            from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (  # noqa: E501
                _topk_softplus_sqrt_torch as _rt,
            )
        except Exception:
            _rt = _kunlun_topk_softplus_sqrt_accel
        _rt(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
            routed_scaling_factor,
        )

    _kunlun_topk_hash_softplus_sqrt._kunlun_hash_fallback = True
    mod.topk_hash_softplus_sqrt = _kunlun_topk_hash_softplus_sqrt


_register_post_import_hook(
    "vllm._custom_ops",
    _hash_topk_applied,
    _hash_topk_apply,
)


def _hash_router_applied(mod):
    return getattr(
        getattr(mod, "vllm_topk_softplus_sqrt", None),
        "_kunlun_sqrt_softplus_accel",
        False,
    )


def _hash_router_apply(mod):
    # Replace the community sqrt(softplus) router with the Kunlun-accelerated
    # variant on BOTH the reference symbol and the dispatched entry point, so
    # any late-binding lookup of ``_topk_softplus_sqrt_torch`` also picks it up.
    mod._topk_softplus_sqrt_torch = _kunlun_topk_softplus_sqrt_accel
    mod.vllm_topk_softplus_sqrt = _kunlun_topk_softplus_sqrt_accel


_register_post_import_hook(
    "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router",
    _hash_router_applied,
    _hash_router_apply,
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

def _compressor_save_applied(mod):
    return getattr(mod, "_kunlun_compressor_ops_patched", False)


def _compressor_save_apply(mod):
    import torch

    try:
        import kunlun_ops as _k
        _HAS_NATIVE_SPS = hasattr(_k, "save_partial_states")
    except Exception:  # noqa: BLE001
        _HAS_NATIVE_SPS = False

    _sps_warned = [False]

    def _torch_save(kv, score, ape, positions, state_cache, slot_mapping,
                    block_size, state_width, compress_ratio):
        head_size = kv.shape[-1]
        valid_mask = slot_mapping >= 0
        if not valid_mask.any():
            return
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        valid_slots = slot_mapping[valid_indices]
        block_idx = valid_slots // block_size
        pos_in_block = valid_slots % block_size
        state_cache[block_idx, pos_in_block, :head_size] = kv[valid_indices]
        valid_positions = positions[valid_indices]
        ape_rows = valid_positions % compress_ratio
        score_with_ape = score[valid_indices] + ape[ape_rows]
        state_cache[block_idx, pos_in_block, state_width:state_width + head_size] = score_with_ape

    def save_partial_states(
        kv, score, ape, positions, state_cache, slot_mapping,
        block_size, state_width, compress_ratio, pdl_kwargs=None,
    ):
        # Prefer native kunlun_ops.save_partial_states. Native arg order (per
        # docstring): kv, score, ape, positions, slot_mapping, state_cache,
        # block_size, state_width, compress_ratio. Native does inplace fp32
        # writes to state_cache; falls back to torch on any failure.
        if _HAS_NATIVE_SPS:
            try:
                import kunlun_ops
                kunlun_ops.save_partial_states(
                    kv, score, ape, positions, slot_mapping, state_cache,
                    block_size, state_width, compress_ratio,
                )
                return
            except Exception as e:  # noqa: BLE001
                if not _sps_warned[0]:
                    logging.getLogger("vllm_kunlun").warning(
                        "native save_partial_states failed (%s); "
                        "falling back to torch save_partial_states for the "
                        "rest of the run", e)
                    _sps_warned[0] = True
        _torch_save(kv, score, ape, positions, state_cache, slot_mapping,
                    block_size, state_width, compress_ratio)

    mod.save_partial_states = save_partial_states
    mod._kunlun_compressor_ops_patched = True
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched V4 save_partial_states "
        "(native=%s; torch fallback armed)", _HAS_NATIVE_SPS,
    )


_register_post_import_hook(
    "vllm.models.deepseek_v4.common.ops.save_partial_states",
    _compressor_save_applied,
    _compressor_save_apply,
)


def _compressor_compress_applied(mod):
    return getattr(mod, "_kunlun_compress_patched", False)


def _compressor_compress_apply(mod):
    import torch

    def compress_norm_rope_store_triton(
        state_cache, num_actual, token_to_req_indices, positions,
        slot_mapping, block_table, block_size, state_width,
        cos_sin_cache, kv_cache, k_cache_metadata, pdl_kwargs,
        head_dim, rope_head_dim, compress_ratio, overlap,
        use_fp4_cache, rms_norm_weight, rms_norm_eps,
        quant_block, token_stride, scale_dim,
    ):
        coff = 2 if overlap else 1
        window = coff * compress_ratio
        nope_head_dim = head_dim - rope_head_dim
        half_rope = rope_head_dim // 2

        all_positions = positions[:num_actual]
        all_slots = slot_mapping[:num_actual]
        valid_mask = (all_slots >= 0) & ((all_positions + 1) % compress_ratio == 0)
        if not valid_mask.any():
            return

        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        kv_slot_mapping = k_cache_metadata.slot_mapping
        kv_cache_block_size = kv_cache.shape[1] if kv_cache.ndim >= 3 else block_size

        for idx in valid_indices:
            token_idx = idx.item()
            position = all_positions[token_idx].item()
            req_idx = token_to_req_indices[token_idx].item()

            start = position - window + 1
            gather_positions = torch.arange(start, position + 1, device=state_cache.device)
            gather_mask = gather_positions >= 0

            block_indices = gather_positions.clamp(min=0) // block_size
            block_numbers = block_table[req_idx, block_indices]
            block_offsets = gather_positions % block_size

            tokens_in_window = torch.arange(window, device=state_cache.device)
            if overlap:
                head_offset = (tokens_in_window >= compress_ratio).long() * head_dim
            else:
                head_offset = torch.zeros(window, dtype=torch.long, device=state_cache.device)

            kv_states = torch.zeros(window, head_dim, dtype=torch.float32, device=state_cache.device)
            score_states = torch.full((window, head_dim), float("-inf"), dtype=torch.float32, device=state_cache.device)

            for i in range(window):
                if not gather_mask[i]:
                    continue
                bn = block_numbers[i].item()
                bo = block_offsets[i].item()
                ho = head_offset[i].item()
                kv_states[i] = state_cache[bn, bo, ho:ho + head_dim]
                score_states[i] = state_cache[bn, bo, state_width + ho:state_width + ho + head_dim]

            weights = torch.softmax(score_states, dim=0)
            compressed_kv = (kv_states * weights).sum(dim=0)

            variance = (compressed_kv * compressed_kv).mean()
            rrms = torch.rsqrt(variance + rms_norm_eps)
            normed = compressed_kv * rrms * rms_norm_weight.float()

            # GPT-J interleaved RoPE on LAST rope_head_dim dims
            compressed_pos = (position // compress_ratio) * compress_ratio
            cs = cos_sin_cache[compressed_pos]
            cos_vals = cs[:half_rope]
            sin_vals = cs[half_rope:]

            rope_part = normed[nope_head_dim:]
            rope_even = rope_part[0::2]
            rope_odd = rope_part[1::2]
            new_even = rope_even * cos_vals - rope_odd * sin_vals
            new_odd = rope_even * sin_vals + rope_odd * cos_vals
            normed[nope_head_dim::2] = new_even
            normed[nope_head_dim + 1::2] = new_odd

            kv_slot_idx = kv_slot_mapping[token_idx].item()
            if kv_slot_idx < 0:
                continue
            kv_block_idx = kv_slot_idx // kv_cache_block_size
            kv_pos_in_block = kv_slot_idx % kv_cache_block_size
            kv_cache[kv_block_idx, kv_pos_in_block, :head_dim] = normed.to(kv_cache.dtype)

    mod.compress_norm_rope_store_triton = compress_norm_rope_store_triton
    mod._kunlun_compress_patched = True
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched V4 compress_norm_rope_store_triton (PyTorch fallback)"
    )


_register_post_import_hook(
    "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache",
    _compressor_compress_applied,
    _compressor_compress_apply,
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
