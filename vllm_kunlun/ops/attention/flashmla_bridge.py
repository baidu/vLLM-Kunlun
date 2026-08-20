"""Host-side replacements for DeepSeek-V4 Triton metadata/auxiliary kernels.

Kunlun XPU has no Triton, so these small bookkeeping kernels are replaced
with CPU-loop launchers: indexer prefill-chunk metadata, C128A compressed
top-K metadata, SWA indices-and-lengths, and prefill gather-lens.
"""
import logging
from typing import Any, List, Tuple

import torch

from vllm_kunlun.adapter_utils import WarningOnce
from vllm_kunlun.patches.registry import _register_lazy

LOGGER = logging.getLogger("vllm_kunlun.ops.attention.flashmla_bridge")
_INDEXER_KERNEL_ATTR = "_build_prefill_chunk_metadata_kernel"
_C128A_FN_NAME = "build_c128a_topk_metadata"
_SWA_KERNEL_ATTR = "_compute_swa_indices_and_lens_kernel"
_PREFILL_LENS_ATTR = "_compute_prefill_metadata_kernel"
_DSV4_WIRED_ATTR = "_dsv4_wired_by_flashmla_bridge"
_FALSE = object()


def _is_already_wired(obj: Any) -> bool:
    """Check whether an installed dispatcher already carries our label."""
    if obj is None:
        return False
    return getattr(obj, _DSV4_WIRED_ATTR, False)


# Indexer prefill-chunk metadata (replaces the Triton kernel).
def _install_indexer_prefill_kernel(indexer_mod: Any) -> List[str]:
    """Replace ``indexer._build_prefill_chunk_metadata_kernel`` with a CPU-loop launcher.

    Upstream builds compressed-context lookup bounds on GPU using a small
    Triton kernel; Kunlun has no Triton frontend for these bookkeeping tasks,
    so we replicate the same index arithmetic in Python and stage the output
    through pinned-CPU buffers before copying back to the device tensor.
    """
    if _is_already_wired(getattr(indexer_mod, _INDEXER_KERNEL_ATTR, None)):
        return []

    class _CpuLaunchable:
        """Host-side equivalent of ``_build_prefill_chunk_metadata_kernel``."""
        def __init__(self):
            setattr(self, _DSV4_WIRED_ATTR, True)

        def __getitem__(self, grid):  # noqa: ARG002 -- mirrors upstream JIT-call pattern
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
                BLOCK_SIZE: int = -1,  # unused by this host implementation
                COMPRESS_RATIO: int,
            ) -> None:
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
                                base = (
                                    context // dcp_interleave // dcp_world
                                ) * dcp_interleave
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

                token_to_seq.copy_(token_cpu.to(device=token_to_seq.device))
                cu_seq_len_ks.copy_(ks_cpu.to(device=cu_seq_len_ks.device))
                cu_seq_len_ke.copy_(ke_cpu.to(device=cu_seq_len_ke.device))

            return launch

    setattr(indexer_mod, _INDEXER_KERNEL_ATTR, _CpuLaunchable())
    LOGGER.info("Installed indexer prefill-chunk metadata into %s", indexer_mod.__name__)
    return [f"{indexer_mod.__name__}.{_INDEXER_KERNEL_ATTR}"]


# C128A compressed top-K metadata (replaces the Triton kernel).
def _install_c128a_metadata(sparse_mla_mod: Any) -> List[str]:
    """Replace ``sparse_mla.build_c128a_topk_metadata`` with a host-side version.

    This produces two sets of gather indices used during V4 decode/prefill:
      * a per-decode-token block-table-resolved list of KV slots;
      * consecutive arange rows inside each prefill token's own workspace.
    The original is compiled to Triton because it reads irregular page-tables;
    here we perform the same loops over short (<128) ranges on CPU tensors and
    copy contiguous results back to XPU buffers.
    """
    fn = getattr(sparse_mla_mod, _C128A_FN_NAME, None)
    if getattr(fn, _DSV4_WIRED_ATTR, False):
        return []

    def build_c128a_topk_metadata(
        positions: torch.Tensor,
        compress_ratio: int,
        num_decode_tokens: int,
        token_to_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        slot_mapping: torch.Tensor,
        global_decode_buffer: torch.Tensor,
        decode_lens_buffer: torch.Tensor,
        prefill_buffer: torch.Tensor,
        max_compressed_tokens: int = 8192,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            num_compressed = min((position + 1) // compress_ratio, max_compressed_tokens)
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

    setattr(build_c128a_topk_metadata, _DSV4_WIRED_ATTR, True)
    sparse_mla_mod.build_c128a_topk_metadata = build_c128a_topk_metadata
    LOGGER.info("Installed C128A top-k metadata into %s", sparse_mla_mod.__name__)
    return [f"{sparse_mla_mod.__name__}.{_C128A_FN_NAME}"]


# SWA indices-and-lengths launcher shim.
def _install_swa_kernel(sparse_swa_mod: Any) -> List[str]:
    """Provide a CPU-loop fallback for ``sparse_swa._compute_swa_indices_and_lens_kernel``.

    The helper translates logical KV positions of every valid token into actual
    paged-cache slot numbers plus the number of visible tokens within the sliding
    window. We compute lengths/indexes offline from CPU copies of small 1-D tensors
    then copy them back to device so that downstream attention can use static
    buffer shapes safely under cudagraph capture.
    """
    if _is_already_wired(getattr(sparse_swa_mod, _SWA_KERNEL_ATTR, None)):
        return []

    class _CpuLaunchable:
        def __init__(self):
            setattr(self, _DSV4_WIRED_ATTR, True)

        def __getitem__(self, grid):  # noqa: ARG002
            def launch(
                swa_indices,
                swa_indices_stride,  # noqa: ARG001 -- present for signature parity
                swa_lens,
                window_size,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                block_table_stride,  # noqa: ARG001
                block_size,
                token_offset,
                *,
                TRITON_BLOCK_SIZE: int,  # noqa: ARG001,N806 -- kept to match caller kwargs
            ) -> None:
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
                        copy_width = len(values)
                        swa_indices[pid, :copy_width].copy_(
                            torch.tensor(values, dtype=swa_indices.dtype, device=swa_indices.device)
                        )
                swa_lens[:rows].copy_(lens_cpu.to(swa_lens.device))

            return launch

    setattr(sparse_swa_mod, _SWA_KERNEL_ATTR, _CpuLaunchable())
    LOGGER.info("Installed SWA indices+lengths kernel into %s", sparse_swa_mod.__name__)
    return [f"{sparse_swa_mod.__name__}.{_SWA_KERNEL_ATTR}"]


# Prefill gather-lens launcher shim.
def _install_prefill_gather_lenses(sparse_swa_mod: Any) -> List[str]:
    """CPU-loop replacement for ``sparse_swa._compute_prefill_metadata_kernel``.

    For each prefilled request this computes how many cached positions must be
    gathered together with its prompt length (i.e., ``query_len +
    min(prefix_len, window_size-1)``). The result lands directly in a device
    tensor consumed by FlashMLASparseAttention without per-token sync later.
    """
    if _is_already_wired(getattr(sparse_swa_mod, _PREFILL_LENS_ATTR, None)):
        return []

    class _CpuLaunchable:
        def __init__(self):
            setattr(self, _DSV4_WIRED_ATTR, True)

        def __getitem__(self, grid):  # noqa: ARG002
            def launch(
                prefill_gather_lens,
                seq_lens,
                query_start_loc,
                num_prefills,
                num_decodes,
                window_size,
                *,
                BLOCK_SIZE: int,  # noqa: ARG001,N806
            ) -> None:
                lengths = seq_lens.cpu().tolist()
                starts = query_start_loc.cpu().tolist()
                values = []
                for offset in range(num_prefills):
                    req_idx = num_decodes + offset
                    query_len = starts[req_idx + 1] - starts[req_idx]
                    prefix_len = lengths[req_idx] - query_len
                    values.append(query_len + min(prefix_len, window_size - 1))
                prefill_gather_lens.copy_(
                    torch.tensor(
                        values,
                        dtype=prefill_gather_lens.dtype,
                        device=prefill_gather_lens.device,
                    )
                )

            return launch

    setattr(sparse_swa_mod, _PREFILL_LENS_ATTR, _CpuLaunchable())
    LOGGER.info("Installed prefill-gather-lens kernel into %s", sparse_swa_mod.__name__)
    return [f"{sparse_swa_mod.__name__}.{_PREFILL_LENS_ATTR}"]


# ---------------------------------------------------------------------------
# Public entry point used by registry/installer
# ---------------------------------------------------------------------------
import torch


def _flashmla_metadata_predicate(mod) -> bool:
    fn = getattr(mod, "get_mla_metadata", None)
    return fn is not None and getattr(fn, "_kunlun_patched", False)


def _flashmla_metadata_applier(mod):
    """Replace get_mla_metadata plus main FlashMLA API symbols with Kunlun impl.

    Mirrors legacy root-hook behaviour; registered once per target namespace via
    patches.registry._STATIC_PATCHES so identical substitution lands whether or
    not eager-install block fires first.
    """
    from vllm_kunlun.ops.attention.flashmla import (
        get_mla_metadata as _kunlun_get,
        flash_mla_with_kvcache as _kunlun_flash_mla_with_kvcache,
        flash_mla_sparse_prefill as _kunlun_flash_mla_sparse_fwd,
    )

    def get_mla_metadata(cache_seqlens=None, num_heads_per_head_k=1,
                         num_heads_k=1):
        if cache_seqlens is None:
            empty = torch.empty(0, dtype=torch.int32)
            return empty, empty
        return _kunlun_get(cache_seqlens, num_heads_per_head_k, num_heads_k)

    get_mla_metadata._kunlun_patched = True
    mod.get_mla_metadata = get_mla_metadata
    mod.flash_mla_with_kvcache = _kunlun_flash_mla_with_kvcache
    mod.flash_mla_sparse_fwd = _kunlun_flash_mla_sparse_fwd


def _flashmla_padded_heads_predicate(mod) -> bool:
    cls = getattr(mod, "DeepseekV4FlashMLAAttention", None)
    if cls is None:
        return False
    return bool(getattr(cls, "_kunlun_no_pad", False))


def _flashmla_padded_heads_applier(mod):
    cls = getattr(mod, "DeepseekV4FlashMLAAttention", None)
    if cls is None:
        return

    @classmethod
    def _kunlun_get_padded_num_q_heads(cls_, num_heads: int) -> int:
        return num_heads

    cls.get_padded_num_q_heads = _kunlun_get_padded_num_q_heads
    cls._kunlun_no_pad = True
    LOGGER.info("Patched DeepseekV4FlashMLAAttention.get_padded_num_q_heads (no padding)")


def apply(master_enabled_check: bool = True) -> List[str]:  # type: ignore[name-defined]  -- returns labels via side effects
    """Register lazy hooks that install flashmla/sparse-attention metadata shims.

    All four hooks above are lightweight enough to fire lazily once their owning
    community modules are imported; they never call custom operators themselves,
    relying instead on deterministic Python CPU loops and explicit H2D copies.
    """
    if not master_enabled_check:
        return []

    from vllm_kunlun.config.deepseek_v4 import FeatureFlags

    flags = FeatureFlags()
    if not flags.flashmla_sparse_backend:
        WarningOnce.emit(
            "dsv4-flashmla-sparse-disabled",
            "KUNLUN_DSV4_FLASHMLA_SPARSE_BACKEND disabled; skipping sparse MLA metadata bridges",
        )
        return []

    _register_lazy(
        "vllm.v1.attention.backends.mla.indexer",
        lambda m: _is_already_wired(getattr(m, _INDEXER_KERNEL_ATTR, None)),
        _install_indexer_prefill_kernel,
    )
    _register_lazy(
        "vllm.models.deepseek_v4.sparse_mla",
        lambda m: getattr(getattr(m, _C128A_FN_NAME, None), _DSV4_WIRED_ATTR, False),
        _install_c128a_metadata,
    )
    _register_lazy(
        "vllm.v1.attention.backends.mla.sparse_swa",
        lambda m: _is_already_wired(getattr(m, _SWA_KERNEL_ATTR, None)),
        _install_swa_kernel,
    )
    _register_lazy(
        "vllm.v1.attention.backends.mla.sparse_swa",
        lambda m: _is_already_wired(getattr(m, _PREFILL_LENS_ATTR, None)),
        _install_prefill_gather_lenses,
    )
    LOGGER.debug("Registered DSV4 FlashMLA metadata lazy hooks")
    return []
