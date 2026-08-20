"""DeepSeek-V4 Lightning Indexer sparse top-K selection on Kunlun XPU.

Decode scoring uses the XPU kernels ``torch.ops._C.I8_paged_mqa_logits`` and
``xspeedgate_ops.topk_per_row``; prefill scoring is a Python loop over the
paged compressed KV cache.
"""
import logging
import os
from typing import List

import torch

from vllm_kunlun.runtime_utils import WarningOnce

LOGGER = logging.getLogger("vllm_kunlun.ops.attention.indexer_decode")
_APPLIED_SENTINEL_KEY = "_dsv4_sparse_attn_indexer_applied"
_WARNED_DECODE_NATIVE_FAILED_KEY = "dsv4-indexer-decode-native-failed"
_FALSE = object()

# The native topk_per_row kernel sizes its output with a hard-coded TOPK=2048
# instead of the runtime topK argument, so with index_topk=512 it writes 6 KiB
# past each row and corrupts neighbouring rows of topk_indices_buffer. Default
# to the torch path; set KUNLUN_DSV4_INDEXER_TOPK_NATIVE=1 to use the kernel.
_TOPK_NATIVE = os.environ.get("KUNLUN_DSV4_INDEXER_TOPK_NATIVE", "0") == "1"
_WARNED_TOPK_TORCH_KEY = "dsv4-indexer-topk-torch"


def _decode_topk_torch(
    logits_flat: torch.Tensor,          # [rows, max_model_len], fp32
    seq_lens: torch.Tensor,             # [batch_size] int32
    batch_size: int,
    next_n: int,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,  # [max_tokens, topk_tokens] int32
) -> None:
    """Capture-safe torch replacement for ``xspeedgate_ops.topk_per_row``.

    Mirrors vLLM's ``ops.top_k_per_row_decode`` semantics: row ``r`` carries
    query position ``r % next_n`` and may attend columns
    ``[0, seq_len - next_n + (r % next_n) + 1)``; slots past the valid count
    keep the -1 sentinel the caller pre-filled. All shapes are static and no
    D2H sync is involved, so this is safe inside a cudagraph capture.
    """
    rows = batch_size * next_n
    device = logits_flat.device
    num_cols = logits_flat.shape[1]
    width = min(topk_tokens, num_cols)

    offsets = torch.arange(next_n, device=device, dtype=torch.int32).view(1, next_n)
    row_ends = (
        seq_lens[:batch_size].to(torch.int32).view(batch_size, 1) - next_n + offsets + 1
    ).reshape(rows, 1)

    cols = torch.arange(num_cols, device=device, dtype=torch.int32).view(1, num_cols)
    scores = logits_flat.masked_fill(cols >= row_ends, float("-inf"))
    chosen = scores.topk(width, dim=1).indices.to(torch.int32)

    ranks = torch.arange(width, device=device, dtype=torch.int32).view(1, width)
    keep = ranks < row_ends.clamp(min=0)
    topk_indices_buffer[:rows, :width] = torch.where(
        keep, chosen, torch.full_like(chosen, -1)
    )


def _install_kunlun_indexer(indexer_module: object) -> List[str]:
    """Register ``SparseAttnIndexer`` OOT subclass backed by Kunlun/XPU kernels."""
    from vllm.model_executor.custom_op import CustomOp
    from vllm.forward_context import get_forward_context

    LOGGER.info("Installing DSV4 sparse-lightning-indexer OOT subclass into %s", indexer_module.__name__)

    @CustomOp.register_oot(name="SparseAttnIndexer")
    class KunlunSparseAttnIndexer(indexer_module.SparseAttnIndexer):
        def forward_oot(self, hidden_states, q_quant, k, weights):
            """Kunlun-compatible score/top-K for lightning-compressed attention indices."""
            del k  # written to page-table KV cache earlier by the compressor

            num_tokens = hidden_states.shape[0]
            topk_tokens = self.topk_tokens
            device = hidden_states.device

            self.topk_indices_buffer[:num_tokens] = -1

            forward_ctx = get_forward_context()
            attn_metadata = getattr(forward_ctx, "attn_metadata", None)
            if not isinstance(attn_metadata, dict):
                # Warmup or tracing without real metadata -- keep sentinel values.
                return self.topk_indices_buffer

            k_prefix_name = self.k_cache.prefix
            indexer_meta = attn_metadata.get(k_prefix_name)
            if indexer_meta is None:
                return self.topk_indices_buffer

            kv_cache = self.k_cache.kv_cache
            if kv_cache.numel() == 0:
                return self.topk_indices_buffer

            num_blocks_cache = kv_cache.shape[0]
            block_size_cache = kv_cache.shape[1]
            head_dim_bytes = kv_cache.shape[-1]
            head_dim = self.head_dim
            num_heads_qkv = q_quant.shape[1]

            num_decode_tokens = indexer_meta.num_decode_tokens
            has_prefill = indexer_meta.num_prefills > 0

            # ---- Decode path -------------------------------------------------
            if (
                num_decode_tokens > 0
                and indexer_meta.decode is not None
            ):
                decode_meta = indexer_meta.decode
                seq_lens_raw = decode_meta.seq_lens
                block_table = decode_meta.block_table
                effective_seq_lens = (
                    seq_lens_raw[:, -1].contiguous()
                    if seq_lens_raw.dim() == 2
                    else seq_lens_raw.contiguous()
                )
                batch_size = effective_seq_lens.shape[0]
                next_n = num_decode_tokens // batch_size
                max_model_len = self.max_model_len

                q_view = q_quant[:num_decode_tokens].view(torch.int8)
                q_4d = q_view.reshape(batch_size, next_n, q_view.shape[1], q_view.shape[2])
                w_3d = weights[:num_decode_tokens].reshape(batch_size, next_n, -1)

                kv_flat = kv_cache.view(num_blocks_cache, -1)
                k_val = (
                    kv_flat[:, :block_size_cache * head_dim]
                    .view(torch.int8)
                    .view(num_blocks_cache, block_size_cache, 1, head_dim)
                )

                block_indices = block_table.flatten().long()
                k_scale_all = (
                    kv_flat[block_indices, block_size_cache * head_dim:]
                    .view(-1, 4)
                    .view(torch.float32)
                )
                k_scale = k_scale_all.view(batch_size, -1)[:, :max_model_len]

                logits_out = torch.empty(
                    (batch_size, next_n, max_model_len),
                    dtype=torch.float32,
                    device=device,
                )

                try:
                    torch.ops._C.I8_paged_mqa_logits(
                        q=q_4d,
                        fused_kv_cache=[k_val, k_scale],
                        weights=w_3d,
                        context_lens=[effective_seq_lens.cpu(), effective_seq_lens],
                        block_table=block_table,
                        max_context_len=max_model_len,
                        clean_logits=True,
                        out=logits_out,
                        use_xfa_boost=False,
                    )

                    logits_flat = logits_out.reshape(-1, max_model_len)
                    if _TOPK_NATIVE:
                        topk_input_buf = self.topk_indices_buffer[
                            :batch_size * next_n, :topk_tokens
                        ]
                        torch.ops.xspeedgate_ops.topk_per_row(
                            logits=logits_flat,
                            srcIndices=topk_input_buf,
                            numRows=batch_size * next_n,
                            stride0=logits_flat.stride(0),
                            stride1=1,
                            topK=topk_tokens,
                            rowStarts=None,
                            rowEnds=None,
                            seqLens=effective_seq_lens,
                            next_n=next_n,
                        )
                    else:
                        WarningOnce.emit(
                            _WARNED_TOPK_TORCH_KEY,
                            "Using torch decode top-K; the native topk_per_row "
                            "kernel writes 2048 indices per row regardless of "
                            "topK=%d and corrupts topk_indices_buffer. Set "
                            "KUNLUN_DSV4_INDEXER_TOPK_NATIVE=1 to override.",
                            topk_tokens,
                        )
                        _decode_topk_torch(
                            logits_flat,
                            effective_seq_lens,
                            batch_size,
                            next_n,
                            topk_tokens,
                            self.topk_indices_buffer,
                        )
                except Exception as exc:  # noqa: BLE001
                    # Keep the -1 sentinels so attention fails fast instead of
                    # silently producing garbage tokens.
                    WarningOnce.emit(
                        _WARNED_DECODE_NATIVE_FAILED_KEY,
                        "Native sparse-indexer decode kernel failed (%s); "
                        "attention layer will receive invalid indices",
                        str(exc),
                    )

            # ---- Prefill path -------------------------------------------------
            if has_prefill and indexer_meta.prefill is not None:
                prefill_meta = indexer_meta.prefill
                q_prefill = q_quant[num_decode_tokens:num_tokens]
                w_prefill = weights[num_decode_tokens:num_tokens]

                q_float_p = q_prefill.view(torch.int8).float()
                q_weighted_p = q_float_p * w_prefill.unsqueeze(-1)
                q_summed_p = q_weighted_p.sum(dim=1)

                for chunk in prefill_meta.chunks:
                    token_start_local = chunk.token_start - num_decode_tokens
                    token_end_local = chunk.token_end - num_decode_tokens
                    cu_seqlen_ks = chunk.cu_seqlen_ks
                    cu_seqlen_ke = chunk.cu_seqlen_ke
                    local_cu = chunk.local_cu_seq_lens
                    chunk_bt = chunk.block_table

                    total_seq_len_chunk = getattr(chunk, "local_total_seq_lens", 0)
                    if total_seq_len_chunk == 0 or local_cu is None:
                        continue

                    # cu_seqlen_ks/ke are per *query token*: row t is valid on
                    # columns [ks[t], ke[t]) of the chunk's compressed K, and
                    # ke-ks is the entry count combine_topk_swa_indices reads
                    # back. Treat them as per-sequence bounds and the requests'
                    # slots shift into each other's workspace.
                    num_seqs = local_cu.shape[0] - 1
                    gathered_values: list[torch.Tensor] = []
                    gathered_scales: list[torch.Tensor] = []
                    for seq_idx in range(num_seqs):
                        seq_k_s = int(local_cu[seq_idx].item())
                        seq_k_e = int(local_cu[seq_idx + 1].item())
                        seq_k_len = seq_k_e - seq_k_s
                        if seq_k_len <= 0:
                            continue

                        blocks_needed = (seq_k_len + block_size_cache - 1) // block_size_cache
                        bt_row = chunk_bt[seq_idx, :blocks_needed].long()
                        collected_slots = 0
                        for blk_ref in bt_row.unbind(0):
                            blk_id = int(blk_ref.item())
                            remaining = seq_k_len - collected_slots
                            take = min(block_size_cache, remaining)
                            slot_data = kv_cache[
                                max(0, min(blk_id, num_blocks_cache - 1)), :take, :
                            ]
                            gathered_values.append(slot_data[:, :head_dim])
                            raw_scale = (
                                slot_data[:, head_dim:head_dim + 4]
                                .contiguous()
                                .view(torch.float32)
                            )
                            gathered_scales.append(raw_scale[:take])
                            collected_slots += take

                    if not gathered_values:
                        continue

                    k_int8_cat = torch.cat(gathered_values, dim=0).view(torch.int8)
                    k_f32_scale = torch.cat(gathered_scales, dim=0)
                    k_float_chunk = k_int8_cat.float() * k_f32_scale
                    total_k = k_float_chunk.shape[0]

                    q_chunk = q_summed_p[token_start_local:token_end_local]
                    num_q = q_chunk.shape[0]
                    if num_q == 0 or total_k == 0:
                        continue
                    ks = cu_seqlen_ks[:num_q].long()
                    ke = cu_seqlen_ke[:num_q].long()

                    scores = q_chunk @ k_float_chunk.T
                    columns = torch.arange(total_k, device=device).unsqueeze(0)
                    scores = scores.masked_fill(
                        (columns < ks.unsqueeze(1)) | (columns >= ke.unsqueeze(1)),
                        float("-inf"),
                    )

                    width = min(topk_tokens, total_k)
                    _, chosen = scores.topk(width, dim=1, sorted=True)
                    # Indices are handed back request-local: the sparse-attention
                    # consumer adds the request's own workspace offset, and ks[t] is
                    # that request's first compressed column.
                    local_idx = (chosen - ks.unsqueeze(1)).to(torch.int32)
                    counts = (ke - ks).clamp(min=0, max=width).unsqueeze(1)
                    keep = torch.arange(width, device=device).unsqueeze(0) < counts
                    self.topk_indices_buffer[
                        num_decode_tokens + token_start_local:
                        num_decode_tokens + token_end_local,
                        :width,
                    ] = torch.where(keep, local_idx, torch.full_like(local_idx, -1))

            return self.topk_indices_buffer

    setattr(indexer_module, "KunlunSparseAttnIndexer", KunlunSparseAttnIndexer)
    setattr(indexer_module, _APPLIED_SENTINEL_KEY, True)
    LOGGER.info("Registered %r into OOT dispatcher from %s", "SparseAttnIndexer", indexer_module.__name__)
    return [f"{indexer_module.__name__}.KunlunSparseAttnIndexer(SparseAttnIndexer OOT)"]


def _applied(mod: object) -> bool:
    return bool(getattr(mod, _APPLIED_SENTINEL_KEY, False))
