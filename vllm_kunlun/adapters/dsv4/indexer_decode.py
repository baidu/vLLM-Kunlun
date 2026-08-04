"""DeepSeek-V4 Lightning Indexer sparse-attention top-K selection adapter.

On CUDA the community backend dispatches ``forward_cuda`` through Triton/CUTLASS;
VLLM additionally defines a generic CPU reference. On Kunlun XPU neither GPU-only
path exists, but XPU custom operators cover the decode hot path:

* :obj:`torch.ops._C.I8_paged_mqa_logits`
* :obj:`torch.ops.xspeedgate_ops.topk_per_row`

Prefill scoring remains an explicit Python loop gathering K values/scales from
the paged compressed KV cache, matching the behaviour of the source-patched
bring-up branch.
"""
import logging
from typing import List

import torch

from ..runtime_utils import WarningOnce
from .registry import _register_lazy

LOGGER = logging.getLogger("vllm_kunlun.adapters.dsv4.indexer_decode")
_APPLIED_SENTINEL_KEY = "_dsv4_sparse_attn_indexer_applied"
_WARNED_DECODE_NATIVE_FAILED_KEY = "dsv4-indexer-decode-native-failed"
_FALSE = object()


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
                except Exception as exc:  # noqa: BLE001
                    # Log once and surface unfilled indices so attention fails fast
                    # instead of producing garbage tokens silently.
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

                    num_seqs = local_cu.shape[0] - 1
                    for seq_idx in range(num_seqs):
                        seq_k_s = int(local_cu[seq_idx].item())
                        seq_k_e = int(local_cu[seq_idx + 1].item())
                        seq_k_len = seq_k_e - seq_k_s
                        if seq_k_len <= 0:
                            continue

                        blocks_needed = (seq_k_len + block_size_cache - 1) // block_size_cache
                        bt_row = chunk_bt[seq_idx, :blocks_needed].long()
                        gathered_values: list[torch.Tensor] = []
                        gathered_scales: list[torch.Tensor] = []
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

                        k_int8_cat = torch.cat(gathered_values, dim=0).view(torch.int8)
                        k_f32_scale = torch.cat(gathered_scales, dim=0)
                        k_float_chunk = k_int8_cat.float() * k_f32_scale

                        q_s = int(cu_seqlen_ks[seq_idx].item())
                        q_e = int(cu_seqlen_ke[seq_idx].item())
                        for qt_pos_offset in range(q_e - q_s):
                            qt_global = token_start_local + q_s + qt_pos_offset
                            if qt_global < 0 or qt_global >= q_summed_p.shape[0]:
                                continue
                            causal_len = seq_k_len
                            if causal_len <= 0:
                                continue
                            scores = (
                                q_summed_p[qt_global:qt_global + 1]
                                @ k_float_chunk[:causal_len].T
                            ).squeeze(0)
                            actual_topk = min(topk_tokens, scores.shape[0])
                            if actual_topk <= 0:
                                continue
                            _, chosen_idx = scores.topk(actual_topk, dim=0, sorted=False)
                            buf_row = num_decode_tokens + qt_global
                            self.topk_indices_buffer[buf_row, :actual_topk] = chosen_idx.to(torch.int32)

            return self.topk_indices_buffer

    setattr(indexer_module, "KunlunSparseAttnIndexer", KunlunSparseAttnIndexer)
    setattr(indexer_module, _APPLIED_SENTINEL_KEY, True)
    LOGGER.info("Registered %r into OOT dispatcher from %s", "SparseAttnIndexer", indexer_module.__name__)
    return [f"{indexer_module.__name__}.KunlunSparseAttnIndexer(SparseAttnIndexer OOT)"]


# ---------------------------------------------------------------------------
def _applied(mod: object) -> bool:
    return bool(getattr(mod, _APPLIED_SENTINEL_KEY, False))


def apply(master_enabled_check: bool = True) -> List[str]:
    """Install lazy hook registering the Kunlun sparse-lightning-indexer backend."""
    if not master_enabled_check:
        return []

    from .gates import FeatureFlags

    flags = FeatureFlags()
    if not flags.indexer_decode_native:
        WarningOnce.emit(
            "dsv4-indexer-decode-disabled",
            "KUNLUN_DSV4_INDEXER_DECODE_NATIVE disabled; skipping sparse-lightning-indexer override",
        )
        return []

    _register_lazy(
        "vllm.model_executor.layers.sparse_attn_indexer",
        _applied,
        _install_kunlun_indexer,
    )
    return []
