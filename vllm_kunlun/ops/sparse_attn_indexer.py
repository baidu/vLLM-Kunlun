import torch

from vllm.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

from vllm_kunlun.ops.deep_gemm import int8_mqa_logits, int8_paged_mqa_logits
from vllm_kunlun.v1.attention.backends.mla.indexer import (
    KunlunDeepSeekV32IndexerDecodeMetadata,
    KunlunDeepseekV32IndexerPrefillChunkMetadata,
)


@CustomOp.register_oot(name="SparseAttnIndexer")
class KunlunSparseAttnIndexer(SparseAttnIndexer):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        block_size = self.k_cache.cache_config.block_size
        if self.max_model_len % block_size != 0:
            self.max_model_len += block_size - (self.max_model_len % block_size)

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if self.use_fp4_cache:
            raise NotImplementedError(
                "Kunlun SparseAttnIndexer does not support FP4 cache."
            )

        if isinstance(q_quant, tuple):
            raise NotImplementedError(
                "Kunlun SparseAttnIndexer does not support FP4 q_quant."
            )

        if q_quant.dtype != torch.int8:
            raise NotImplementedError(
                "Kunlun SparseAttnIndexer kernels expect int8 Q "
                f"(got {q_quant.dtype}). Upstream Indexer emits FP8 and "
                "folds q_scale into weights; do not bitcast or requantize "
                "here. Either keep Indexer-side int8 quant2d, or add FP8 "
                "kernels."
            )

        # careful! this will be None in dummy run
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return self.topk_indices_buffer

        metadata = attn_metadata[self.k_cache.prefix]
        assert isinstance(metadata, DeepseekV32IndexerMetadata)

        kv_cache = self.k_cache.kv_cache
        if isinstance(kv_cache, (list, tuple)):
            kv_cache = kv_cache[0]

        slot_mapping = metadata.slot_mapping
        if k is not None:
            k = k[: slot_mapping.shape[0]]

        if not self.skip_k_cache_insert:
            torch.ops.xspeedgate_ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                self.quant_block_size,
                self.scale_fmt,
            )

        self.topk_indices_buffer[: hidden_states.shape[0]] = -1

        if metadata.num_prefills > 0:
            self._prefill(q_quant, weights, kv_cache, metadata)
        if metadata.num_decodes > 0:
            self._decode(q_quant, weights, kv_cache, metadata)

        return self.topk_indices_buffer

    def _prefill(self, q_quant, weights, kv_cache, metadata):
        prefill_metadata = metadata.prefill
        assert prefill_metadata is not None
        for chunk in prefill_metadata.chunks:
            if not isinstance(chunk, KunlunDeepseekV32IndexerPrefillChunkMetadata):
                raise TypeError(
                    "Kunlun SparseAttnIndexer prefill requires "
                    "KunlunDeepseekV32IndexerPrefillChunkMetadata "
                    f"(got {type(chunk).__name__}). Builder/Backend is not wired."
                )

            k_fp8 = torch.empty(
                [chunk.total_seq_lens, self.head_dim],
                device=kv_cache.device,
                dtype=torch.int8,
            )
            k_scale = torch.empty(
                [chunk.total_seq_lens, 4], device=kv_cache.device, dtype=torch.uint8
            )
            torch.ops.xspeedgate_ops.cp_gather_indexer_k_quant_cache(
                kv_cache=kv_cache,
                dst_k=k_fp8,
                dst_scale=k_scale,
                block_table=chunk.block_table,
                cu_seq_lens=chunk.cu_seq_lens,
            )
            logits = int8_mqa_logits(
                q_quant[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                context_q_lens_xpu=chunk.context_q_lens,
                context_q_lens_cpu=chunk.context_q_lens_cpu,
                context_k_lens_xpu=chunk.context_k_lens,
                context_k_lens_cpu=chunk.context_k_lens_cpu,
            )
            del k_fp8, k_scale

            num_rows = logits.shape[0]
            topk_indices = self.topk_indices_buffer[
                chunk.token_start : chunk.token_end, : self.topk_tokens
            ]

            # when seqLens=None and next_n=None, it means that it is used to calculate topk_indices in prefill
            # refer to top_k_per_row_prefill：https://github.com/vllm-project/vllm/blob/6a09612b2e0e09d037a220ea8115632b8084e008/csrc/sampler.cu#L698
            torch.ops.xspeedgate_ops.topk_per_row(
                logits=logits,
                srcIndices=topk_indices,
                numRows=num_rows,
                stride0=logits.stride(0),
                stride1=logits.stride(1),
                topK=self.topk_tokens,
                rowStarts=chunk.cu_seqlen_ks,
                rowEnds=chunk.cu_seqlen_ke,
                seqLens=None,
                next_n=None,
            )

    def _decode(self, q_quant, weights, kv_cache, metadata):
        decode_metadata = metadata.decode
        assert decode_metadata is not None
        if not isinstance(decode_metadata, KunlunDeepSeekV32IndexerDecodeMetadata):
            raise TypeError(
                "Kunlun SparseAttnIndexer decode requires "
                "KunlunDeepSeekV32IndexerDecodeMetadata "
                f"(got {type(decode_metadata).__name__}). Builder/Backend is not wired."
            )

        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        if kv_cache.ndim == 3:
            kv_cache = kv_cache.unsqueeze(-2)

        num_decode_tokens = metadata.num_decode_tokens
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_quant[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )

        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n

        # deal with 2D seq_lens
        # Kunlun I8_paged_mqa_logits and topk_per_row take 1D seq_lens [B], matching
        # the old metadata. Upstream v0.25.1 may pass (B, 1) for plain decode; squeeze
        # it so it matches request-level seq_lens_cpu. A (B, next_n) tensor with
        # next_n > 1 is the new native-MTP layout and is not wired to the old kernel;
        # fail the asserts below instead of silently treating it as 1D.
        seq_lens = decode_metadata.seq_lens
        seq_lens_cpu = decode_metadata.seq_lens_cpu
        if seq_lens.ndim == 2 and seq_lens.shape[-1] == 1:
            seq_lens = seq_lens.squeeze(-1)
        assert seq_lens.shape[0] == batch_size
        assert seq_lens_cpu.shape == seq_lens.shape

        logits = int8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            seq_lens,
            seq_lens_cpu,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=self.max_model_len,
        )

        num_rows = logits.shape[0]
        topk_indices = self.topk_indices_buffer[:num_padded_tokens, : self.topk_tokens]

        # when row_starts=None and row_ends=None, it means that it is used to calculate topk_indices in decode
        # refer to top_k_per_row_decode：https://github.com/vllm-project/vllm/blob/6a09612b2e0e09d037a220ea8115632b8084e008/csrc/sampler.cu#L643
        torch.ops.xspeedgate_ops.topk_per_row(
            logits=logits,
            srcIndices=topk_indices,
            numRows=num_rows,
            stride0=logits.stride(0),
            stride1=logits.stride(1),
            topK=self.topk_tokens,
            rowStarts=None,
            rowEnds=None,
            seqLens=seq_lens,
            next_n=next_n,
        )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            self.topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )
