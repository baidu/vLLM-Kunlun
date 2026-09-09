# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import logging
import torch
import vllm.v1.attention.backends.mla.indexer as mla_indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV32IndexerPrefillChunkMetadata,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
)


def fill_prefill_chunk_meta_torch(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    cu_seq_len_ks: torch.Tensor,
    cu_seq_len_ke: torch.Tensor,
    query_slice_start: int,
    query_slice_stop: int,
):
    device = query_start_loc.device
    num_requests = seq_lens.shape[0]

    query_starts = query_start_loc[:-1]
    query_lens = query_start_loc[1:] - query_starts
    seq_starts = cu_seq_lens[:-1]
    start_pos = seq_lens - query_lens

    token_to_seq[:] = torch.repeat_interleave(
        torch.arange(num_requests, device=device, dtype=torch.int32),
        seq_lens,
        output_size=token_to_seq.numel(),
    )

    abs_pos = torch.arange(
        query_slice_start,
        query_slice_stop,
        device=device,
        dtype=torch.int32,
    )
    request_idx = torch.searchsorted(query_starts, abs_pos, right=True) - 1
    query_offset = abs_pos - query_starts[request_idx]
    cu_seq_len_ks[:] = seq_starts[request_idx]
    cu_seq_len_ke[:] = (
        seq_starts[request_idx] + start_pos[request_idx] + 1 + query_offset
    )


class KunlunBuildPrefillChunkMetaKernel:
    """Duck-type the upstream @triton.jit kernel."""

    def __getitem__(self, grid):
        return self

    def warmup(self, *args, **kwargs):
        return

    def __call__(
        self,
        query_start_loc,
        uncompressed_seq_lens,
        cu_compressed_seq_lens,
        row_start_cu_compressed_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        query_slice_start,
        query_slice_stop,
        DCP_RANK,
        DCP_WORLD,
        DCP_INTERLEAVE,
        *,
        BLOCK_SIZE,
        COMPRESS_RATIO,
    ) -> None:
        if DCP_WORLD != 1:
            raise NotImplementedError("Kunlun indexer metadata: DCP not supported")
        if COMPRESS_RATIO != 1:
            raise NotImplementedError(
                "Kunlun indexer metadata: compression not supported"
            )
        fill_prefill_chunk_meta_torch(
            query_start_loc,
            uncompressed_seq_lens,
            cu_compressed_seq_lens,
            token_to_seq,
            cu_seq_len_ks,
            cu_seq_len_ke,
            query_slice_start,
            query_slice_stop,
        )


def patch_prefill_chunk_metadata_kernel() -> None:
    kernel = mla_indexer._build_prefill_chunk_metadata_kernel
    if getattr(kernel, "_kunlun_patched", False):
        return
    logging.getLogger("vllm_kunlun").info(
        "[KunlunPlugin] patched _build_prefill_chunk_metadata_kernel"
    )
    replacement = KunlunBuildPrefillChunkMetaKernel()
    replacement._kunlun_patched = True
    mla_indexer._build_prefill_chunk_metadata_kernel = replacement


@dataclass(kw_only=True)
class KunlunDeepseekV32IndexerPrefillChunkMetadata(
    DeepseekV32IndexerPrefillChunkMetadata
):
    context_q_lens: torch.Tensor
    context_q_lens_cpu: torch.Tensor
    context_k_lens: torch.Tensor
    context_k_lens_cpu: torch.Tensor


@dataclass(kw_only=True)
class KunlunDeepSeekV32IndexerDecodeMetadata(DeepSeekV32IndexerDecodeMetadata):
    # Request-level final sequence lengths, shape [num_decodes], not a CPU mirror of
    # the inherited decode.seq_lens.
    seq_lens_cpu: torch.Tensor


def _adapt_prefill_chunk(
    chunk: DeepseekV32IndexerPrefillChunkMetadata,
    device: torch.device,
) -> KunlunDeepseekV32IndexerPrefillChunkMetadata:
    seq_len_q = chunk.token_end - chunk.token_start
    seq_len_kv = chunk.total_seq_lens

    return KunlunDeepseekV32IndexerPrefillChunkMetadata(
        block_table=chunk.block_table,
        cu_seqlen_ks=chunk.cu_seqlen_ks,
        cu_seqlen_ke=chunk.cu_seqlen_ke,
        cu_seq_lens=chunk.cu_seq_lens,
        token_to_seq=chunk.token_to_seq,
        total_seq_lens=chunk.total_seq_lens,
        token_start=chunk.token_start,
        token_end=chunk.token_end,
        num_reqs=chunk.num_reqs,
        skip_kv_gather=chunk.skip_kv_gather,
        local_cu_seq_lens=chunk.local_cu_seq_lens,
        local_total_seq_lens=chunk.local_total_seq_lens,
        max_local_total_seq_lens=chunk.max_local_total_seq_lens,
        context_q_lens=torch.tensor([0, seq_len_q], dtype=torch.int32, device=device),
        context_k_lens=torch.tensor([0, seq_len_kv], dtype=torch.int32, device=device),
        context_q_lens_cpu=torch.tensor(
            [0, seq_len_q], dtype=torch.int32, device="cpu"
        ),
        context_k_lens_cpu=torch.tensor(
            [0, seq_len_kv], dtype=torch.int32, device="cpu"
        ),
    )


def _adapt_decode_metadata(
    decode_metadata: DeepSeekV32IndexerDecodeMetadata,
    common_attn_metadata: CommonAttentionMetadata,
    num_decodes: int,
) -> KunlunDeepSeekV32IndexerDecodeMetadata:
    return KunlunDeepSeekV32IndexerDecodeMetadata(
        block_table=decode_metadata.block_table,
        seq_lens=decode_metadata.seq_lens,
        seq_lens_cpu=common_attn_metadata.seq_lens_cpu[:num_decodes],
        decode_lens=decode_metadata.decode_lens,
        requires_padding=decode_metadata.requires_padding,
        schedule_metadata=decode_metadata.schedule_metadata,
        global_seq_lens=decode_metadata.global_seq_lens,
    )


class KunlunDeepseekV32IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        # Do not support DCP for Kunlun sparse indexer, as it is not implemented yet.
        if self.dcp_world_size != 1:
            raise NotImplementedError("DCP is not supported by Kunlun sparse indexer.")
        if self.compress_ratio != 1:
            raise NotImplementedError(
                "Compressed indexer cache is not supported by Kunlun sparse indexer."
            )
        if self.use_flattening:
            raise NotImplementedError(
                "Kunlun sparse indexer supports at most 1 speculative token "
                f"(next_n <= 2), got num_speculative_tokens="
                f"{self.num_speculative_tokens}."
            )
        indexer_meta_data = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        if indexer_meta_data.prefill is not None:
            indexer_meta_data.prefill.chunks = [
                _adapt_prefill_chunk(chunk, self.device)
                for chunk in indexer_meta_data.prefill.chunks
            ]
        if indexer_meta_data.decode is not None:
            indexer_meta_data.decode = _adapt_decode_metadata(
                indexer_meta_data.decode,
                common_attn_metadata,
                indexer_meta_data.num_decodes,
            )

        return indexer_meta_data


class KunlunDeepseekV32IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_builder_cls() -> type["KunlunDeepseekV32IndexerMetadataBuilder"]:
        return KunlunDeepseekV32IndexerMetadataBuilder
