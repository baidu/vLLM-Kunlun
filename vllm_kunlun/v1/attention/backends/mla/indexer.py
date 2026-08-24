# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch
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
    self, chunk: DeepseekV32IndexerPrefillChunkMetadata
) -> KunlunDeepseekV32IndexerPrefillChunkMetadata:
    if chunk.num_reqs > 1:
        raise NotImplementedError(
            "Kunlun I8_mqa_logits LOD is still one sequence per chunk "
            f"([0, total]); this chunk has num_reqs={chunk.num_reqs}."
        )

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
        context_q_lens=torch.tensor(
            [0, seq_len_q], dtype=torch.int32, device=self.device
        ),
        context_k_lens=torch.tensor(
            [0, seq_len_kv], dtype=torch.int32, device=self.device
        ),
        context_q_lens_cpu=torch.tensor(
            [0, seq_len_q], dtype=torch.int32, device="cpu"
        ),
        context_k_lens_cpu=torch.tensor(
            [0, seq_len_kv], dtype=torch.int32, device="cpu"
        ),
    )


def _adapt_decode_metadata(
    self,
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
                "Compressed indexer cache is not supported by Kunlun."
            )
        indexer_meta_data = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        if indexer_meta_data.prefill is not None:
            indexer_meta_data.prefill.chunks = [
                _adapt_prefill_chunk(self, chunk)
                for chunk in indexer_meta_data.prefill.chunks
            ]
        if indexer_meta_data.decode is not None:
            indexer_meta_data.decode = _adapt_decode_metadata(
                self,
                indexer_meta_data.decode,
                common_attn_metadata,
                indexer_meta_data.num_decodes,
            )

        return indexer_meta_data


class KunlunDeepseekV32IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_builder_cls() -> type["KunlunDeepseekV32IndexerMetadataBuilder"]:
        return KunlunDeepseekV32IndexerMetadataBuilder
