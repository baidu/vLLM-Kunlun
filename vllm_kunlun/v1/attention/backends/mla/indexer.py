# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from dataclasses import dataclass
from typing import ClassVar, Optional
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import ( CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.attention.backends.mla.indexer import (split_prefill_chunks,
                                                    DeepseekV32IndexerMetadataBuilder,
                                                    DeepseekV32IndexerPrefillMetadata)

logger = init_logger(__name__)

@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor


@dataclass
class DeepseekV32IndexerMetadata:

    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    # The dimension of the attention heads
    head_dim: int

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: Optional[DeepSeekV32IndexerDecodeMetadata] = None
    prefill: Optional[DeepseekV32IndexerPrefillMetadata] = None

def kunlun_build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> DeepseekV32IndexerMetadata:

        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold)

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            chunk_seq_ids = split_prefill_chunks(
                common_attn_metadata.seq_lens_cpu,
                self.max_prefill_buffer_size,
                num_decodes,
            )
            chunks = [
                self.build_one_prefill_chunk(
                    reqs_start, reqs_end, query_start_loc_cpu,
                    common_attn_metadata.seq_lens_cpu,
                    common_attn_metadata.block_table_tensor)
                for reqs_start, reqs_end in chunk_seq_ids
            ]
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(
                chunks=chunks, )

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(common_attn_metadata.query_start_loc[:num_decodes + 1],
                       out=self.decode_lens_buffer[:num_decodes])
            decode_lens = self.decode_lens_buffer[:num_decodes]
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[:num_decodes + 1])

            # Use CPU to avoid GPU sync; breaking async scheduling
            requires_padding = (decode_lens_cpu.max()
                                > decode_lens_cpu.min()).item()

            seq_lens = common_attn_metadata.seq_lens[:num_decodes]

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=common_attn_metadata.
                block_table_tensor[:num_decodes, ...],
                seq_lens=common_attn_metadata.seq_lens[:num_decodes],
                seq_lens_cpu=common_attn_metadata.seq_lens[:num_decodes].cpu(),
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=self.scheduler_metadata_buffer,
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            seq_lens_cpu=common_attn_metadata.seq_lens.cpu(),
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            head_dim=128,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        # if get_tensor_model_parallel_rank() == 0:
        #     logger.info(f"attn_metadata: {attn_metadata}")
        return attn_metadata

DeepseekV32IndexerMetadataBuilder.build = kunlun_build