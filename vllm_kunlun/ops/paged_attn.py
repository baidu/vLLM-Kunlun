# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from dataclasses import dataclass
from typing import List, Optional, Tuple
from vllm.platforms import current_platform


from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops


# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length in the batch. 0 if it is prefill-only batch.
    max_decode_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]


class PagedAttention:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 80, 96, 112, 120, 128, 192, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        """
            获取KV缓存的形状，根据是否在芯片上进行计算返回不同的形状。
        如果在芯片上（is_kunlun()为True），则返回形状(2, num_blocks, num_kv_heads, block_size, head_size)；
        否则，返回形状(2, num_blocks, block_size * num_kv_heads * head_size)。
        
        Args:
            num_blocks (int): 块数量。
            block_size (int): 每个块大小。
            num_kv_heads (int): KV头数量。
            head_size (int): 每个头大小。
        
        Returns:
            Tuple[int, ...]: KV缓存的形状，包括两个元素：第一个元素为2，表示维度数量为2；第二个元素为num_blocks、num_kv_heads、block_size和head_size中的任意一个。
        """
        if current_platform.is_kunlun():
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            将一个缓存张量（包含key和value）分成两部分，每个部分是一个张量。
        如果在KUNLUN上运行，则返回的第一个张量是key缓存，第二个张量是value缓存。
        否则，第一个张量是key缓存，第二个张量是key缓存的view，其形状为(num_blocks, num_kv_heads, head_size//x, -1, x)，
        第三个张量是value缓存，其形状为(num_blocks, num_kv_heads, head_size, -1)。
        
        Args:
            kv_cache (torch.Tensor): 包含key和value的张量，形状为(2, num_blocks, kv_cache_size)。
            num_kv_heads (int): 多头注意力中的头数。
            head_size (int): 每个头的大小。
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - key_cache (torch.Tensor): 形状为(num_blocks, num_kv_heads, head_size//x, -1, x)，包含key缓存。
                - value_cache (torch.Tensor): 形状为(num_blocks, num_kv_heads, head_size, -1)，包含value缓存。
        """
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        if current_platform.is_kunlun():
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
        else:
            key_cache = kv_cache[0]
            key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                    -1, x)
            value_cache = kv_cache[1]
            value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        context_lens_cpu: torch.Tensor,
        is_context,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> torch.Tensor:
        if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
            # use blocksparse paged attention
            block_size = value_cache.size(-1)
            assert (blocksparse_block_size > 0 and
                    blocksparse_block_size % block_size == 0), \
                (f"{blocksparse_block_size=} needs to be a multiple of"
                 f"{block_size=} used in block_tables.")

        output = torch.empty_like(query)
        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                              _PARTITION_SIZE)
        # NOTE(woosuk): We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # TODO(woosuk): Tune this heuristic.
        # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        use_v1 = (max_seq_len <= 8192
                  and (max_num_partitions == 1 or num_seqs * num_heads > 512))
        
        if use_v1:
            # Run PagedAttention V1.
            ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                context_lens_cpu,
                is_context,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
                tp_rank,
                blocksparse_local_blocks,
                blocksparse_vert_stride,
                blocksparse_block_size,
                blocksparse_head_sliding_step,
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                context_lens_cpu,
                is_context,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
                tp_rank,
                blocksparse_local_blocks,
                blocksparse_vert_stride,
                blocksparse_block_size,
                blocksparse_head_sliding_step,
            )
        return output

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_dtype: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        max_query_len: int,
        alibi_slopes: Optional[torch.Tensor],
        sliding_window: Optional[int],
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        max_seq_len = None
        context_attention_fwd(
            query,
            key,
            value,
            output,
            kv_cache_dtype,
            key_cache,
            value_cache,
            block_tables,
            # query_start_loc is (batch_size + 1,)
            query_start_loc,
            seq_lens_tensor,
            max_seq_len,
            max_query_len,
            k_scale,
            v_scale,
            alibi_slopes,
            sliding_window,
        )
        return output

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)