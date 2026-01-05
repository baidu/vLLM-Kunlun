# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionLayer,
                                              AttentionMetadata)
from vllm.attention.backends.utils import get_mla_dims
from vllm_kunlun.ops.attention.flashmla import (flash_mla_sparse_prefill,
                                         flash_mla_with_kvcache,
                                         get_mla_metadata,
                                         kunlun_flash_mla_with_kvcache)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import cdiv
from vllm.v1.attention.backends.mla.common import MLACommonBaseImpl
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              reshape_attn_output_for_spec_decode,
                                              reshape_query_for_spec_decode,
                                              split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.distributed import get_tp_group

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)
"""
NOTE: FlashMLA Sparse uses an fp8 cache with the following format

In the "FP8 with scale" format, each token's KV cache is 656 Bytes, 
structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512 
    `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values. 
    The first `float32` is the scale for the first 128 `float8_e4m3` values, 
    the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This 
    part is not quantized for accuracy.
"""


def _lse2_to_lse(lse_base2: torch.Tensor) -> torch.Tensor:
    # Convert base-2 LSE to natural-log LSE
    # Keep FP32 for numerical stability during the merge.
    return (lse_base2.to(torch.float32) * math.log(2.0))


class FlashMLASparseBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return FlashMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashMLASparseMetadataBuilder"]:
        return FlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLASparseImpl"]:
        return FlashMLASparseImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # custom storage fromat is 656 bytes
            #  see FlashMLA readme.md for details
            return (num_blocks, block_size, 656)
        else:
            return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]


@dataclass
class MLASparsePrefillMetadata:
    # NOTE(Chen): not call it "FlashMLASparsePrefillMetadata" because
    # the kernel is not from flashmla
    block_table: torch.Tensor = None
    has_context: bool = False
    context_lens: Optional[torch.Tensor] = None

    # Sequence lengths (context + query) for prefill requests
    # Shape: [num_prefill_reqs]
    seq_lens: torch.Tensor = None

    # Request ID for each token: -1 for decode tokens, request index
    # (0, 1, 2, ...) for prefill tokens.
    # Shape: [num_actual_tokens]
    request_ids: torch.Tensor = None
    query_start_loc: torch.Tensor = None
    query_start_loc_cpu: torch.Tensor = None

@dataclass
class FlashMLASparseDecodeAndContextMetadata:
    scheduler_metadata: torch.Tensor = None
    num_splits: torch.Tensor = None
    cache_lens: torch.Tensor = None
    prefill_context_lengths: Optional[torch.Tensor] = None
    prefill_new_k_start_locs: Optional[torch.Tensor] = None
    dummy_block_table: torch.Tensor = None

    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    max_seq_len: int = -1 # needed for reshape in spec decode

    def filter_prefill_indices(
            self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.prefill_context_lengths is not None
        prefill_context_lengths = self.prefill_context_lengths.unsqueeze(-1)
        context_indices = torch.where(indices < prefill_context_lengths,
                                      indices, -1)
        new_token_indices = torch.where(indices >= prefill_context_lengths,
                                        indices - prefill_context_lengths, -1)
        return context_indices, new_token_indices


@dataclass
class FlashMLASparseMetadata:
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048

    num_prefills: int = 0
    num_decodes: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    decode_metadata: Optional[FlashMLASparseDecodeAndContextMetadata] = None
    prefill_metadata: Optional[MLASparsePrefillMetadata] = None

    @dataclass
    class FP8KernelMetadata:
        scheduler_metadata: Optional[torch.Tensor]
        num_splits: torch.Tensor
        dummy_block_table: torch.Tensor
        cache_lens: torch.Tensor

    fp8_extra_metadata: Optional[FP8KernelMetadata] = None


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0

    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = block_id < max_num_blocks_per_req
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)

    # If token == -1 OR block_id OOB, output -1; else base * BLOCK_SIZE + offset
    out_val = tl.where(is_invalid_tok | (~valid_block), -1,
                       base * BLOCK_SIZE + inblock_off)

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
        req_id: torch.Tensor,  # int32 [num_tokens]
        block_table: torch.
    Tensor,  # int32 [num_requests, max_num_blocks_per_req]
        token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
        BLOCK_SIZE: int = 64,
        NUM_TOPK_TOKENS: int = 2048,
        BLOCK_N: int = 128,  # tile width along columns
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id], 
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be 
        out-of-bounds.
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, \
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by" \
        f"BLOCK_N ({BLOCK_N})"

    num_tokens = req_id.shape[0]
    num_requests, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )
    return out

def kunlun_convert_req_index_to_global_index(
        req_id: torch.Tensor,  # int32 [num_tokens]
        block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
        token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
        BLOCK_SIZE: int = 64,
        NUM_TOPK_TOKENS: int = 2048,
):
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS

    num_tokens = req_id.shape[0]
    num_requests, max_num_blocks_per_req = block_table.shape
    
    out = torch.zeros_like(token_indices)
    
    # Compute block_id and inblock_off for all tokens at once
    block_id = token_indices // BLOCK_SIZE
    inblock_off = token_indices % BLOCK_SIZE
    
    # Create mask for invalid tokens (tok < 0)
    invalid_tok_mask = token_indices < 0
    
    # Create mask for out-of-bounds block_id
    oob_block_mask = block_id >= max_num_blocks_per_req
    
    # Combine masks - output -1 for either condition
    invalid_mask = invalid_tok_mask | oob_block_mask
    
    # Get request IDs expanded to match token_indices shape
    req_ids_expanded = req_id.unsqueeze(1).expand(-1, NUM_TOPK_TOKENS)
    
    # Gather base addresses from block_table
    # Clamp block_id to avoid index errors (we'll mask these out anyway)
    block_id_clamped = torch.clamp(block_id, 0, max_num_blocks_per_req - 1)
    
    # Use advanced indexing to get base addresses
    base_addrs = block_table[req_ids_expanded, block_id_clamped]
    
    # Compute the global indices
    global_indices = base_addrs * BLOCK_SIZE + inblock_off
    
    # Apply mask: set invalid positions to -1
    out = torch.where(invalid_mask, torch.tensor(-1, dtype=torch.int32, device=token_indices.device), global_indices)
    
    return out

def kunlun_concat_and_cache_mla(
    kv_c: torch.Tensor, #[num_tokens, kv_lora_rank]
    k_pe: torch.Tensor, #[num_tokens, pe_dim]
    kv_cache: torch.Tensor, #[num_blocks, block_size, (kv_lora_rank + pe_dim)]
    slot_mapping: torch.Tensor, #[num_tokens] or [num_actual_tokens]
    kv_cache_dtype: str, 
    scale: torch.Tensor
):
    num_tokens = slot_mapping.shape[0]
    kv_lora_rank = kv_c.shape[1]
    pe_dim = k_pe.shape[1]
    block_size = kv_cache.shape[1]
    
    def kunlun_fp8_ds_mla():
        for token_idx in range(num_tokens):
            slot = slot_mapping[token_idx].item()
            if slot < 0: continue
            block_idx = slot // block_size
            block_offset = slot % block_size
            kv_c_i = kv_c[token_idx].view(4,kv_lora_rank//4).contiguous()
            kv_c_i_int8 = torch.zeros(
            kv_c_i.shape,
            device=kv_c.device,
            dtype=torch.int8,
            )
            kv_c_i_scale = torch.zeros(
                [kv_c_i.shape[0], 1],
                device=kv_c.device,
                dtype=torch.float32,
            )
            torch.ops._C.quant2d(kv_c_i, kv_c_i_int8, kv_c_i_scale, force_sdnn=True)
            kv_c_i_scale /= 127
            kv_cache[block_idx, block_offset, :kv_lora_rank] = kv_c_i_int8.view(-1).view(torch.uint8).contiguous()
            kv_cache[block_idx, block_offset, kv_lora_rank:kv_lora_rank + 16] = kv_c_i_scale.view(-1).view(torch.uint8).contiguous()
            kv_cache[block_idx, block_offset, kv_lora_rank+16:] = k_pe[token_idx, :].view(torch.uint8).contiguous()
            
    def kunlun_mla():
        for token_idx in range(num_tokens):
            slot = slot_mapping[token_idx].item()
            if slot < 0: continue
            block_idx = slot // block_size
            block_offset = slot % block_size
            kv_cache[block_idx, block_offset, :kv_lora_rank] = kv_c[token_idx, :].contiguous()
            kv_cache[block_idx, block_offset, kv_lora_rank:] = k_pe[token_idx, :].contiguous()
            
    if (kv_cache_dtype == "fp8_ds_mla"):
        assert kv_lora_rank == 512, "kv_lora_rank must be 512 for fp8_ds_mla"
        assert pe_dim == 64, "pe_dim must be 64 for fp8_ds_mla"
        assert kv_cache.shape[2] == 656 // kv_cache.element_size(), "kv_cache.shape[2] must be 656 bytes for fp8_ds_mla"
        assert kv_c.element_size() == 2, "kv_c.element_size() must be 2 for fp8_ds_mla"
        assert k_pe.element_size() == 2, "k_pe.element_size() must be 2 for fp8_ds_mla"
        kunlun_fp8_ds_mla()
    else:
        assert kv_cache.shape[2] == kv_lora_rank + pe_dim
        kunlun_mla()
    
    
@dataclass
class FlashMLASparseMetadataBuilder(
        AttentionMetadataBuilder[FlashMLASparseMetadata]):
    cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        cache_config = vllm_config.cache_config
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.device = device

        # Treat requests with query length <= 1 as decodes to match the
        # DeepGEMM indexer constraint (fp8_paged_mqa_logits only supports next_n <= 2)
        # 从最新版本vllm中引入的
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        props = torch.cuda.get_device_properties(device)
        sm_count = props.multi_processor_count

        self.num_heads = self.model_config.get_num_attention_heads(
            parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.use_fp8_kv_cache = cache_config.cache_dtype == "fp8_ds_mla"

        self.topk_tokens_tensor = torch.tensor([self.topk_tokens],
                                               device=device,
                                               dtype=torch.int32)
        # self.max_model_len_tensor = torch.tensor(
        #     [self.model_config.max_model_len],
        #     device=device,
        #     dtype=torch.int32)

        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty((1, 1),
                                             dtype=torch.int32,
                                             device=self.device)

        # Equation taken from FlashMLA/csrc/pybind.cpp
        h_q, h_k = self.num_heads, 1
        s_q = 1  # inversely proportional to s_q, so s_q = 1 is the largest
        max_num_sm_parts = int(
            max((sm_count // 2) / h_k // (cdiv(h_q // h_k, 2 * 64) * s_q), 1))
        if current_platform.is_device_capability(100):
            max_num_sm_parts *= 2
        self.tile_scheduler_metadata_buffer = torch.zeros(
            # TileSchedulerMetaDataSize = 8
            # see: FlashMLA/csrc/params.h
            (max_num_sm_parts, 8),
            dtype=torch.int32,
            device=device)
        self.num_splits_buffer = torch.zeros(
            # We pack all the tokens into one batch for sparse attention.
            # Otherwise, we can exceed the sm of `get_mla_metadata`.
            (
                2, ),
            dtype=torch.int32,
            device=device)
        self.req_id_per_token_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_batched_tokens, ),
            dtype=torch.int32,
            device=device)
    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> FlashMLASparseMetadata:

        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu,
                            dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths)
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[:req_id_per_token.shape[0]]\
            .copy_(torch.from_numpy(req_id_per_token), non_blocking=True)
        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]

        fp8_extra_metadata = None

        if self.use_fp8_kv_cache:
            cache_seqlens_cpu, cache_seqlens = get_mla_metadata(
                cache_seqlens=self.topk_tokens_tensor,
            )
            fp8_extra_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
                scheduler_metadata=None,
                num_splits=None,
                # cache_lens and block_table are basically unused in sparse case
                # but the decode kernel will treat -1 and indices >= cache_lens
                # as invalid so we make sure cache_lens is large enough to not
                # accidentally mark indices invalid, we will use -1 exclusively
                # to mark invalid indices
                cache_lens=cache_seqlens_cpu,
                dummy_block_table=self.dummy_block_table)

        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold or 1,
                require_uniform=True,
            )
        )

        # For pure decode batches, prefill_request_id will be None
        # For mixed batches, it will have -1 for decode and request_id for prefill
        prefill_metadata = None
        if num_prefills > 0:
            prefill_metadata = MLASparsePrefillMetadata(
                query_start_loc = common_attn_metadata.query_start_loc[num_decodes:] - common_attn_metadata.query_start_loc[num_decodes], #因为prefiil、decode请求是分离，所以需要对q进行切分，故需调整该值
                query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[num_decodes:] - common_attn_metadata.query_start_loc_cpu[num_decodes],
            )

        decode_metadata = None
        if num_decodes > 0:
            max_seq_len = int(common_attn_metadata.seq_lens_cpu[:num_decodes].max())

            decode_metadata = FlashMLASparseDecodeAndContextMetadata(
                max_seq_len=max_seq_len,
                seq_lens=common_attn_metadata.seq_lens[:num_decodes],
                seq_lens_cpu=common_attn_metadata.seq_lens_cpu[:num_decodes],
            )


        metadata = FlashMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            fp8_extra_metadata=fp8_extra_metadata,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            decode_metadata=decode_metadata,
            prefill_metadata=prefill_metadata
        )
        return metadata


class FlashMLASparseImpl(MLACommonBaseImpl[FlashMLASparseMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            topk_indice_buffer: Optional[torch.Tensor] = None,
            indexer: Optional["Indexer"] = None,
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)
        self.softmax_scale = scale
        assert indexer is not None
        self.topk_indices_buffer = indexer.topk_indices_buffer
        self.padding = 128 if current_platform.is_device_capability(
            100) else 64

    def _forward_bf16_kv(
            self, q: torch.Tensor, kv_c_and_k_pe_cache: torch.Tensor,
            topk_indices: torch.Tensor,
            attn_metadata: FlashMLASparseMetadata) -> torch.Tensor:
        
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.contiguous().view(
            -1,  kv_c_and_k_pe_cache.shape[-1])
    
        # num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decodes = attn_metadata.num_decodes

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        def _bf16_decode(q: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
            # Reshape q: (num_decode_tokens, num_heads, head_dim)
            #         -> (num_decodes, seq_len, num_heads, head_dim)
            q = reshape_query_for_spec_decode(q, num_decodes)
            seq_len = q.shape[1]
            # Reshape topk_indices: (num_decode_tokens, topk)
            #                    -> (num_decodes, seq_len, topk)
            topk_indices = topk_indices.view(num_decodes, seq_len, -1)
            decode_metadata = attn_metadata.decode_metadata
            _attn_out, _, _ = kunlun_flash_mla_with_kvcache(
                q=q,
                k_cache=kv_c_and_k_pe_cache,
                head_dim_v=512,
                cache_seqlens=decode_metadata.seq_lens,
                cache_seqlens_cpu=decode_metadata.seq_lens_cpu,
                is_fp8_kvcache=False,
                indices=topk_indices, 
                softmax_scale=self.softmax_scale,
                max_seq_kv=decode_metadata.max_seq_len
            )
            # Reshape output: (num_decodes, seq_len, num_heads, head_dim_v)
            #              -> (num_decode_tokens, num_heads, head_dim_v)
            return reshape_attn_output_for_spec_decode(_attn_out)
        
        def _bf16_prefill(q: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
            prefill_metadata = attn_metadata.prefill_metadata
            topk_indices = topk_indices.view(num_prefill_tokens, 1, -1)
            # NOTE: 只有prefill阶段attn_metadata.query_start_loc是符合klx算子需求的
            _attn_out = flash_mla_sparse_prefill(
                q=q,
                kv=kv_c_and_k_pe_cache, 
                indices=topk_indices,
                sm_scale=self.softmax_scale,
                q_lod_xpu=prefill_metadata.query_start_loc,
                q_lod_cpu=prefill_metadata.query_start_loc_cpu
            )[0]
            return _attn_out

        topk_indices_global = torch.ops.xspeedgate_ops.convert_req_index_to_global_index(
                req_id=attn_metadata.req_id_per_token,
                block_table=attn_metadata.block_table,
                token_indices=topk_indices,
                block_size=attn_metadata.block_size,
                num_topk_tokens=attn_metadata.topk_tokens,
        )

        attn_out = torch.empty(
            (num_tokens, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        if has_prefill:
            prefill_q = q[num_decode_tokens:]
            prefill_topk_indices_global = topk_indices_global[num_decode_tokens:]
            attn_out[num_decode_tokens:] = _bf16_prefill(prefill_q, prefill_topk_indices_global)

        # 处理decode部分 - 需要正确的block table映射print
        if has_decode:
            decode_q = q[:num_decode_tokens]
            decode_topk_indices_global = topk_indices_global[:num_decode_tokens]
            attn_out[:num_decode_tokens] = _bf16_decode(decode_q, decode_topk_indices_global)

        return attn_out
        

    def _forward_fp8_kv(self, q: torch.Tensor,
                        kv_c_and_k_pe_cache: torch.Tensor,
                        topk_indices: torch.Tensor,
                        attn_metadata: FlashMLASparseMetadata) -> torch.Tensor:
        # TODO: When fwd_kvcache_mla supports uint8 kv cache, execute this function.
        assert attn_metadata.fp8_extra_metadata is not None
        extra_metadata = attn_metadata.fp8_extra_metadata

        _attn_out, _ = flash_mla_with_kvcache(
            q=q.unsqueeze(0),  # unsqueeze to add batch_dim
            k_cache=kv_c_and_k_pe_cache,
            block_table=extra_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=extra_metadata.cache_lens,
            tile_scheduler_metadata=extra_metadata.scheduler_metadata, # None
            num_splits=extra_metadata.num_splits, # None
            is_fp8_kvcache=True,
            indices=topk_indices.unsqueeze(0),  # unsqueeze to add batch_dim
            softmax_scale=self.softmax_scale,
            max_seq_kv=attn_metadata.max_seq_len
        )

        return _attn_out

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for MLACommonImpl")

        if attn_metadata is None:
            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs

        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        ql_nope = ql_nope.transpose(0, 1)

        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        
        q = torch.cat([ql_nope, q_pe], dim=-1)

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            torch.ops._C.concat_and_cache_mla(
                kv_c=k_c_normed,
                k_pe=k_pe.squeeze(1),
                kv_cache=kv_cache,
                slot_mapping=attn_metadata.slot_mapping.flatten(),
            )

        if self.kv_cache_dtype != "fp8_ds_mla":
            attn_out = self._forward_bf16_kv(q, kv_cache, topk_indices,
                                             attn_metadata)
        else:
            # attn_out = self._forward_fp8_kv(q, kv_cache, topk_indices_global,
            #                                 attn_metadata)
            raise NotImplementedError

        self._v_up_proj(attn_out, out=output[:num_actual_toks])
        return output
