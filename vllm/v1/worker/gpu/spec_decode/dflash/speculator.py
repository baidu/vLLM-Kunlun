# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.utils import (
    get_dflash_causal,
    load_dflash_model,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id

logger = init_logger(__name__)


class DFlashSpeculator(DraftModelSpeculator):
    _speculator_name = "DFlash"  # For logging, so we can share methods with subclasses

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )

        # Multimodal inputs not currently supported.
        self.supports_mm_inputs = False

        # Each request emits exactly (bonus + N mask) query tokens per step.
        self.num_query_per_req = 1 + self.num_speculative_steps

        self.parallel_drafting_token_id = get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )

        self.dflash_causal = get_dflash_causal(self.draft_model_config)

        # Whether the anchor query position is itself a prediction. DFlash default uses
        # the anchor as the bonus token (only mask tokens predict); DSpark samples from
        # the anchor and the N-1 mask token positions. See _prepare_dflash_inputs_kernel
        self.sample_from_anchor = False

        # Context positions for the K/V precompute. Populated by
        # prepare_dflash_inputs, and processed by the model's
        # precompute_and_store_context_kv method. NOT captured by CUDA graphs.
        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # Per-mask-token sampling buffers. Flattened from (num_reqs, num_spec_tokens).
        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_idx_mapping = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int32, device=device
        )
        # [0, 1, ..., N-1, 0, 1, ..., N-1, ...] -> the per-token column index into
        # draft_logits[req, step, :].
        self.sample_col = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)

        self.query_cudagraph_manager: DFlashCudaGraphManager | None = None
        self.draft_kv_cache_group_id: int = -1

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # PIECEWISE cudagraphs are not supported for dflash
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.num_query_per_req,
            causal=self.dflash_causal,
        )

    def capture(self, attn_states: dict | None = None) -> None:
        logger.info("Capturing model for %s speculator...", self._speculator_name)
        # Reset sampling indices to zero to prevent stale values from prior
        # dummy runs from being baked into the captured graph.
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.zero_()
        assert self.query_cudagraph_manager is not None
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            progress_bar_desc=f"Capturing {self._speculator_name.lower()} CUDA graphs",
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_dflash_model(target_model, self.vllm_config)

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)

        self.draft_kv_cache_group_ids = [
            gid for gid, g in enumerate(self.attn_groups) if g
        ]
        assert self.draft_kv_cache_group_ids, "No draft attention groups found."
        self.draft_kv_cache_group_id = self.draft_kv_cache_group_ids[0]
        self.draft_block_size = self.block_tables.block_sizes[
            self.draft_kv_cache_group_id
        ]

        # Per-group context slot buffers for the precompute (one row per group).
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        # Map each draft decoder layer to the index (within draft_kv_cache_group_ids)
        # of the kv-cache group its cache belongs to. Models that share a single group
        # leave this as None and share one context slot mapping.
        self._layer_group_idx: list[int] | None = None
        if hasattr(self.model, "get_draft_kv_cache_layer_names"):
            name_to_gid = {
                ln: gid
                for gid, group in enumerate(kv_cache_config.kv_cache_groups)
                for ln in group.layer_names
            }
            gid_to_idx = {gid: i for i, gid in enumerate(self.draft_kv_cache_group_ids)}
            self._layer_group_idx = [
                gid_to_idx[name_to_gid[name]]
                for name in self.model.get_draft_kv_cache_layer_names()
            ]

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            last_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )
        return last_hidden_states

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        # sample_pos is the predicted token's position Q; verification keys
        # Gumbel by the predecessor (Q-1). sample_draft adds +1, so pass Q-2.
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            self.sample_pos[:num_sample] - 2,
            self.sample_idx_mapping[:num_sample],
            self.temperature,
            self.seeds,
            self.sample_col[:num_sample],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs] = draft_tokens.view(
            num_reqs, self.num_speculative_steps
        )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        num_query_per_req: int | None = None,
        causal: bool = False,
    ) -> dict[str, Any] | None:
        if not self.draft_attn_layer_names:
            return None
        assert num_query_per_req is None  # Omitted for DFlash, read from self instead
        return super()._build_draft_attn_metadata(
            num_reqs,
            num_reqs_padded,
            num_tokens_padded,
            num_query_per_req=self.num_query_per_req,
            causal=causal,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len
        )

        # NOTE: To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions.
        if aux_hidden_states:
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        if dummy_run and skip_attn_for_dummy_run:
            # Memory profiling path: block_tables / kv_cache_config are not initialized.
            # Since DFlash needs to build its own attention metadata, we must skip the
            # preparation in this path and run a minimal forward pass.
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_target_tokens],
                self.context_positions[:num_target_tokens],
            )
            # DFlash processes all speculative tokens in one forward pass,
            # so the real token count is num_query_tokens.
            self._prepare_eplb_forward(num_query_tokens)
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]

        # The query slot mapping is written into the shared BlockTables slot_mappings.
        # That buffer's address is what the captured CUDA graph reads from at replay.
        assert self.draft_kv_cache_group_id >= 0
        # Support multiple draft KV cache groups by preparing inputs once for each
        for i, gid in enumerate(self.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[gid],
                self.context_positions,
                self._context_slot_mappings[i],
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                self.block_tables.input_block_tables[gid],
                self.block_tables.block_sizes[gid],
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                self.max_num_tokens,
                self.max_model_len,
                self.sample_from_anchor,
            )

        # Pre-insert context K/V into the cache. Runs eagerly outside the captured graph
        # because the context shape varies per step. During dummy runs the block tables
        # are placeholders, so we skip the cache write to avoid clobbering real entries.
        # Each layer uses the context slots of its own kv-cache group.
        if dummy_run:
            context_slots: torch.Tensor | list[torch.Tensor | None] | None = None
        elif self._layer_group_idx is not None:
            context_slots = [
                self._context_slot_mappings[gidx][:num_target_tokens]
                for gidx in self._layer_group_idx
            ]
        else:
            context_slots = self._context_slot_mappings[0][:num_target_tokens]
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:num_target_tokens],
            self.context_positions[:num_target_tokens],
            context_slots,
        )

        # Every DFlash step has exactly num_query_per_req tokens, so we can use FULL CGs
        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_query_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens

        # Rebuild the draft attention metadata even when replaying the FULL
        # graph so that any attention metadata builder state is updated.
        draft_attn_metadata = self._build_draft_attn_metadata(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            causal=self.dflash_causal,
        )
        draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )

        # DFlash processes all speculative tokens in one forward pass,
        # so the real token count is num_query_tokens.
        self._prepare_eplb_forward(num_query_tokens)

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )

        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_dflash_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_query = (j >= num_ctx) & (j < num_ctx + num_query_per_req)
    query_off = j - num_ctx

    # --- Context positions / slots ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_ctx,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    # --- Query positions / input_ids / slots ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // block_size
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_slot = q_block_id * block_size + (query_pos % block_size)

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # --- Sample indices / positions / idx_mapping ---
    # When SAMPLE_FROM_ANCHOR (DSpark), so we sample at EVERY query position
    # and each position k predicts the NEXT token (sampled position = query_pos + 1).
    # Otherwise (DFlash default) the anchor is the bonus token and only the mask tokens
    # at offsets > 0 are sampled from, each AT its own position.
    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    is_sample = is_query & (query_off >= sample_off)
    sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
    sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        # seq_lens is the absolute sequence length the draft attention
        # reads up to (context + query), not just the count of accepted
        # tokens this step.
        tl.store(out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req)
        if req_idx == num_reqs - 1:
            # Pad per-request buffers to max_num_reqs for CUDA graph safety.
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            # Padded sample slots point at query index 0 (a valid row in
            # last_hidden_states) so CG replay never reads OOB. Padded
            # sample idx mappings point to -1, which is ignored during
            # sampling to prevent writing stale values to draft logits.
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            # Pad query slot mappings past num_query_tokens with PAD so the
            # captured CG sees PAD slots (no K/V write) for replay sizes
            # larger than the current request count.
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs, max_num_blocks]
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    # Cover the longest possible per-request span (ctx + query). Use the max
    # per-request query length, not the total token count across the batch.
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# === KUNLUN_NATIVE_P1_PATCH ===

def prepare_dflash_inputs_native(
    input_buffers, query_slot_mapping, context_positions, context_slot_mapping,
    sample_indices, sample_pos, sample_idx_mapping, input_batch,
    num_sampled, num_rejected, last_sampled, next_prefill_tokens,
    block_table, block_size, parallel_drafting_token_id, num_query_per_req,
    num_speculative_steps, max_num_reqs, max_num_tokens, max_model_len,
    sample_from_anchor=False,
):
    import torch as _t
    dev = input_buffers.input_ids.device
    num_reqs = input_batch.num_reqs
    tpos = input_batch.positions
    tqsl = input_batch.query_start_loc
    idx_mapping = input_batch.idx_mapping
    nblk = block_table.size(1)
    out_input_ids = input_buffers.input_ids
    out_qpos = input_buffers.positions
    out_qsl = input_buffers.query_start_loc
    out_seqlens = input_buffers.seq_lens
    sample_off = 0 if sample_from_anchor else 1

    tpos_l = tpos.reshape(-1).tolist()
    tqsl_l = tqsl.reshape(-1).tolist()
    idx_l = idx_mapping.reshape(-1).tolist()
    nrej_l = num_rejected.reshape(-1).tolist()
    nsamp_l = num_sampled.reshape(-1).tolist()
    last_l = last_sampled.reshape(-1).tolist()
    nextp_l = next_prefill_tokens.reshape(-1).tolist()

    for req_idx in range(num_reqs):
        req_state_idx = int(idx_l[req_idx])
        ctx_start = int(tqsl_l[req_idx])
        ctx_end = int(tqsl_l[req_idx + 1])
        num_ctx = ctx_end - ctx_start
        num_rej = int(nrej_l[req_idx])
        valid_ctx_end = ctx_end - num_rej
        num_samp = int(nsamp_l[req_idx])
        bonus_token = int(last_l[req_state_idx]) if num_samp > 0 else int(nextp_l[req_state_idx])
        last_valid_pos = int(tpos_l[valid_ctx_end - 1])
        query_base = req_idx * num_query_per_req

        if num_ctx > 0:
            cpos = tpos[ctx_start:ctx_start + num_ctx].to(_t.int64)
            cbn = _t.clamp(cpos // block_size, max=nblk - 1)
            cbid = block_table[req_idx, cbn].to(_t.int64)
            cslot = cbid * block_size + (cpos % block_size)
            context_positions[ctx_start:ctx_start + num_ctx] = cpos.to(context_positions.dtype)
            context_slot_mapping[ctx_start:ctx_start + num_ctx] = cslot.to(context_slot_mapping.dtype)

        qoff = _t.arange(num_query_per_req, device=dev, dtype=_t.int64)
        qpos = last_valid_pos + 1 + qoff
        input_id = _t.full((num_query_per_req,), parallel_drafting_token_id, device=dev, dtype=out_input_ids.dtype)
        input_id[0] = bonus_token
        qbn = _t.clamp(qpos // block_size, max=nblk - 1)
        qbid = block_table[req_idx, qbn].to(_t.int64)
        qslot = qbid * block_size + (qpos % block_size)
        out_input_ids[query_base:query_base + num_query_per_req] = input_id
        out_qpos[query_base:query_base + num_query_per_req] = _t.clamp(qpos, max=max_model_len - 1).to(out_qpos.dtype)
        query_slot_mapping[query_base:query_base + num_query_per_req] = qslot.to(query_slot_mapping.dtype)

        nsmp = num_query_per_req - sample_off
        if nsmp > 0:
            k = _t.arange(nsmp, device=dev, dtype=_t.int64)
            s_idx = req_idx * num_speculative_steps + k
            s_qidx = query_base + (k + sample_off)
            base_pos = last_valid_pos + 1 + (k + sample_off)
            s_pos = base_pos + 1 if sample_from_anchor else base_pos
            sample_indices[s_idx] = s_qidx.to(sample_indices.dtype)
            sample_pos[s_idx] = s_pos.to(sample_pos.dtype)
            sample_idx_mapping[s_idx] = req_state_idx

        out_qsl[req_idx] = query_base
        out_seqlens[req_idx] = last_valid_pos + 1 + num_query_per_req

    last_query_end = num_reqs * num_query_per_req
    out_qsl[num_reqs:max_num_reqs + 1] = last_query_end
    if max_num_reqs > num_reqs:
        out_seqlens[num_reqs:max_num_reqs] = 0
    pad_start = num_reqs * num_speculative_steps
    pad_end = max_num_reqs * num_speculative_steps
    if pad_end > pad_start:
        sample_indices[pad_start:pad_end] = 0
        sample_pos[pad_start:pad_end] = 0
        sample_idx_mapping[pad_start:pad_end] = -1
    q_pad_start = num_reqs * num_query_per_req
    if max_num_tokens > q_pad_start:
        query_slot_mapping[q_pad_start:max_num_tokens] = PAD_SLOT_ID


prepare_dflash_inputs = prepare_dflash_inputs_native  # noqa: F811


# === KUNLUN_DFLASH_VEC_PATCH ===
def prepare_dflash_inputs_vec(
    input_buffers, query_slot_mapping, context_positions, context_slot_mapping,
    sample_indices, sample_pos, sample_idx_mapping, input_batch,
    num_sampled, num_rejected, last_sampled, next_prefill_tokens,
    block_table, block_size, parallel_drafting_token_id, num_query_per_req,
    num_speculative_steps, max_num_reqs, max_num_tokens, max_model_len,
    sample_from_anchor=False,
):
    import torch as _t
    dev = input_buffers.input_ids.device
    R = input_batch.num_reqs
    nqpr = num_query_per_req
    nss = num_speculative_steps
    bs = block_size
    nblk = block_table.size(1)
    positions = input_batch.positions.reshape(-1)
    qsl = input_batch.query_start_loc.reshape(-1)
    idx_mapping = input_batch.idx_mapping.reshape(-1)
    out_input_ids = input_buffers.input_ids
    out_qpos = input_buffers.positions
    out_qsl = input_buffers.query_start_loc
    out_seqlens = input_buffers.seq_lens
    sample_off = 0 if sample_from_anchor else 1

    r = _t.arange(R, device=dev, dtype=_t.int64)
    req_state_idx = idx_mapping[:R].to(_t.int64)
    ctx_start = qsl[:R].to(_t.int64)
    ctx_end = qsl[1:R + 1].to(_t.int64)
    num_ctx = ctx_end - ctx_start
    num_rej = num_rejected.reshape(-1)[:R].to(_t.int64)
    valid_ctx_end = ctx_end - num_rej
    num_samp = num_sampled.reshape(-1)[:R]
    last_flat = last_sampled.reshape(-1)
    nextp_flat = next_prefill_tokens.reshape(-1)
    bonus = _t.where(num_samp > 0, last_flat[req_state_idx], nextp_flat[req_state_idx]).reshape(-1)
    last_valid_pos = positions[valid_ctx_end - 1].to(_t.int64)
    query_base = r * nqpr

    qsl_np = getattr(input_batch, "query_start_loc_np", None)
    if qsl_np is not None:
        T = int(qsl_np[R])
    else:
        T = int(ctx_end[R - 1].item())
    if T > 0:
        tok_req = _t.repeat_interleave(r, num_ctx)
        cpos = positions[:T].to(_t.int64)
        cbn = _t.clamp(cpos // bs, max=nblk - 1)
        cbid = block_table[tok_req, cbn].to(_t.int64)
        cslot = cbid * bs + (cpos % bs)
        context_positions[:T] = cpos.to(context_positions.dtype)
        context_slot_mapping[:T] = cslot.to(context_slot_mapping.dtype)

    qoff = _t.arange(nqpr, device=dev, dtype=_t.int64)
    qpos = last_valid_pos[:, None] + 1 + qoff[None, :]
    input_id = _t.full((R, nqpr), parallel_drafting_token_id, device=dev, dtype=out_input_ids.dtype)
    input_id[:, 0] = bonus.to(out_input_ids.dtype)
    qbn = _t.clamp(qpos // bs, max=nblk - 1)
    qbid = block_table[r[:, None], qbn].to(_t.int64)
    qslot = qbid * bs + (qpos % bs)
    Q = R * nqpr
    out_input_ids[:Q] = input_id.reshape(-1)
    out_qpos[:Q] = _t.clamp(qpos, max=max_model_len - 1).reshape(-1).to(out_qpos.dtype)
    query_slot_mapping[:Q] = qslot.reshape(-1).to(query_slot_mapping.dtype)

    nsmp = nqpr - sample_off
    if nsmp > 0:
        k = _t.arange(nsmp, device=dev, dtype=_t.int64)
        s_idx = (r[:, None] * nss + k[None, :]).reshape(-1)
        s_qidx = (query_base[:, None] + (k + sample_off)[None, :]).reshape(-1)
        base_pos = last_valid_pos[:, None] + 1 + (k + sample_off)[None, :]
        s_pos = (base_pos + 1) if sample_from_anchor else base_pos
        sample_indices[s_idx] = s_qidx.to(sample_indices.dtype)
        sample_pos[s_idx] = s_pos.reshape(-1).to(sample_pos.dtype)
        sample_idx_mapping[s_idx] = req_state_idx[:, None].expand(R, nsmp).reshape(-1).to(sample_idx_mapping.dtype)

    out_qsl[:R] = query_base.to(out_qsl.dtype)
    out_qsl[R:max_num_reqs + 1] = R * nqpr
    out_seqlens[:R] = (last_valid_pos + 1 + nqpr).to(out_seqlens.dtype)
    if max_num_reqs > R:
        out_seqlens[R:max_num_reqs] = 0

    pad_start = R * nss
    pad_end = max_num_reqs * nss
    if pad_end > pad_start:
        sample_indices[pad_start:pad_end] = 0
        sample_pos[pad_start:pad_end] = 0
        sample_idx_mapping[pad_start:pad_end] = -1
    q_pad_start = R * nqpr
    if max_num_tokens > q_pad_start:
        query_slot_mapping[q_pad_start:max_num_tokens] = PAD_SLOT_ID


import os as _dfv_os  # noqa: E402
if _dfv_os.environ.get("KUNLUN_DFLASH_VEC", "0") == "1":
    prepare_dflash_inputs = prepare_dflash_inputs_vec
    try:
        print("[KUNLUN_DFLASH_VEC] vectorized prepare_dflash_inputs ACTIVE", flush=True)
    except Exception:
        pass
