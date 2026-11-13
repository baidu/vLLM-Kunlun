# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-width DFlash2 proposals for the vLLM 0.25.1 V2 runner, eagerly."""

from typing import Any

import kunlun_ops
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._validate_config(vllm_config)
        super().__init__(vllm_config, device)
        self.selector_top_k = int(
            self.draft_model_config.hf_config.dflash_config["selector_top_k"]
        )
        width = self.num_query_per_req
        self._query_capacity = max(self.max_num_tokens, self.max_num_reqs * width)
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self._query_capacity,
            device=device,
        )
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device) * width
        )
        samples = (
            self._anchor_indices[:, None]
            + torch.arange(1, width, device=device)[None, :]
        ).reshape(-1)
        self.sample_indices.copy_(samples)

    @staticmethod
    def _validate_config(vllm_config: VllmConfig) -> None:
        spec = vllm_config.speculative_config
        parallel = vllm_config.parallel_config
        unsupported = []
        if spec.draft_sample_method != "greedy":
            unsupported.append("greedy draft sampling is required")
        if spec.rejection_sample_method != "standard":
            unsupported.append("standard rejection sampling is required")
        if not 1 <= spec.num_speculative_tokens <= 31:
            unsupported.append("num_speculative_tokens must be between 1 and 31")
        # Also guard direct/forced-MRV2 construction that bypasses the gate.
        if parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1:
            unsupported.append("PP1 and DP1 are required")
        if parallel.enable_expert_parallel:
            unsupported.append("expert parallelism is unsupported")
        if (
            parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            unsupported.append("context parallelism is unsupported")
        if vllm_config.lora_config is not None:
            unsupported.append("LoRA is unsupported")
        if spec.use_local_argmax_reduction:
            unsupported.append("local argmax reduction is unsupported by the selector")
        if (
            spec.draft_model_config.get_vocab_size()
            != vllm_config.model_config.get_vocab_size()
        ):
            unsupported.append("draft and target vocabularies must match")
        if unsupported:
            raise ValueError("Kunlun MRV2 DFlash2: " + "; ".join(unsupported))

    def init_cudagraph_manager(self, cudagraph_mode):
        # The target may use graphs; every draft proposal remains eager.
        self.query_cudagraph_manager = None

    def capture(self, attn_states=None):
        return None

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        target = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        if model.draft_id_to_target_id is not None:
            raise ValueError("DFlash2 selector requires the full target vocabulary")
        # v0.25.1's DFlash loader looks on the outer target object for lm_head;
        # Qwen3.5 conditional-generation models keep it on the language model.
        model.lm_head = target.lm_head
        return model

    def set_attn(self, model_state, kv_cache_config, block_tables):
        super().set_attn(model_state, kv_cache_config, block_tables)
        # Separate draft query slots from target slots. A parallel proposal can
        # exceed the target's token budget during a small chunked prefill.
        self._query_slots = torch.full(
            (len(kv_cache_config.kv_cache_groups), self._query_capacity),
            -1,
            dtype=torch.int64,
            device=self.device,
        )

    def _generate_draft(
        self,
        num_reqs,
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    ):
        hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            CUDAGraphMode.NONE,
        )
        count = num_reqs * self.num_speculative_steps
        hidden = hidden[self.sample_indices[:count]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidates, unary = self.model.compute_candidates(hidden.flatten(0, 1))
        candidates = candidates.view(
            num_reqs, self.num_speculative_steps, self.selector_top_k
        )
        scores = self.model.model.candidate_selector(
            candidates,
            unary.view_as(candidates),
            hidden,
            self.input_buffers.input_ids[self._anchor_indices[:num_reqs]],
        )
        self.draft_tokens[:num_reqs].copy_(
            kunlun_ops.dflash_greedy_select_path(candidates, scores)
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        batch, context_count = input_batch.num_reqs, input_batch.num_tokens
        if batch == 0:
            return self.draft_tokens[:0]
        if mm_inputs is not None:
            raise NotImplementedError("Kunlun MRV2 DFlash2 supports text inputs only")
        hidden = last_hidden_states[:context_count]
        if aux_hidden_states:
            hidden = self.model.combine_hidden_states(
                torch.cat([x[:context_count] for x in aux_hidden_states], dim=-1)
            )
        self.hidden_states[:context_count].copy_(hidden)
        query_count = batch * self.num_query_per_req
        if dummy_run and skip_attn_for_dummy_run:
            self.input_buffers.input_ids[:query_count].zero_()
            self.input_buffers.positions[:query_count].zero_()
            self.context_positions[:context_count].zero_()
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:context_count],
                self.context_positions[:context_count],
            )
            self._generate_draft(batch, query_count, None, None, num_tokens_across_dp)
            return self.draft_tokens[:batch]

        self.block_tables.gather_block_tables(input_batch.idx_mapping[:batch], batch)
        # Gathered tables contain expanded kernel-page IDs in hybrid caches.
        for index, gid in enumerate(self.draft_kv_cache_group_ids):
            kunlun_ops.prepare_dflash_inputs(
                target_positions=input_batch.positions[:context_count],
                target_query_start_loc=input_batch.query_start_loc[: batch + 1],
                idx_mapping=input_batch.idx_mapping[:batch],
                last_sampled=last_sampled,
                next_prefill_tokens=next_prefill_tokens,
                num_sampled=num_sampled[:batch],
                num_rejected=num_rejected[:batch],
                input_temperature=temperature,
                input_seeds=seeds,
                block_table=self.block_tables.input_block_tables[gid],
                input_ids_out=self.input_buffers.input_ids,
                query_positions_out=self.input_buffers.positions,
                query_start_loc_out=self.input_buffers.query_start_loc,
                seq_lens_out=self.input_buffers.seq_lens,
                query_slot_mapping_out=self._query_slots[gid],
                context_positions_out=self.context_positions,
                context_slot_mapping_out=self._context_slot_mappings[index],
                sample_indices_out=self.sample_indices,
                sample_pos_out=self.sample_pos,
                sample_idx_mapping_out=self.sample_idx_mapping,
                temperature_out=self.temperature,
                seeds_out=self.seeds,
                block_size=self.block_tables.kernel_block_sizes[gid],
                num_query_per_req=self.num_query_per_req,
                num_speculative_steps=self.num_speculative_steps,
                max_num_reqs=self.max_num_reqs,
                max_num_tokens=self._query_capacity,
                max_model_len=self.max_model_len,
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                cp_rank=self.block_tables.cp_rank,
                cp_size=self.block_tables.cp_size,
                cp_interleave=self.block_tables.cp_interleave,
                sample_from_anchor=self.sample_from_anchor,
            )
        context_slots = None
        if not dummy_run:
            if self._layer_group_idx is not None:
                context_slots = [
                    self._context_slot_mappings[i, :context_count]
                    for i in self._layer_group_idx
                ]
            else:
                context_slots = self._context_slot_mappings[0, :context_count]
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:context_count],
            self.context_positions[:context_count],
            context_slots,
        )
        # Speculative CPU length mirrors are optimistic. The attention builder
        # obtains exact lengths from the device for the eager XFA ABI.
        draft_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=batch,
            num_tokens=query_count,
            query_start_loc_gpu=self.input_buffers.query_start_loc[: batch + 1],
            query_start_loc_cpu=self.arange[: batch + 1] * self.num_query_per_req,
            max_query_len=self.num_query_per_req,
            seq_lens=self.input_buffers.seq_lens[:batch],
            max_seq_len=self.max_model_len,
            block_tables=[x[:batch] for x in self.block_tables.input_block_tables],
            slot_mappings=self._query_slots[:, :query_count],
            kv_cache_config=self.kv_cache_config,
            causal=self.dflash_causal,
        )
        by_layer = build_slot_mappings_by_layer(
            self._query_slots[:, :query_count], self.kv_cache_config
        )
        self._generate_draft(
            batch, query_count, draft_metadata, by_layer, num_tokens_across_dp
        )
        return self.draft_tokens[:batch]
