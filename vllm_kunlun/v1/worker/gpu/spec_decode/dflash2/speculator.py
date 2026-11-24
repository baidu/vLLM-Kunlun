# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-width DFlash2 proposals for the vLLM 0.25.1 V2 runner."""

from typing import Any

import kunlun_ops
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.cudagraph_utils import AttentionState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from .cudagraph import DFlash2CudaGraphManager


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._validate_config(vllm_config)
        super().__init__(vllm_config, device)
        if self.dtype == torch.float16:
            # Keep the auxiliary projection finite until context RMSNorm.
            self.hidden_states = self.hidden_states.float()
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
        self._aux_staging: torch.Tensor | None = None

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
        self.query_cudagraph_manager = DFlash2CudaGraphManager(self, cudagraph_mode)

    def capture(self, attn_states=None):
        self.query_cudagraph_manager.capture()

    def _pad_graph_inputs(self, num_reqs, num_reqs_padded):
        if num_reqs_padded == num_reqs:
            return
        width = self.num_query_per_req
        start, end = num_reqs * width, num_reqs_padded * width
        self.input_buffers.input_ids[start:end].zero_()
        self.input_buffers.positions[start:end].zero_()
        self._query_slots[:, start:end].fill_(-1)
        for table in self.block_tables.input_block_tables:
            table[num_reqs:num_reqs_padded].zero_()
        # Keep complete query blocks for grouped conv and avoid empty-softmax
        # rows in XFA. These synthetic requests read only the zeroed null block.
        self.input_buffers.seq_lens[num_reqs:num_reqs_padded].fill_(width)
        self.input_buffers.query_start_loc[num_reqs + 1 : num_reqs_padded + 1].copy_(
            self._anchor_indices[num_reqs:num_reqs_padded] + width
        )
        # Reuse a real hidden row for discarded candidates, even if the padded
        # model rows overflow in FP16. Candidate selection is per request.
        steps = self.num_speculative_steps
        self.sample_indices[num_reqs * steps : num_reqs_padded * steps].zero_()

    def _precompute_context_kv(self, start, end, dummy_run=False):
        slots = None
        if not dummy_run:
            if self._layer_group_idx is not None:
                slots = [
                    self._context_slot_mappings[i, start:end]
                    for i in self._layer_group_idx
                ]
            else:
                slots = self._context_slot_mappings[0, start:end]
        self.model.precompute_and_store_context_kv(
            self.hidden_states[start:end],
            self.context_positions[start:end],
            slots,
        )

    def _build_query_attn_state(self, batch):
        # Eager uses exact device lengths; capture fixes host launch bounds later.
        count = batch * self.num_query_per_req
        slots = self._query_slots[:, :count]
        metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=batch,
            num_tokens=count,
            query_start_loc_gpu=self.input_buffers.query_start_loc[: batch + 1],
            query_start_loc_cpu=self.arange[: batch + 1] * self.num_query_per_req,
            max_query_len=self.num_query_per_req,
            seq_lens=self.input_buffers.seq_lens[:batch],
            max_seq_len=self.max_model_len,
            block_tables=[x[:batch] for x in self.block_tables.input_block_tables],
            slot_mappings=slots,
            kv_cache_config=self.kv_cache_config,
            causal=self.dflash_causal,
        )
        return AttentionState(
            metadata, build_slot_mappings_by_layer(slots, self.kv_cache_config)
        )

    def _prepare_graph_forward(self, batch, metadata):
        # Same fixed verify-width prefix as main; long prefill overflow is eager.
        count = min(batch * self.num_query_per_req, self.max_num_tokens)
        if self._aux_staging is not None:
            self.hidden_states[:count].copy_(
                self.model.combine_hidden_states(self._aux_staging[:count])
            )
        self._precompute_context_kv(0, count)
        for md in {id(md): md for md in metadata.values()}.values():
            md.kv_lod_xpu[:1].zero_()
            torch.cumsum(
                self.input_buffers.seq_lens[:batch], dim=0, out=md.kv_lod_xpu[1:]
            )

    def _prepare_graph_replay(
        self, input_batch, num_graph_reqs, last_hidden_states, aux_hidden_states
    ):
        self._pad_graph_inputs(input_batch.num_reqs, num_graph_reqs)
        count = input_batch.num_tokens
        capacity = min(num_graph_reqs * self.num_query_per_req, self.max_num_tokens)
        self._store_hidden_states(
            count, last_hidden_states, aux_hidden_states, capacity
        )
        if count > capacity:
            self._precompute_context_kv(capacity, count)
        else:
            self.hidden_states[count:capacity].zero_()
            self.context_positions[count:capacity].zero_()
            self._context_slot_mappings[:, count:capacity].fill_(-1)

    def _store_hidden_states(
        self, count, last_hidden_states, aux_hidden_states, graph_count=0
    ):
        start = 0
        if graph_count and self._aux_staging is not None:
            if not aux_hidden_states:
                raise ValueError(
                    "DFlash2 aux graph requires target auxiliary hidden states"
                )
            if (
                any(h.ndim != 2 or h.shape[0] < count for h in aux_hidden_states)
                or sum(h.shape[-1] for h in aux_hidden_states)
                != self._aux_staging.shape[1]
            ):
                raise ValueError(
                    "DFlash2 auxiliary states do not match the fixed graph input shape"
                )
            start = min(count, graph_count)
            if start:
                torch.cat(
                    [h[:start] for h in aux_hidden_states],
                    dim=-1,
                    out=self._aux_staging[:start],
                )
            if start < graph_count:
                self._aux_staging[start:graph_count].zero_()
        if start == count:
            return
        hidden = last_hidden_states[start:count]
        if aux_hidden_states:
            hidden = self.model.combine_hidden_states(
                torch.cat([h[start:count] for h in aux_hidden_states], dim=-1)
            )
        self.hidden_states[start:count].copy_(hidden)

    def _run_model(
        self,
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    ):
        # FULL capture records eager native ops; never compile a new model path
        # inside capture or invoke nested piecewise graph wrappers.
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
            slot_mapping=slot_mappings,
            skip_compiled=bool(attn_metadata)
            and any(
                getattr(md, "use_static_lod_bounds", False)
                for md in attn_metadata.values()
            ),
        ):
            return self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )

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
        self._init_aux_staging(model)
        return model

    def _init_aux_staging(self, model):
        # Reserve this storage before memory profiling and KV allocation.
        mode = self.vllm_config.compilation_config.cudagraph_mode
        if (
            mode is not None
            and mode.decode_mode() == CUDAGraphMode.FULL
            and not self.speculative_config.enforce_eager
            and model.model.use_aux_hidden_state
        ):
            self._aux_staging = torch.zeros(
                min(self.max_num_reqs * self.num_query_per_req, self.max_num_tokens),
                model.model.fc.input_size,
                dtype=self.dtype,
                device=self.device,
            )

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
        if self.query_cudagraph_manager is not None:
            self.query_cudagraph_manager.validate_capture()

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
            cudagraph_runtime_mode,
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
        query_count = batch * self.num_query_per_req
        if dummy_run and skip_attn_for_dummy_run:
            self._store_hidden_states(
                context_count, last_hidden_states, aux_hidden_states
            )
            self.input_buffers.input_ids[:query_count].zero_()
            self.input_buffers.positions[:query_count].zero_()
            self.context_positions[:context_count].zero_()
            self._precompute_context_kv(0, context_count, dummy_run=True)
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
        manager = self.query_cudagraph_manager
        if manager is not None and not (dummy_run or is_profile):
            desc = manager.dispatch(batch, query_count, self.num_query_per_req, 0)
            if desc.cg_mode == CUDAGraphMode.FULL:
                self._prepare_graph_replay(
                    input_batch, desc.num_reqs, last_hidden_states, aux_hidden_states
                )
                manager.run_fullgraph(desc)
                return self.draft_tokens[:batch]
        self._store_hidden_states(context_count, last_hidden_states, aux_hidden_states)
        self._precompute_context_kv(0, context_count, dummy_run)
        draft_metadata, by_layer = self._build_query_attn_state(batch)
        self._generate_draft(
            batch, query_count, draft_metadata, by_layer, num_tokens_across_dp
        )
        return self.draft_tokens[:batch]
