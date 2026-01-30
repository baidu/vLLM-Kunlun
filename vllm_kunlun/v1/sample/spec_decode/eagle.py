# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import numpy as np
import torch
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


def propose(
    self,
    # [num_tokens]
    target_token_ids: torch.Tensor,
    # [num_tokens]
    target_positions: torch.Tensor,
    # [num_tokens, hidden_size]
    target_hidden_states: torch.Tensor,
    # [batch_size]
    next_token_ids: torch.Tensor,
    last_token_indices: Optional[torch.Tensor],
    common_attn_metadata: CommonAttentionMetadata,
    sampling_metadata: SamplingMetadata,
    mm_embeds: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    num_tokens = target_token_ids.shape[0]
    batch_size = next_token_ids.shape[0]

    if last_token_indices is None:
        last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

    if self.method == "eagle3":
        assert isinstance(self.model, Eagle3LlamaForCausalLM)
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        assert target_hidden_states.shape[-1] == self.hidden_size

    # Shift the input ids by one token.
    # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
    self.input_ids[: num_tokens - 1] = target_token_ids[1:]
    # Replace the last token with the next token.
    # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
    self.input_ids[last_token_indices] = next_token_ids

    assert self.runner is not None

    ubatch_id = dbo_current_ubatch_id()
    attn_metadata_builder = self.runner.attn_groups[0][0].metadata_builders[ubatch_id]
    attn_metadata = attn_metadata_builder.build_for_drafting(
        common_attn_metadata=common_attn_metadata, draft_index=0
    )
    if (
        hasattr(attn_metadata, "decode")
        and attn_metadata.decode is not None
        and hasattr(attn_metadata.decode, "spec_num_seq_len")
        and attn_metadata.decode.spec_num_seq_len is not None
    ):
        attn_metadata.decode.spec_num_seq_len = -1

    if self.draft_indexer_metadata_builder:
        draft_indexer_metadata = self.draft_indexer_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata,
            draft_index=0,
        )
    else:
        draft_indexer_metadata = None

    # At this moment, we assume all eagle layers belong to the same KV
    # cache group, thus using the same attention metadata.
    per_layer_attn_metadata = {}
    for layer_name in self.attn_layer_names:
        per_layer_attn_metadata[layer_name] = attn_metadata
    for layer_name in self.indexer_layer_names:
        assert draft_indexer_metadata is not None
        per_layer_attn_metadata[layer_name] = draft_indexer_metadata
    if self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
        num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
    else:
        num_input_tokens = num_tokens

    # copy inputs to buffer for cudagraph
    self.positions[:num_tokens] = target_positions
    self.hidden_states[:num_tokens] = target_hidden_states
    if self.is_multimodal_model:
        input_ids = self.input_ids[:num_tokens]
        inputs_embeds = self.model.get_input_embeddings(
            input_ids,
            multimodal_embeddings=mm_embeds or None,
        )
        self.inputs_embeds[:num_tokens] = inputs_embeds
        inputs_embeds = self.inputs_embeds[:num_input_tokens]
        input_ids = None
    else:
        inputs_embeds = None
        input_ids = self.input_ids[:num_input_tokens]

    with set_forward_context(
        per_layer_attn_metadata, self.vllm_config, num_tokens=num_input_tokens
    ):
        ret_hidden_states = self.model(
            input_ids=input_ids,
            positions=self.positions[:num_input_tokens],
            hidden_states=self.hidden_states[:num_input_tokens],
            inputs_embeds=inputs_embeds,
        )
        if self.method == "mtp":
            last_hidden_states = ret_hidden_states
            hidden_states = self.hidden_states[:num_input_tokens]
        else:
            last_hidden_states, hidden_states = ret_hidden_states
    sample_hidden_states = last_hidden_states[last_token_indices]
    if self.method == "mtp":
        logits = self.model.compute_logits(sample_hidden_states, 0)
    else:
        logits = self.model.compute_logits(sample_hidden_states, None)
    positions = target_positions[last_token_indices]
    hidden_states = hidden_states[last_token_indices]

    if isinstance(attn_metadata, TreeAttentionMetadata):
        # Draft using tree attention.
        draft_token_ids_list = self.propose_tree(
            batch_size=batch_size,
            logits=logits,
            positions=positions,
            hidden_states=hidden_states,
            common_attn_metadata=common_attn_metadata,
        )
        # [batch_size, num_tree_tokens]
        return torch.cat(draft_token_ids_list, dim=1)

    draft_token_ids = logits.argmax(dim=-1)

    # Early exit if there is only one draft token to be generated.
    if self.num_speculative_tokens == 1:
        # [batch_size, 1]
        return draft_token_ids.view(-1, 1)

    # TODO: Currently, MTP module released by deepseek only has
    # one layer. Adapt this code to support multiple layers once
    # there's a multi-layer MTP module.

    # Generate the remaining draft tokens.
    draft_token_ids_list = [draft_token_ids]
    if self.use_cuda_graph and batch_size <= self.cudagraph_batch_sizes[-1]:
        input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size)
    else:
        input_batch_size = batch_size

    common_attn_metadata.num_actual_tokens = batch_size
    common_attn_metadata.max_query_len = 1
    common_attn_metadata.query_start_loc = self.arange[: batch_size + 1].to(torch.int32)
    common_attn_metadata.query_start_loc_cpu = (
        torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone().to(torch.int32)
    )

    attn_metadata_builder = self.runner.attn_groups[0][0].metadata_builder
    for token_index in range(self.num_speculative_tokens - 1):
        # Update the inputs.
        # cast to int32 is crucial when eagle model is compiled.
        # tensor.argmax() returns int64 by default.
        input_ids = draft_token_ids_list[-1].int()
        positions += 1

        # NOTE(woosuk): We should handle the case where the draft model
        # generates tokens beyond the max model length. Since it is complex
        # to remove such requests from the batch, we keep them in the batch
        # but adjust the position ids and slot mappings to avoid the
        # out-of-range access during the model execution. The draft tokens
        # generated with this adjustment should be ignored.
        exceeds_max_model_len = positions >= self.max_model_len
        # Mask out the position ids that exceed the max model length.
        # Otherwise, we may get out-of-range error in RoPE.
        clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

        # Increment the sequence lengths.
        common_attn_metadata.seq_lens += 1
        common_attn_metadata.seq_lens_cpu += 1
        # For the requests that exceed the max model length, we set the
        # sequence length to 1 to minimize their overheads in attention.
        common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
        common_attn_metadata.num_computed_tokens_cpu = (
            common_attn_metadata.seq_lens_cpu - 1
        )

        # Compute the slot mapping.
        block_numbers = clamped_positions // self.block_size
        block_ids = common_attn_metadata.block_table_tensor.gather(
            dim=1, index=block_numbers.view(-1, 1)
        )
        block_ids = block_ids.view(-1)
        common_attn_metadata.slot_mapping = (
            block_ids * self.block_size + clamped_positions % self.block_size
        )
        # Mask out the slot mappings that exceed the max model length.
        # Otherwise, the KV cache will be inadvertently updated with the
        # padding tokens.
        common_attn_metadata.slot_mapping.masked_fill_(
            exceeds_max_model_len, PADDING_SLOT_ID
        )

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=token_index + 1
        )
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        # copy inputs to buffer for cudagraph
        self.input_ids[:batch_size] = input_ids
        self.positions[:batch_size] = clamped_positions
        self.hidden_states[:batch_size] = hidden_states
        if self.is_multimodal_model:
            inputs_embeds = self.model.get_input_embeddings(input_ids)
            self.inputs_embeds[:batch_size] = inputs_embeds
            inputs_embeds = self.inputs_embeds[:input_batch_size]
            input_ids = None
        else:
            inputs_embeds = None
            input_ids = self.input_ids[:input_batch_size]

        # Run the model.
        with set_forward_context(
            per_layer_attn_metadata, self.vllm_config, num_tokens=input_batch_size
        ):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=self.positions[:input_batch_size],
                hidden_states=self.hidden_states[:input_batch_size],
                inputs_embeds=inputs_embeds,
            )
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        hidden_states = hidden_states[:batch_size]
        logits = self.model.compute_logits(last_hidden_states[:batch_size])
        draft_token_ids = logits.argmax(dim=-1)
        draft_token_ids_list.append(draft_token_ids)

    # [batch_size, num_speculative_tokens]
    draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
    return draft_token_ids


def prepare_next_token_ids_padded(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    sampled_token_ids: torch.Tensor,
    requests: dict[str, CachedRequestState],
    gpu_input_batch: InputBatch,
    discard_request_indices: torch.Tensor,
    num_discarded_requests: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function is used to prepare the inputs for speculative decoding.
    It calculates the next token ids and the number of valid sampled tokens
    for each request, considering the "discarded" requests whose next token
    is not sampled and comes from `request.get_token_id()` instead.
    It also accounts for the rejected tokens in `sampled_token_ids`.
    This function must use device functions to operate on the inputs, and
    should not introduce any blocking CPU-GPU synchronization.
    """
    # TODO(Ben): Combine this into a custom fused kernel

    # Precompute get_token_id for when there is no valid next token
    num_reqs = gpu_input_batch.num_reqs
    self.backup_next_token_ids.np[:num_reqs] = np.array(
        [
            requests[gpu_input_batch.req_ids[i]].get_token_id(
                common_attn_metadata.seq_lens_cpu[i].item()
            )
            for i in range(num_reqs)
        ]
    )
    self.backup_next_token_ids.copy_to_gpu(num_reqs)

    # Mask out the sampled tokens indices that should not be sampled.
    discard_sampled_tokens_req_indices = discard_request_indices[
        :num_discarded_requests
    ]

    valid_sampled_token_ids_gpu = sampled_token_ids.clone()
    # valid_sampled_token_ids_gpu.index_fill_(
    #     0, discard_sampled_tokens_req_indices, -1)
    # ---- FIX START ----
    # XPU/XMLIR index_fill_ does NOT accept empty index tensor.
    if num_discarded_requests > 0:
        # make sure index is on same device and is int64
        idx = discard_sampled_tokens_req_indices
        if idx.device != valid_sampled_token_ids_gpu.device:
            idx = idx.to(valid_sampled_token_ids_gpu.device, non_blocking=True)
        if idx.dtype != torch.long:
            idx = idx.to(torch.long)
        if idx.numel() > 0:
            valid_sampled_token_ids_gpu.index_fill_(0, idx, -1)
    # ---- FIX END ----
    # Generate a mask for all valid tokens within those requests
    max_gen_len = sampled_token_ids.shape[-1]
    if max_gen_len == 1:
        valid_mask = torch.ones_like(valid_sampled_token_ids_gpu, dtype=torch.bool)
    else:
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
        )

    # Count the number of valid tokens in each request
    valid_sampled_tokens_count = valid_mask.sum(dim=1)

    # Get the rightmost valid index per row
    last_valid_indices = valid_sampled_tokens_count - 1
    last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

    # Get last valid token from each row
    # (assume undefined state where there is no valid token)
    selected_tokens = torch.gather(
        valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
    ).squeeze(1)

    # Use last token if valid, pre-computed backup if not
    batch_size = valid_sampled_token_ids_gpu.shape[0]
    next_token_ids = torch.where(
        last_valid_indices != -1,
        selected_tokens,
        self.backup_next_token_ids.gpu[:batch_size],
    )

    return next_token_ids, valid_sampled_tokens_count


EagleProposer.propose = propose
EagleProposer.prepare_next_token_ids_padded = prepare_next_token_ids_padded
