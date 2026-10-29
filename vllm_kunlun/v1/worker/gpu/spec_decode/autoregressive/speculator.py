# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep official MRV2 orchestration; replace its three Triton helpers."""

import torch
import vllm.v1.worker.gpu.spec_decode.autoregressive.speculator as _up


def prepare_prefill_inputs(
    last_token_indices,
    current_draft_step,
    input_buffers,
    input_batch,
    num_sampled,
    num_rejected,
    last_sampled,
    next_prefill_tokens,
    max_num_reqs,
):
    torch.ops.xspeedgate_ops.prepare_prefill_inputs_kernel(
        last_token_indices,
        current_draft_step,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.input_ids,
        input_batch.positions,
        input_batch.idx_mapping,
        last_sampled.view(-1),
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        input_batch.num_reqs,
        max_num_reqs,
    )
    return last_token_indices


def prepare_decode_inputs(
    draft_tokens,
    target_seq_lens,
    num_rejected,
    input_buffers,
    max_model_len,
    max_num_reqs,
    advance_draft_positions=True,
):
    torch.ops.xspeedgate_ops.prepare_decode_inputs_kernel(
        draft_tokens,
        target_seq_lens,
        num_rejected,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        draft_tokens.numel(),
        max_num_reqs,
        max_model_len,
        draft_tokens.stride(0),
        advance_draft_positions,
    )


def update_draft_inputs(
    draft_tokens,
    current_draft_step,
    hidden_states,
    output_draft_tokens,
    next_input_hidden_states,
    input_buffers,
    num_reqs,
    max_model_len,
    num_speculative_steps,
    advance_draft_positions=True,
):
    torch.ops.xspeedgate_ops.update_draft_inputs_kernel(
        draft_tokens,
        current_draft_step,
        hidden_states,
        output_draft_tokens,
        next_input_hidden_states,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.seq_lens,
        num_reqs,
        max_model_len,
        num_speculative_steps,
        output_draft_tokens.stride(0),
        hidden_states.shape[1] * hidden_states.element_size(),
        hidden_states.stride(0) * hidden_states.element_size(),
        next_input_hidden_states.stride(0) * next_input_hidden_states.element_size(),
        advance_draft_positions,
    )


_up.prepare_prefill_inputs = prepare_prefill_inputs
_up.prepare_decode_inputs = prepare_decode_inputs
_up.update_draft_inputs = update_draft_inputs
