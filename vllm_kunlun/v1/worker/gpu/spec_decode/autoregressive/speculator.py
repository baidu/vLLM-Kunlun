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


_up.prepare_prefill_inputs = prepare_prefill_inputs
