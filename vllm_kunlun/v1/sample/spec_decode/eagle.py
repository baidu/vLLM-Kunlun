# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

_orig_prepare_next_token_ids_padded = EagleProposer.prepare_next_token_ids_padded
_orig_prepare_inputs_padded = EagleProposer.prepare_inputs_padded
_orig_eagle_init = EagleProposer.__init__


def _is_qwen35_mtp(self) -> bool:
    if getattr(self, "method", None) == "qwen3_next_mtp":
        return True
    hf_config = getattr(getattr(self, "draft_model_config", None), "hf_config", None)
    return getattr(hf_config, "model_type", None) == "qwen3_5_mtp"


def prepare_next_token_ids_padded(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    sampled_token_ids: torch.Tensor,
    requests: dict[str, CachedRequestState],
    gpu_input_batch: InputBatch,
    discard_request_mask: torch.Tensor,
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
    if not _is_qwen35_mtp(self):
        return _orig_prepare_next_token_ids_padded(
            self,
            common_attn_metadata,
            sampled_token_ids,
            requests,
            gpu_input_batch,
            discard_request_mask,
        )

    # Precompute get_token_id for when there is no valid next token
    num_reqs = gpu_input_batch.num_reqs
    self.backup_next_token_ids.np[:num_reqs] = np.array(
        [
            requests[gpu_input_batch.req_ids[i]].get_token_id(
                common_attn_metadata.seq_lens_cpu[i].item()
            )
            for i in range(num_reqs)
        ],
        dtype=np.int32,
    )
    self.backup_next_token_ids.copy_to_gpu(num_reqs)

    # Mask out the sampled tokens indices that should not be sampled.
    discard_sampled_tokens_req_indices = torch.nonzero(
        discard_request_mask[:num_reqs], as_tuple=False
    ).flatten()

    valid_sampled_token_ids_gpu = sampled_token_ids.clone()
    # valid_sampled_token_ids_gpu.index_fill_(
    #     0, discard_sampled_tokens_req_indices, -1)
    # ---- FIX START ----
    # XPU/XMLIR index_fill_ does NOT accept empty index tensor.
    if discard_sampled_tokens_req_indices.numel() > 0:
        # make sure index is on same device and is int64
        idx = discard_sampled_tokens_req_indices
        if idx.device != valid_sampled_token_ids_gpu.device:
            idx = idx.to(valid_sampled_token_ids_gpu.device, non_blocking=True)
        if idx.dtype != torch.long:
            idx = idx.to(torch.long)
        valid_sampled_token_ids_gpu.index_fill_(0, idx, -1)
    # ---- FIX END ----
    # Generate a mask for all valid tokens within those requests
    valid_mask = (valid_sampled_token_ids_gpu != -1) & (
        valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
    )

    # Count the number of valid tokens in each request
    valid_sampled_tokens_count_long = valid_mask.sum(dim=1)
    valid_sampled_tokens_count = valid_sampled_tokens_count_long.to(torch.int32)

    # Get the rightmost valid index per row
    last_valid_indices = valid_sampled_tokens_count_long - 1
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


def prepare_inputs_padded(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    spec_decode_metadata,
    valid_sampled_tokens_count: torch.Tensor,
):
    """XPU fallback for upstream's Triton prepare-inputs kernel."""
    if not _is_qwen35_mtp(self):
        return _orig_prepare_inputs_padded(
            self,
            common_attn_metadata,
            spec_decode_metadata,
            valid_sampled_tokens_count,
        )

    num_reqs = common_attn_metadata.num_reqs

    cu_num_draft = spec_decode_metadata.cu_num_draft_tokens
    num_draft = cu_num_draft.clone()
    if num_reqs > 1:
        num_draft[1:] = cu_num_draft[1:] - cu_num_draft[:-1]

    valid_count = valid_sampled_tokens_count.to(num_draft.dtype)
    num_rejected = num_draft + 1 - valid_count
    num_rejected = torch.where(
        num_draft > 0, num_rejected, torch.zeros_like(num_rejected)
    )

    q_last_tok_idx = common_attn_metadata.query_start_loc[1:] - 1
    token_indices_to_sample = (q_last_tok_idx - num_rejected).to(torch.int32)
    num_rejected_tokens_gpu = num_rejected.to(torch.int32)

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    total_num_tokens = query_start_loc_cpu[-1].item()

    spec_common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=common_attn_metadata.query_start_loc,
        seq_lens=common_attn_metadata.seq_lens,
        query_start_loc_cpu=query_start_loc_cpu,
        _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
        _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
        num_reqs=common_attn_metadata.num_reqs,
        num_actual_tokens=total_num_tokens,
        max_query_len=new_query_len_per_req.max().item(),
        max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
        block_table_tensor=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
        causal=True,
        dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
    )

    return (
        spec_common_attn_metadata,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
    )


def _patched_eagle_init(self, *args, **kwargs):
    _orig_eagle_init(self, *args, **kwargs)
    if (
        _is_qwen35_mtp(self)
        and getattr(self, "uses_mrope", False)
        and not hasattr(self, "positions")
    ):
        self.positions = self.mrope_positions


EagleProposer.prepare_next_token_ids_padded = prepare_next_token_ids_padded
EagleProposer.prepare_inputs_padded = prepare_inputs_padded
EagleProposer.__init__ = _patched_eagle_init
