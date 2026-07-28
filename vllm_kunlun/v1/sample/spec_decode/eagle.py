# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun overrides for the EAGLE / MTP speculative-decode proposer."""

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)

PADDING_SLOT_ID = -1

_orig_prepare_next_token_ids_padded = EagleProposer.prepare_next_token_ids_padded
_orig_prepare_inputs_padded = EagleProposer.prepare_inputs_padded
_orig_update_positions_dependent_metadata = (
    EagleProposer._update_positions_dependent_metadata
)
_orig_eagle_init = EagleProposer.__init__


def _is_qwen35_mtp(self) -> bool:
    # 上游会把 draft 的 model_type 归一化：qwen3_next -> "qwen3_next_mtp"，
    # qwen3_5/qwen3_5_moe -> "qwen3_5_mtp"（vllm/config/speculative.py）。同时
    # self.method 会被统一改写成 "mtp"，故不能用它来识别具体模型。
    hf_config = getattr(getattr(self, "draft_model_config", None), "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    return model_type in ("qwen3_5_mtp", "qwen3_next_mtp")


def prepare_next_token_ids_padded(
    self,
    sampled_token_ids: torch.Tensor,
    requests: dict[str, CachedRequestState],
    gpu_input_batch: InputBatch,
    discard_request_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch-native replacement (Qwen3.5-MTP) for the Triton kernel
    ``eagle_prepare_next_token_padded_kernel``。
    """
    if not _is_qwen35_mtp(self):
        return _orig_prepare_next_token_ids_padded(
            self,
            sampled_token_ids,
            requests,
            gpu_input_batch,
            discard_request_mask,
        )

    # Precompute backup token ids
    num_reqs = gpu_input_batch.num_reqs
    self.backup_next_token_ids.np[:num_reqs] = np.array(
        [
            requests[gpu_input_batch.req_ids[i]].get_token_id(
                gpu_input_batch.num_tokens_no_spec[i] - 1
            )
            for i in range(num_reqs)
        ],
        dtype=np.int32,
    )
    self.backup_next_token_ids.copy_to_gpu(num_reqs)

    # Mask out discarded requests' sampled tokens.
    discard_sampled_tokens_req_indices = torch.nonzero(
        discard_request_mask[:num_reqs], as_tuple=False
    ).flatten()

    valid_sampled_token_ids_gpu = sampled_token_ids.clone()

    if discard_sampled_tokens_req_indices.numel() > 0:
        idx = discard_sampled_tokens_req_indices
        if idx.device != valid_sampled_token_ids_gpu.device:
            idx = idx.to(valid_sampled_token_ids_gpu.device, non_blocking=True)
        if idx.dtype != torch.long:
            idx = idx.to(torch.long)
        valid_sampled_token_ids_gpu.index_fill_(0, idx, -1)

    valid_mask = (valid_sampled_token_ids_gpu != -1) & (
        valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
    )
    valid_sampled_tokens_count_long = valid_mask.sum(dim=1)
    valid_sampled_tokens_count = valid_sampled_tokens_count_long.to(torch.int32)

    last_valid_indices = valid_sampled_tokens_count_long - 1
    last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)
    selected_tokens = torch.gather(
        valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
    ).squeeze(1)

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
    spec_decode_metadata: SpecDecodeMetadata,
    valid_sampled_tokens_count: torch.Tensor,
) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
    """Torch-native replacement (仅 Qwen3.5-MTP) for the Triton kernel
    ``eagle_prepare_inputs_padded_kernel``"""
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


def _update_positions_dependent_metadata(
    self,
    positions: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,
    batch_size: int,
    input_batch_size: int,
    block_size: int,
) -> torch.Tensor:
    """Torch-native replacement (仅 Qwen3.5-MTP) for the Triton kernel
    ``eagle_step_slot_mapping_metadata_kernel``。
    """
    if not _is_qwen35_mtp(self):
        return _orig_update_positions_dependent_metadata(
            self,
            positions,
            common_attn_metadata,
            batch_size,
            input_batch_size,
            block_size,
        )

    cad = common_attn_metadata
    positions_1d = positions[0] if self.uses_mrope else positions

    new_position = positions_1d + 1
    exceeds_max = new_position >= self.max_model_len
    clamped_position = torch.where(
        exceeds_max, torch.zeros_like(new_position), new_position
    )

    n_blocks_per_req = cad.block_table_tensor.shape[1]
    block_number = (clamped_position // block_size).clamp(max=n_blocks_per_req - 1)
    block_id = cad.block_table_tensor.gather(
        dim=1, index=block_number.view(-1, 1)
    ).view(-1)
    slot = block_id * block_size + (clamped_position % block_size)
    slot = torch.where(exceeds_max, torch.full_like(slot, PADDING_SLOT_ID), slot)

    new_seq_lens = torch.where(
        exceeds_max, torch.ones_like(cad.seq_lens), cad.seq_lens + 1
    ).clamp(max=self.max_model_len)
    cad.seq_lens.copy_(new_seq_lens)

    if self.uses_mrope:
        out_pos = self.mrope_positions[0, :batch_size]
    elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
        out_pos = self.xdrope_positions[0, :batch_size]
    else:
        out_pos = self.positions[:batch_size]
    out_pos.copy_(clamped_position)

    self._slot_mapping_buffer[:batch_size].copy_(slot)
    if input_batch_size > batch_size:
        self._slot_mapping_buffer[batch_size:input_batch_size].fill_(PADDING_SLOT_ID)
    cad.slot_mapping = self._slot_mapping_buffer[:batch_size]

    if self.uses_mrope:
        self.mrope_positions[1:, :batch_size] = self.mrope_positions[0, :batch_size]
        positions = self.mrope_positions[:, :batch_size]
    elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
        self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[0, :batch_size]
        positions = self.xdrope_positions[0, :batch_size]
    else:
        positions = self.positions[:batch_size]

    cad.max_seq_len = min(cad.max_seq_len + 1, self.max_model_len)

    if cad._seq_lens_cpu is not None:
        cad._seq_lens_cpu += 1
    if cad._num_computed_tokens_cpu is not None:
        cad._num_computed_tokens_cpu += 1
    if cad.seq_lens_cpu_upper_bound is not None:
        cad.seq_lens_cpu_upper_bound += 1

    return positions


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
EagleProposer._update_positions_dependent_metadata = (
    _update_positions_dependent_metadata
)
EagleProposer.__init__ = _patched_eagle_init
