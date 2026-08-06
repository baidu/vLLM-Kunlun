# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
from vllm.v1.spec_decode.eagle import EagleProposer


class DFlashProposer(EagleProposer):
    """Minimal DFlash proposer backport for vLLM 0.15.1.

    This keeps the DFlash method on the EAGLE-style speculative decoding path
    while avoiding the upstream Triton DFlash input expansion kernel. The full
    DFlash parallel-drafting path can be layered on top once the 0.15.1 runner
    has all upstream #36847 state fields.
    """

    def __init__(self, vllm_config, device: torch.device, runner=None):
        super().__init__(vllm_config, device, runner=runner)
        self.parallel_drafting_hidden_state_tensor = None

    def _raise_if_multimodal(self):
        # DFlash targets Qwen3/Qwen3.5 style models. Keep multimodal checks
        # permissive to match upstream DFlash; actual multimodal behavior is
        # still guarded by model support.
        pass

    def model_returns_tuple(self) -> bool:
        return True

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            return dflash_config.get("use_aux_hidden_state", True)
        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is not None:
            return eagle_config.get("use_aux_hidden_state", True)
        return True


def copy_and_expand_dflash_inputs_native(
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    num_speculative_tokens: int,
    parallel_drafting_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native Torch fallback for DFlash query input expansion.

    Returns (input_ids, query_positions, query_slot_mapping) for a batch with
    one bonus token followed by num_speculative_tokens mask tokens per request.
    """
    batch_size = next_token_ids.shape[0]
    num_query_per_req = num_speculative_tokens + 1
    device = next_token_ids.device
    offsets = torch.arange(
        num_query_per_req, device=device, dtype=target_positions.dtype
    )
    last_indices = query_start_loc[1:].to(torch.long) - 1
    last_positions = target_positions[last_indices]
    query_positions = (last_positions[:, None] + 1 + offsets[None, :]).reshape(-1)
    input_ids = torch.full(
        (batch_size, num_query_per_req),
        int(parallel_drafting_token_id),
        dtype=next_token_ids.dtype,
        device=device,
    )
    input_ids[:, 0] = next_token_ids
    input_ids = input_ids.reshape(-1)
    block_numbers = query_positions // block_size
    block_ids = block_table.gather(1, block_numbers.to(torch.long).view(batch_size, -1))
    query_slot_mapping = (
        block_ids * block_size + query_positions.view(batch_size, -1) % block_size
    ).reshape(-1)
    return input_ids, query_positions, query_slot_mapping
