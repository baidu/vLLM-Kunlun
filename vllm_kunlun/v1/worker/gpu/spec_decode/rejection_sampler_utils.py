# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as _up


def rejection_sample(
    target_logits,
    draft_logits,
    draft_sampled,
    cu_num_logits,
    pos,
    idx_mapping,
    expanded_idx_mapping,
    expanded_local_pos,
    temperature,
    seed,
    num_speculative_steps,
    synthetic_conditional_rates=None,
    use_fp64=False,
    use_block_verification=False,
    *,
    all_greedy=None,
):
    if synthetic_conditional_rates is not None or use_block_verification:
        raise NotImplementedError("Kunlun MRV2 MTP supports standard rejection only")
    if not 1 <= num_speculative_steps <= 32:
        raise ValueError(
            "Kunlun MTP rejection sampling requires 1 to 32 speculative steps"
        )
    if all_greedy is None:
        # Standalone callers only. The MRV2 runner supplies CPU sampling metadata.
        all_greedy = bool((temperature[idx_mapping.long()] == 0).all().item())
    return torch.ops.xspeedgate_ops.rejection_sample_fused_kernel(
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps,
        all_greedy,
        use_fp64,
        0,
    )


_up.rejection_sample = rejection_sample
