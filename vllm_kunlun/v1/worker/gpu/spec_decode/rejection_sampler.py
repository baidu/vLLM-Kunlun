# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import vllm.v1.worker.gpu.spec_decode.rejection_sampler as _up
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_kunlun.v1.worker.gpu._kernels import TorchKernel
from vllm_kunlun.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)


def _flatten_sampled(
    flat, sampled, sampled_stride, num_sampled, cu_num_logits, **kwargs
):
    torch.ops.xspeedgate_ops.flatten_sampled_kernel(
        flat, sampled, num_sampled, cu_num_logits, sampled.shape[0], sampled_stride
    )


def _call(
    self,
    logits: torch.Tensor,
    input_batch: InputBatch,
    draft_logits: torch.Tensor | None = None,
) -> SamplerOutput:
    # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
    # that num_nans is computed before applying penalties and temperature.
    num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

    draft_sampled = input_batch.input_ids[input_batch.logits_indices]
    pos = input_batch.positions[input_batch.logits_indices]
    processed_logits = self.sampler.apply_sampling_params(
        logits,
        input_batch.expanded_idx_mapping,
        input_batch.idx_mapping_np,
        pos,
        draft_sampled,
        input_batch.expanded_local_pos,
    )
    sampled, num_sampled = rejection_sample(
        processed_logits,
        draft_logits,
        draft_sampled,
        input_batch.cu_num_logits,
        pos,
        input_batch.idx_mapping,
        input_batch.expanded_idx_mapping,
        input_batch.expanded_local_pos,
        self.sampler.sampling_states.temperature.gpu,
        self.sampler.sampling_states.seeds.gpu,
        self.num_speculative_steps,
        self.synthetic_conditional_rates,
        use_fp64=self.sampler.use_fp64_gumbel,
        use_block_verification=self.use_block_verification,
        all_greedy=bool(
            (
                self.sampler.sampling_states.temperature.np[input_batch.idx_mapping_np]
                == 0
            ).all()
        ),
    )
    logprobs_tensors = self._get_logprobs_tensors(
        input_batch,
        sampled,
        num_sampled,
        (
            processed_logits
            if self.sampler.logprobs_mode == "processed_logprobs"
            else logits
        ),
    )

    num_sampled, num_rejected = get_num_sampled_and_rejected(
        num_sampled,
        input_batch.seq_lens,
        input_batch.cu_num_logits,
        input_batch.idx_mapping,
        self.sampler.req_states.prefill_len.gpu,
    )

    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=logprobs_tensors,
        num_nans=num_nans,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )


_up._flatten_sampled_kernel = TorchKernel(_flatten_sampled)
_up.rejection_sample = rejection_sample
_up.RejectionSampler.__call__ = _call
