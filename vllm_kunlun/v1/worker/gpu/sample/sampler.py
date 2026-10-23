# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun sampling and metadata updates for Model Runner V2."""

import numpy as np
import torch
import vllm.v1.worker.gpu.sample.sampler as _up
from vllm_kunlun.v1.sample.ops.topk_topp_sampler import flashinfer_sample

_original_init = _up.Sampler.__init__
_original_add_request = _up.Sampler.add_request
_original_apply_staged_writes = _up.Sampler.apply_staged_writes
_original_sample = _up.Sampler.sample


def _init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self._sampling_dirty = True


def _add_request(self, *args, **kwargs):
    self._sampling_dirty = True
    _original_add_request(self, *args, **kwargs)


def _apply_staged_writes(self):
    # Token counts and per-step penalties are updated outside this method.
    if self._sampling_dirty:
        _original_apply_staged_writes(self)
        self._sampling_dirty = False


def _filter_top_k_top_p(logits, k, p, max_top_k):
    if (
        k is None
        or max_top_k >= logits.shape[-1]
        or max_top_k > 1024
        or torch.cuda.is_current_stream_capturing()
    ):
        return _up.apply_top_k_top_p(logits, k, p)
    # Extra candidates include ordinary ties at the top-k cutoff.
    num_candidates = min(max_top_k + 32, logits.shape[-1] - 1)
    values, indices = logits.topk(num_candidates + 1, dim=-1)
    cutoff = values.gather(1, (k.long() - 1).unsqueeze(1))
    if p is None:
        return logits.masked_fill(logits < cutoff, float("-inf"))

    # Fall back if tied candidates extend beyond the compact buffer.
    ambiguous = (values[:, -1] >= cutoff.squeeze(1)) | (
        ~torch.isfinite(cutoff)
    ).squeeze(1)
    values = values[:, :num_candidates].flip(-1)
    indices = indices[:, :num_candidates].flip(-1)
    values = values.masked_fill(values < cutoff, float("-inf"))
    cumulative = values.softmax(-1).cumsum(-1)
    threshold = 1 - p.unsqueeze(1)
    ambiguous |= (((cumulative - threshold).abs() < 1e-6) & torch.isfinite(values)).any(
        -1
    )
    mask = cumulative <= threshold
    mask[:, -1] = False
    ambiguous |= (
        (values[:, :-1] == values[:, 1:]) & (mask[:, :-1] != mask[:, 1:])
    ).any(-1)
    values = values.masked_fill(mask, float("-inf"))
    # Synchronize one small flag array; sort only rows with ambiguous cutoffs.
    fallback = np.flatnonzero(ambiguous.cpu().numpy())
    if len(fallback) == logits.shape[0]:
        return _up.apply_top_k_top_p(logits, k, p)
    result = torch.full_like(logits, float("-inf")).scatter_(1, indices, values)
    if len(fallback):
        rows = torch.as_tensor(fallback, device=logits.device)
        result[rows] = _up.apply_top_k_top_p(logits[rows], k[rows], p[rows])
    return result


def sample(
    self,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    return_logprobs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = self.sampling_states
    # Retain the negative-request compatibility path used by dummy batches.
    if np.any(idx_mapping_np < 0):
        return _original_sample(
            self,
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            return_logprobs=return_logprobs,
        )
    processed_logits = self.apply_sampling_params(
        logits,
        expanded_idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        skip_top_k_top_p=True,
    )
    top_k, top_p = states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
    if np.all(states.temperature.np[idx_mapping_np] == 0):
        processed_logits = _up.apply_top_k_top_p(processed_logits, top_k, top_p)
        return processed_logits.argmax(dim=-1), processed_logits

    use_native = not (
        self.use_fp64_gumbel
        or (return_logprobs and self.logprobs_mode == "processed_logprobs")
        or states.any_greedy(idx_mapping_np)
        or states.any_explicit_seed(idx_mapping_np)
    )
    if use_native and (top_k is not None or top_p is not None):
        sampled = flashinfer_sample(processed_logits.contiguous(), top_k, top_p, {}).to(
            torch.int64
        )
    else:
        max_top_k = int(states.top_k.np[idx_mapping_np].max())
        processed_logits = _filter_top_k_top_p(
            processed_logits, top_k, top_p, max_top_k
        )
        # CPU request metadata guarantees valid indices here. The general
        # Gumbel wrapper retains its argmax fix for negative request IDs.
        sampled = torch.ops.xspeedgate_ops.gumbel_sample(
            processed_logits,
            expanded_idx_mapping,
            states.temperature.gpu,
            states.seeds.gpu,
            pos,
            False,
            None,
            None,
            self.use_fp64_gumbel,
        ).squeeze(-1)
    return sampled, processed_logits


_up.Sampler.__init__ = _init
_up.Sampler.add_request = _add_request
_up.Sampler.apply_staged_writes = _apply_staged_writes
_up.Sampler.sample = sample
