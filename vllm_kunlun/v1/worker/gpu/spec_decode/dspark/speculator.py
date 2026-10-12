# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun compact Markov sampling for the upstream DSpark runner."""

import logging

import torch
import vllm.v1.worker.gpu.spec_decode.dspark.speculator as _up

from vllm_kunlun.platforms import envs
from vllm_kunlun.v1.worker.gpu.spec_decode.dspark.utils import configured_topk

_LOG = logging.getLogger("vllm_kunlun")
_REPORTED_XSG = False
_REPORTED_TP_TOPK = False
_original_sample_sequential = _up.DSparkSpeculator._sample_sequential


def _compute_tp_local_topk(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact base-logit top-k without gathering the full vocabulary."""
    from vllm.distributed.communication_op import tensor_model_parallel_all_gather
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

    logits_processor = model.logits_processor
    lm_head = model.lm_head
    if getattr(lm_head, "num_added_embeddings", 0) != 0:
        raise RuntimeError(
            "Kunlun DSpark TP-local top-k does not support added vocabulary rows"
        )

    local_logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=None)
    soft_cap = getattr(logits_processor, "soft_cap", None)
    if soft_cap is not None:
        local_logits = torch.tanh(local_logits / soft_cap) * soft_cap
    scale = float(getattr(logits_processor, "scale", 1.0))
    if scale != 1.0:
        local_logits = local_logits * scale

    num_pad = int(lm_head.shard_indices.num_org_vocab_padding)
    valid_local_vocab = local_logits.shape[-1] - num_pad
    if topk > valid_local_vocab:
        raise ValueError(
            f"DSpark top-k {topk} exceeds local valid vocab {valid_local_vocab}"
        )
    if num_pad:
        local_logits[..., -num_pad:] = -float("inf")

    local_values, local_indices = local_logits.topk(topk, dim=-1)
    vocab_start = int(lm_head.shard_indices.org_vocab_start_index)
    global_indices = local_indices + vocab_start

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return local_values, global_indices

    # One small collective replaces the full-vocabulary all-gather. Token IDs
    # are exactly representable in FP32 for the supported vocabulary range.
    if int(lm_head.org_vocab_size) >= 2**24:
        raise RuntimeError("DSpark TP-local top-k requires vocab_size < 2**24")
    # Match the graph-safe layout used by upstream LogitsProcessor.get_top_tokens:
    # every candidate is an adjacent (value, id) pair and the collective only
    # concatenates the contiguous last dimension.
    local_pairs = torch.stack((local_values.float(), global_indices.float()), dim=-1)
    gathered_pairs = tensor_model_parallel_all_gather(local_pairs, dim=-1)
    gathered_pairs = gathered_pairs.view(hidden_states.shape[0], topk, tp_size, 2)
    gathered_values = (
        gathered_pairs[..., 0]
        .transpose(1, 2)
        .reshape(hidden_states.shape[0], tp_size * topk)
        .to(local_values.dtype)
    )
    gathered_indices = (
        gathered_pairs[..., 1]
        .transpose(1, 2)
        .reshape(hidden_states.shape[0], tp_size * topk)
        .to(torch.long)
    )
    global_values, merge_positions = gathered_values.topk(topk, dim=-1)
    global_indices = gathered_indices.gather(-1, merge_positions)
    return global_values, global_indices


def sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
    """Run compact greedy DSpark sampling; called by an import-hook wrapper."""
    global _REPORTED_TP_TOPK, _REPORTED_XSG
    topk = configured_topk()
    if topk is None:
        raise RuntimeError("DSpark top-k dispatch called without a configured top-k")
    if self.draft_logits is not None:
        raise RuntimeError(
            "Kunlun compact DSpark top-k currently supports greedy draft only"
        )

    n_spec = self.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[self.sample_indices[:num_sample]]
    vocab_size = int(self.model.config.draft_vocab_size)
    if topk > vocab_size:
        raise ValueError(f"DSpark top-k {topk} exceeds draft vocab {vocab_size}")

    if envs.VLLM_KUNLUN_DSPARK_TP_LOCAL_TOPK:
        base_values, candidate_ids = _compute_tp_local_topk(
            self.model, sample_hidden, topk
        )
        base_values = base_values.view(num_reqs, n_spec, topk)
        candidate_ids = candidate_ids.view(num_reqs, n_spec, topk)
        if not _REPORTED_TP_TOPK:
            _LOG.info(
                "[KunlunPlugin] DSpark TP-local LM-head top-k enabled: " "topk=%d",
                topk,
            )
            _REPORTED_TP_TOPK = True
    else:
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
        base_values, candidate_ids = base_logits.topk(topk, dim=-1)

    head = self.model.model.markov_head
    w1 = getattr(head, "_kunlun_markov_w1_full", None)
    w2 = getattr(head, "_kunlun_markov_w2_full", None)
    if w1 is None or w2 is None:
        raise RuntimeError(
            "DSpark top-k Markov weights were not replicated at load time"
        )

    anchor_ids = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
    scale = float(getattr(self.model.logits_processor, "scale", 1.0))

    if self.model.draft_id_to_target_id is not None:
        raise RuntimeError(
            "Kunlun fused DSpark top-k requires identical draft/target vocab"
        )
    sampled = self.draft_tokens[:num_reqs, :n_spec]
    torch.ops.xspeedgate_ops.dspark_markov_topk_greedy_out(
        sampled, base_values, candidate_ids, anchor_ids, w1, w2, scale
    )
    if not _REPORTED_XSG:
        _LOG.info(
            "[KunlunPlugin] DSpark fused XSpeedGate top-k enabled: topk=%d steps=%d",
            topk,
            n_spec,
        )
        _REPORTED_XSG = True


def _sample_sequential(self, num_reqs, head_hidden):
    if configured_topk() is not None and self.draft_logits is None:
        return sample_sequential_topk(self, num_reqs, head_hidden)
    return _original_sample_sequential(self, num_reqs, head_hidden)


_up.DSparkSpeculator._sample_sequential = _sample_sequential
