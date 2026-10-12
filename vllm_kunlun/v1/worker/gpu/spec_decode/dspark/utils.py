# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Share the target LM head and prepare Kunlun DSpark sampling weights."""

import logging

import torch
import vllm.v1.worker.gpu.spec_decode.dspark.utils as _up

from vllm_kunlun.platforms import envs

_LOG = logging.getLogger("vllm_kunlun")
_original_load_dspark_model = _up.load_dspark_model


def configured_topk() -> int | None:
    topk = envs.VLLM_KUNLUN_DSPARK_TOPK
    if topk is not None and topk <= 0:
        raise ValueError("VLLM_KUNLUN_DSPARK_TOPK must be positive")
    return topk


def replicate_markov_weights(draft_model: torch.nn.Module) -> None:
    """Replicate the TP-sharded FP16 Markov factors on every TP rank."""
    topk = configured_topk()
    if topk is None:
        return

    head = draft_model.model.markov_head
    if hasattr(head, "_kunlun_markov_w1_full"):
        return

    w1 = head.markov_w1.weight.detach()
    w2 = head.markov_w2.weight.detach()
    if w1.dtype != torch.float16 or w2.dtype != torch.float16:
        raise RuntimeError(
            "Kunlun DSpark top-k requires FP16 Markov weights; "
            f"received {w1.dtype} and {w2.dtype}"
        )
    if w1.ndim != 2 or w2.ndim != 2 or w1.shape[1] != w2.shape[1]:
        raise RuntimeError(
            f"Invalid DSpark Markov weights: w1={tuple(w1.shape)}, "
            f"w2={tuple(w2.shape)}"
        )

    from vllm.distributed.communication_op import tensor_model_parallel_all_gather
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1:
        full_w1 = tensor_model_parallel_all_gather(w1, dim=0).contiguous()
        full_w2 = tensor_model_parallel_all_gather(w2, dim=0).contiguous()
    else:
        full_w1 = w1.contiguous()
        full_w2 = w2.contiguous()

    vocab_size = int(getattr(draft_model.config, "vocab_size"))
    draft_vocab_size = int(
        getattr(draft_model.config, "draft_vocab_size", None) or vocab_size
    )
    if full_w1.shape[0] < vocab_size or full_w2.shape[0] < draft_vocab_size:
        raise RuntimeError(
            "Gathered DSpark Markov weights are smaller than the logical vocab: "
            f"w1={tuple(full_w1.shape)}, w2={tuple(full_w2.shape)}, "
            f"vocab={vocab_size}, draft_vocab={draft_vocab_size}"
        )
    # Drop vocabulary padding. Current Qwen3.5 has none, but keeping logical
    # shapes makes candidate IDs and model mappings unambiguous.
    head._kunlun_markov_w1_full = full_w1[:vocab_size]
    head._kunlun_markov_w2_full = full_w2[:draft_vocab_size]
    head._kunlun_markov_topk = topk
    mib = (
        (head._kunlun_markov_w1_full.numel() + head._kunlun_markov_w2_full.numel())
        * 2
        / (1024**2)
    )
    _LOG.info(
        "[KunlunPlugin] Replicated FP16 DSpark Markov weights for top-k=%d: "
        "w1=%s w2=%s tp=%d extra=%.1f MiB",
        topk,
        tuple(head._kunlun_markov_w1_full.shape),
        tuple(head._kunlun_markov_w2_full.shape),
        tp_size,
        mib,
    )


def load_dspark_model(target_model, vllm_config):
    draft = _original_load_dspark_model(target_model, vllm_config)

    # Multimodal wrappers keep the LM head on their inner language model.
    language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    head = getattr(target_model, "lm_head", None)
    if head is None:
        head = getattr(language_model, "lm_head", None)
    if head is None and not getattr(draft, "has_own_lm_head", False):
        raise ValueError(
            "DSpark checkpoint has no LM head and target head is unavailable"
        )

    target_vocab = vllm_config.model_config.get_vocab_size()
    draft_config = vllm_config.speculative_config.draft_model_config
    draft_vocab = (
        getattr(draft_config.hf_config, "draft_vocab_size", None)
        or draft_config.get_vocab_size()
    )

    from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

    if (
        head is not None
        and draft_vocab == target_vocab
        and _should_share(draft, "has_own_lm_head", draft.lm_head, head)
    ):
        draft.lm_head = head
    if not getattr(draft, "has_own_lm_head", False) and draft.lm_head is not head:
        raise ValueError("DSpark requires a shared full-vocabulary target LM head")

    _LOG.info(
        "[KunlunPlugin] DSpark target embedding/head shared: %s/%s",
        draft.model.embed_tokens is language_model.model.embed_tokens,
        draft.lm_head is head,
    )

    replicate_markov_weights(draft)
    return draft


_up.load_dspark_model = load_dspark_model
