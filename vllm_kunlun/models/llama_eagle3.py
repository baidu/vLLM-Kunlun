# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kunlun-specific performance patch for the Llama EAGLE3 drafter."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import kunlun_ops
import torch
from vllm.logger import init_logger


logger = init_logger(__name__)


def _refresh_fc_norm_weight(model: Any) -> None:
    fc_norm = getattr(model.model, "fc_norm", None)
    if fc_norm is None:
        return
    weight = torch.stack([norm.weight.detach() for norm in fc_norm]).contiguous()
    if "_kunlun_fc_norm_weight" in model.model._buffers:
        model.model._buffers["_kunlun_fc_norm_weight"] = weight
    else:
        model.model.register_buffer(
            "_kunlun_fc_norm_weight", weight, persistent=False
        )


def _combine_hidden_states(
    self: Any,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if not self.model.use_aux_hidden_state:
        return hidden_states

    if self.model.norm_before_fc:
        hidden_states = self.model.input_norm(hidden_states)

    fc_norm = self.model.fc_norm
    if fc_norm is not None:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        fused_weight = getattr(self.model, "_kunlun_fc_norm_weight", None)
        if fused_weight is None or fused_weight.device != hidden_states.device:
            # Dummy-weight and unusual reload paths can bypass load_weights.
            # Refresh once before entering the steady-state decode path.
            _refresh_fc_norm_weight(self)
            fused_weight = self.model._kunlun_fc_norm_weight

        num_aux = self.model.num_aux_hidden_states
        if num_aux <= 0:
            raise ValueError(
                f"EAGLE3 requires at least one auxiliary state, got {num_aux}"
            )
        target_hidden_size = hidden_states.shape[-1] // num_aux
        if hidden_states.shape[-1] != num_aux * target_hidden_size:
            raise ValueError(
                "EAGLE3 auxiliary hidden size is not divisible by the "
                f"number of states: shape={hidden_states.shape}, num_aux={num_aux}"
            )
        expected_weight_shape = (num_aux, target_hidden_size)
        if tuple(fused_weight.shape) != expected_weight_shape:
            raise ValueError(
                "EAGLE3 fc_norm weight shape does not match the auxiliary "
                f"hidden states: weight={tuple(fused_weight.shape)}, "
                f"expected={expected_weight_shape}"
            )
        output = torch.empty_like(hidden_states)
        ret = kunlun_ops.head_rmsnorm(
            hidden_states.view(-1, num_aux, target_hidden_size),
            fused_weight,
            output.view(-1, num_aux, target_hidden_size),
            0,
            num_aux,
            fc_norm[0].variance_epsilon,
        )
        if ret != 0:
            raise RuntimeError(f"kunlun_ops.head_rmsnorm failed with ret={ret}")
        hidden_states = output

    return self.model.fc(hidden_states)


_combine_hidden_states._kunlun_patched = True  # type: ignore[attr-defined]


def _disable_identity_draft_vocab_mapping(model: Any) -> bool:
    """Represent an all-zero draft-to-target offset map as identity.

    EAGLE checkpoints without a ``d2t`` tensor leave vLLM's initialized
    all-zero parameter in place.  ``compute_logits`` then unnecessarily
    allocates and scatters a full-vocabulary logits tensor even though
    ``base + d2t`` is exactly ``base``.  ``None`` is the model's existing
    identity-map representation and also enables vocab-parallel argmax.
    """
    mapping = getattr(model, "draft_id_to_target_id", None)
    if mapping is None:
        return False
    if mapping.ndim != 1:
        raise ValueError(
            "EAGLE3 draft_id_to_target_id must be one-dimensional, got "
            f"shape={tuple(mapping.shape)}"
        )
    if torch.count_nonzero(mapping).item() != 0:
        return False
    model.draft_id_to_target_id = None
    return True


def _get_top_tokens(
    self: Any,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Greedy draft selection without gathering the full TP vocabulary."""
    if getattr(self, "draft_id_to_target_id", None) is not None:
        logits = self.compute_logits(hidden_states)
        if logits is None:
            raise RuntimeError("EAGLE3 compute_logits unexpectedly returned None")
        return logits.argmax(dim=-1)
    return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)


_get_top_tokens._kunlun_patched = True  # type: ignore[attr-defined]


def patch_llama_eagle3(module: Any) -> None:
    """Apply Kunlun XPU performance patches to Eagle3LlamaForCausalLM."""
    cls = module.Eagle3LlamaForCausalLM
    original_load_weights = cls.load_weights

    def load_weights(
        self: Any, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Any:
        result = original_load_weights(self, weights)
        if _disable_identity_draft_vocab_mapping(self):
            logger.info(
                "EAGLE3 draft vocabulary mapping is identity; enabled "
                "the logits fast path and local argmax eligibility."
            )
        _refresh_fc_norm_weight(self)
        return result

    load_weights._kunlun_patched = True  # type: ignore[attr-defined]
    cls.load_weights = load_weights
    cls.combine_hidden_states = _combine_hidden_states
    cls.get_top_tokens = _get_top_tokens
