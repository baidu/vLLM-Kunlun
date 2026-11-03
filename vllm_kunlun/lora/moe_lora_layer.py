from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.lora.layers.utils import _get_lora_device
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.fused_moe import MoERunner

from vllm_kunlun.lora.moe_lora import _KERNEL_DTYPE

CONTEXT_ATTR = "kunlun_expert_lora"


@dataclass
class KunlunMoELoRAContext:
    """Everything the seam needs: four groups of stacked weights + the source of the
    current batch's adapter indices.

    The two elements of ``w13_lora_b_stacked`` (gate first, up second) correspond to the
    two column-offset segments of the w13 output; ``w13_lora_a_stacked`` also has two but
    with identical content (PEFT's fused parameter has only one shrink, and the conversion
    script copies one per slice, keeping it consistent with upstream without dedup).
    """

    w13_lora_a_stacked: tuple[torch.Tensor, ...]
    w13_lora_b_stacked: tuple[torch.Tensor, ...]
    w2_lora_a_stacked: tuple[torch.Tensor, ...]
    w2_lora_b_stacked: tuple[torch.Tensor, ...]
    adapter_enabled: torch.Tensor  # int32 [max_loras + 1]
    max_loras: int
    local_num_experts: int
    top_k: int
    punica_wrapper: object

    @property
    def token_lora_indices(self) -> torch.Tensor:
        return self.punica_wrapper.token_lora_indices

    @property
    def single_lora_slot(self) -> int | None:
        return getattr(self.punica_wrapper, "single_lora_slot", None)


class KunlunFusedMoEWithLoRA(FusedMoEWithLoRA):
    """Expert LoRA container for Kunlun monolithic MoE (does not participate in forward)."""

    def __init__(self, base_layer: MoERunner) -> None:
        nn.Module.__init__(self)
        self.base_layer = base_layer
        self.moe_config = base_layer.moe_config
        self._shared_experts = base_layer._shared_experts

        moe_parallel_config = self.moe_config.moe_parallel_config
        if getattr(moe_parallel_config, "dp_size", 1) > 1 or getattr(
            moe_parallel_config, "is_sequence_parallel", False
        ):
            raise ValueError(
                "Kunlun expert LoRA does not support data parallel / sequence parallel MoE: "
                "after splitting, the tokens MoE sees do not match punica's per-token "
                "adapter indices. Please use TP / EP (--enable-expert-parallel is supported)."
            )
        if getattr(base_layer, "lora_config", None) is not None and getattr(
            base_layer.lora_config, "fully_sharded_loras", False
        ):
            raise ValueError(
                "Kunlun expert LoRA does not support --fully-sharded-loras: "
                "it requires the upstream signature of add_shrink / add_expand, but those "
                "two Kunlun functions have three MoE placeholder parameters inserted."
            )

        routed_experts = base_layer.routed_experts
        # After wrapping, the runner lands at ``mlp.experts.base_layer``, so the weight
        # name mapping has to change accordingly.
        routed_experts.lora_base_layer_prefix = "base_layer."

        self.tp_size = moe_parallel_config.tp_size
        self.tp_rank = moe_parallel_config.tp_rank
        self.device = _get_lora_device(base_layer)

        self._enable_aux_cuda_stream = False
        self._lora_stream = None
        self._events = None
        self._moe_kernel = None

        self._w13_slices = 2 if self.moe_config.is_act_and_mul else 1
        self.n_slices = self.local_num_experts * (self._w13_slices + 1)

    # -- Weight allocation ------------------------------------------------

    @staticmethod
    def _fp16_config(lora_config: LoRAConfig) -> LoRAConfig:
        """Shallow-copy the config with ``lora_dtype`` changed to fp16."""
        patched = copy.copy(lora_config)
        patched.lora_dtype = _KERNEL_DTYPE
        return patched

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        if self._w13_slices != 2:
            raise ValueError(
                "Kunlun expert LoRA currently only supports gated MoE (gate_proj + up_proj)."
            )
        super().create_lora_weights(
            max_loras, self._fp16_config(lora_config), model_config
        )

    # -- Context publishing -----------------------------------------------

    def set_mapping(self, punica_wrapper) -> None:
        """Attach the buffers and punica wrapper onto routed_experts for the seam to use."""
        self.punica_wrapper = punica_wrapper
        context = KunlunMoELoRAContext(
            w13_lora_a_stacked=self.w13_lora_a_stacked,
            w13_lora_b_stacked=self.w13_lora_b_stacked,
            w2_lora_a_stacked=self.w2_lora_a_stacked,
            w2_lora_b_stacked=self.w2_lora_b_stacked,
            adapter_enabled=self.adapter_enabled,
            max_loras=self.max_loras,
            local_num_experts=self.local_num_experts,
            top_k=self.moe_config.experts_per_token,
            punica_wrapper=punica_wrapper,
        )
        setattr(self.base_layer.routed_experts, CONTEXT_ATTR, context)

    # -- Replacement decision ---------------------------------------------

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        from vllm_kunlun.registration.compat_patches import (
            _kunlun_expert_lora_enabled,
            _routed_experts_are_monolithic,
        )

        if not _kunlun_expert_lora_enabled():
            return False
        moe_cls = maybe_get_oot_by_class(MoERunner)
        if not isinstance(source_layer, moe_cls):
            return False
        if len(packed_modules_list) != 2:
            return False
        return _routed_experts_are_monolithic(source_layer)


KunlunFusedMoEWithLoRA.__name__ = "FusedMoEWithLoRA"
