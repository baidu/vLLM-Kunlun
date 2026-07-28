# SPDX-License-Identifier: Apache-2.0
"""Kunlun overrides for Qwen3 MoE models."""

from collections.abc import Iterable, Iterator

import torch
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeForCausalLM as VllmQwen3MoeForCausalLM,
)

_KV_SCALE_SUFFIXES = ("k_scale", "v_scale", "k_zero_point", "v_zero_point")


def remap_qwen3_kv_scale_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Map llm-compressor KV metadata to vLLM Attention parameters."""
    for name, weight in weights:
        for suffix in _KV_SCALE_SUFFIXES:
            checkpoint_suffix = f".self_attn.{suffix}"
            if name.endswith(checkpoint_suffix):
                name = name.removesuffix(checkpoint_suffix) + (
                    f".self_attn.attn.{suffix}"
                )
                break
        yield name, weight


class Qwen3MoeForCausalLM(VllmQwen3MoeForCausalLM):
    """Qwen3 MoE with per-head KV-cache scale checkpoint loading."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(remap_qwen3_kv_scale_weights(weights))
