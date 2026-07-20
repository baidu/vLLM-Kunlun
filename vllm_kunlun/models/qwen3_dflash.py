# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    maybe_prefix,
)


class DFlashQwen3Model(Qwen3Model):
    """Qwen3 draft model shell for DFlash.

    The initial Kunlun backport reuses the stable Qwen3 model implementation and
    exposes the DFlash-specific API surface expected by the proposer.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        dflash_config = getattr(self.config, "dflash_config", None)
        self.use_aux_hidden_state = (
            dflash_config.get("use_aux_hidden_state", True)
            if dflash_config is not None
            else True
        )

    def precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Upstream DFlash pre-inserts cross-attention context K/V here. In this
        # 0.15.1 backport the proposer runs on the EAGLE-style path, so no extra
        # pre-insertion is required yet.
        del hidden_states, positions, slot_mapping


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    packed_modules_mapping = Qwen3ForCausalLM.packed_modules_mapping

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = vllm_config.quant_config
        if getattr(config, "draft_vocab_size", None) is None:
            config.draft_vocab_size = getattr(config, "vocab_size", None)
        self.config = config
        self.quant_config = quant_config
        self.model = DFlashQwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            if getattr(config, "tie_word_embeddings", False):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        logits_processor_arg=None,
    ) -> torch.Tensor | None:
        del logits_processor_arg
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
