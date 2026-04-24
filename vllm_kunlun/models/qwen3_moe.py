from __future__ import annotations

import logging
import typing
from collections.abc import Iterable

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models import qwen3_moe as upstream

logger = logging.getLogger(__name__)


class Qwen3MoeModel(upstream.Qwen3MoeModel):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        skipped_count = 0
        skipped_example: str | None = None
        for name, loaded_weight in weights:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(
                    param, "weight_loader", upstream.default_weight_loader
                )
                assert (
                    loaded_weight.numel() == 1
                ), f"KV scale numel {loaded_weight.numel()} != 1"
                loaded_weight = loaded_weight.squeeze()
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name or "zero_point" in name:
                name = upstream.maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if upstream.is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    name = upstream.maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", upstream.default_weight_loader
                )
                if weight_loader == upstream.default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)

                    if upstream.is_pp_missing_parameter(name_mapped, self):
                        continue
                    if (
                        name_mapped.endswith(ignore_suffixes)
                        and name_mapped not in params_dict
                    ):
                        continue
                    if name_mapped not in params_dict:
                        skipped_count += 1
                        if skipped_example is None:
                            skipped_example = name
                        continue

                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        typing.Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    if upstream.is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", upstream.default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if skipped_count:
            logger.warning(
                "Skipping %d unmatched Qwen3-MoE expert weights. Example: %s",
                skipped_count,
                skipped_example,
            )

        return loaded_params


class Qwen3MoeForCausalLM(upstream.Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        original_model_cls = upstream.Qwen3MoeModel
        upstream.Qwen3MoeModel = Qwen3MoeModel
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            upstream.Qwen3MoeModel = original_model_cls
