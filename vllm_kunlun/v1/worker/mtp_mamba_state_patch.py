# SPDX-License-Identifier: Apache-2.0
"""MTP Mamba state lifecycle fixes."""

from __future__ import annotations

import sys
from typing import Any

import torch


def _is_torch_compiling() -> bool:
    try:
        compiler = getattr(torch, "compiler", None)
        if compiler is not None and compiler.is_compiling():
            return True
    except Exception:
        pass
    try:
        import torch._dynamo as dynamo

        return bool(dynamo.is_compiling())
    except Exception:
        return False


def _is_mamba_group(group: Any) -> bool:
    spec = getattr(group, "kv_cache_spec", None)
    return spec is not None and spec.__class__.__name__ == "MambaSpec"


def _is_qwen35_mtp_runner(runner: Any) -> bool:
    spec_config = getattr(runner, "speculative_config", None)
    draft_model_config = getattr(spec_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == "qwen3_5_mtp"


def _collect_new_mamba_block_ids(
    runner: Any, scheduler_output: Any
) -> dict[int, set[int]]:
    kv_cache_config = getattr(runner, "kv_cache_config", None)
    groups = getattr(kv_cache_config, "kv_cache_groups", ())
    mamba_group_ids = [
        gid for gid, group in enumerate(groups) if _is_mamba_group(group)
    ]
    if not mamba_group_ids:
        return {}

    new_block_ids_by_group: dict[int, set[int]] = {
        gid: set() for gid in mamba_group_ids
    }

    for new_req_data in getattr(scheduler_output, "scheduled_new_reqs", ()):
        if int(getattr(new_req_data, "num_computed_tokens", 0)) != 0:
            continue
        block_ids = getattr(new_req_data, "block_ids", ())
        for gid in mamba_group_ids:
            if gid >= len(block_ids):
                continue
            group_block_ids = block_ids[gid]
            if len(group_block_ids) > 1:
                group_block_ids = group_block_ids[1:]
            new_block_ids_by_group[gid].update(
                int(block_id) for block_id in group_block_ids if int(block_id) >= 0
            )

    return {
        gid: block_ids for gid, block_ids in new_block_ids_by_group.items() if block_ids
    }


def _iter_contiguous_ranges(block_ids: list[int]):
    if not block_ids:
        return
    start = prev = block_ids[0]
    for block_id in block_ids[1:]:
        if block_id == prev + 1:
            prev = block_id
            continue
        yield start, prev + 1
        start = prev = block_id
    yield start, prev + 1


def _clear_new_mamba_blocks(runner: Any, scheduler_output: Any) -> int:
    if getattr(runner, "speculative_config", None) is None:
        return 0
    if _is_torch_compiling():
        return 0

    new_block_ids_by_group = _collect_new_mamba_block_ids(runner, scheduler_output)
    if not new_block_ids_by_group:
        return 0

    kv_cache_config = getattr(runner, "kv_cache_config", None)
    groups = getattr(kv_cache_config, "kv_cache_groups", ())
    forward_context = getattr(
        getattr(runner, "compilation_config", None),
        "static_forward_context",
        {},
    )
    cleared_rows = 0

    for gid, raw_block_ids in new_block_ids_by_group.items():
        if gid >= len(groups):
            continue
        group = groups[gid]
        layer_names = getattr(group, "layer_names", ())
        for layer_name in layer_names:
            attention = forward_context.get(layer_name)
            kv_cache = getattr(attention, "kv_cache", None)
            if not kv_cache:
                continue
            state_tensors = kv_cache[0]
            if not isinstance(state_tensors, (list, tuple)):
                continue
            for state in state_tensors:
                if not isinstance(state, torch.Tensor):
                    continue
                valid_block_ids = sorted(
                    block_id
                    for block_id in raw_block_ids
                    if 0 <= block_id < state.size(0)
                )
                if not valid_block_ids:
                    continue
                for start, stop in _iter_contiguous_ranges(valid_block_ids):
                    state[start:stop].zero_()
                    cleared_rows += stop - start

    return cleared_rows


def patch_gpu_model_runner_module(module: Any) -> None:
    runner_cls = getattr(module, "GPUModelRunner", None)
    if runner_cls is None or getattr(
        runner_cls, "_kunlun_mtp_mamba_state_patched", False
    ):
        return

    original_init = runner_cls.__init__
    original_update_states = runner_cls._update_states
    original_calc_spec_decode_metadata = runner_cls._calc_spec_decode_metadata

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if _is_qwen35_mtp_runner(self) and hasattr(self, "rejection_sampler"):
            from vllm_kunlun.v1.sample.rejection_sampler import RejectionSampler

            self.rejection_sampler = RejectionSampler(self.sampler)

    def _patched_update_states(self: Any, scheduler_output: Any) -> Any:
        if getattr(self, "speculative_config", None) is None:
            return original_update_states(self, scheduler_output)
        result = original_update_states(self, scheduler_output)
        _clear_new_mamba_blocks(self, scheduler_output)
        return result

    def _patched_calc_spec_decode_metadata(self: Any, *args: Any, **kwargs: Any) -> Any:
        metadata = original_calc_spec_decode_metadata(self, *args, **kwargs)
        if _is_qwen35_mtp_runner(self):
            metadata._kunlun_qwen35_mtp = True
        return metadata

    runner_cls.__init__ = _patched_init
    runner_cls._update_states = _patched_update_states
    runner_cls._calc_spec_decode_metadata = _patched_calc_spec_decode_metadata
    runner_cls._kunlun_mtp_mamba_state_patched = True


def patch_loaded_modules() -> None:
    runner_module = sys.modules.get("vllm.v1.worker.gpu_model_runner")
    if runner_module is not None:
        patch_gpu_model_runner_module(runner_module)
