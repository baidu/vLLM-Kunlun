# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import uuid

import torch

MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | torch.Tensor | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len


def is_moe_layer(module: torch.nn.Module) -> bool:
    # TODO(bnell): Should use isinstance but can't due to circular dependencies.
    def _check_bases(cls):
        if cls.__name__ == "FusedMoE":
            return True

        for b in cls.__bases__:
            if _check_bases(b):
                return True

    return _check_bases(module.__class__)



def compile_unless_eager(fn, **compile_kwargs):
    """torch.compile(fn, **kw) that runs eagerly on Inductor-less platforms.

    Standalone ``@torch.compile`` decorators bypass the per-op guard in
    ``vllm/model_executor/custom_op.py``. On Kunlun XPU (CUDA-compat) Inductor's
    triton kernels cannot execute, so such decorators crash at inference (e.g.
    update_num_computed_tokens_for_batch_change during spec-decode). Run eagerly
    on such platforms.

    NOTE: get_cached_compilation_config() (used by custom_op.py) is only valid
    inside the set_current_vllm_config() context during init; it raises during
    model forward in worker processes. We therefore decide at import time using
    env vars that the Kunlun start script sets and workers inherit
    (VLLM_USE_TRITON_KERNELS=0 and XPU_VISIBLE_DEVICES) instead of the
    compilation config.
    """
    if (os.environ.get("VLLM_USE_TRITON_KERNELS") == "0"
            or bool(os.environ.get("XPU_VISIBLE_DEVICES"))):
        return fn
    return torch.compile(fn, **compile_kwargs)