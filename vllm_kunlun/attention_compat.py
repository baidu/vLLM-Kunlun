from __future__ import annotations

import importlib.util

import torch
from vllm.model_executor.layers.attention import Attention

_FLASH_ATTENTION_MODULES = (
    "vllm.vllm_flash_attn",
    "flash_attn",
)
_FLASH_ATTENTION_DTYPES = {
    torch.float16,
    torch.bfloat16,
}

__all__ = ["Attention", "check_upstream_fa_availability"]


def check_upstream_fa_availability(dtype: torch.dtype) -> bool:
    if dtype not in _FLASH_ATTENTION_DTYPES:
        return False

    return any(
        importlib.util.find_spec(module_name) is not None
        for module_name in _FLASH_ATTENTION_MODULES
    )
