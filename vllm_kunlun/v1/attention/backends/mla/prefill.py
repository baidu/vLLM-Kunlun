# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA prefill backend registration for Kunlun XPU.

vLLM made MLA prefill a pluggable backend chosen by
``vllm.v1.attention.backends.mla.prefill.selector``, and its automatic priority
list only contains CUDA/ROCm backends (FLASH_ATTN, TRTLLM_RAGGED, ...). All of
them fail to import on Kunlun, so ``MLAAttention.__init__`` raised
``No valid MLA prefill backend found`` before any MLA model could load -- even
sparse ones, which never run MHA prefill at all.

Sparse MLA (DeepSeek-V3.2 / GLM-5.2) routes *every* token through
``forward_mqa``, so the prefill backend is constructed and then unused. This
placeholder exists to satisfy that construction. It deliberately raises if it is
ever actually called: Kunlun's dense MLA prefill lives in
``vllm_kunlun/v1/attention/backends/mla/common.py`` and has not been ported to
this interface, and a silently wrong prefill is worse than a clear failure.
"""

from typing import TYPE_CHECKING, ClassVar, Optional

import torch
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.platforms.interface import DeviceCapability

_UNPORTED = (
    "Kunlun has no MLA prefill backend on this interface yet. Sparse MLA "
    "(index_topk models) never reaches it because every token goes through "
    "forward_mqa; a dense MLA model does, and needs "
    "vllm_kunlun/v1/attention/backends/mla/common.py ported to "
    "run_prefill_new_tokens / run_prefill_context_chunk first."
)


class KunlunMLAPrefillBackend(MLAPrefillBackend):
    """Placeholder MLA prefill backend so sparse MLA layers can be built."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]

    @staticmethod
    def get_name() -> str:
        return "KUNLUN_MLA_PREFILL"

    @classmethod
    def supports_compute_capability(
        cls, device_capability: "DeviceCapability"
    ) -> bool:
        # Kunlun reports a synthetic capability; it says nothing about this
        # backend's validity.
        return True

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError(_UNPORTED)

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        raise NotImplementedError(_UNPORTED)
