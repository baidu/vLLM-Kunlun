#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-kunlun project.
"""MLA prefill backend placeholder for Kunlun.

``MLAAttention.__init__`` always builds an MLA prefill backend for the MHA
(chunked-prefill) path, and every upstream candidate needs a CUDA-only
FlashAttention/FlashInfer kernel. The Kunlun sparse MLA impl routes all tokens
through ``forward_mqa``, so the prefill backend is constructed but never used.
"""

import torch
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)

CUSTOM_BACKEND_PATH = "vllm_kunlun.v1.attention.backends.mla.prefill.KunlunMLAPrefillBackend"


class KunlunMLAPrefillBackend(MLAPrefillBackend):
    """Never-invoked backend; see module docstring."""

    @staticmethod
    def get_name() -> str:
        return "KUNLUN_MLA_PREFILL"

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Kunlun MLA runs prefill inside the sparse MQA kernel; the MHA "
            "prefill path is not implemented."
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Kunlun MLA runs prefill inside the sparse MQA kernel; chunked "
            "context prefill is not implemented."
        )


register_mla_prefill_backend(MLAPrefillBackendEnum.CUSTOM, CUSTOM_BACKEND_PATH)
