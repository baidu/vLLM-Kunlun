# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacement for ``vllm.v1.worker.gpu.structured_outputs``.

``StructuredOutputsWorker.apply_grammar_bitmask`` launches the Triton
``_apply_grammar_bitmask_kernel`` upstream. Kunlun XPU cannot JIT-compile Triton
kernels, so it is replaced by its xspeedgate_ops equivalent. The host-side
bitmask -> logits mapping and the async copies are unchanged.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import numpy as np
import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = logging.getLogger("vllm_kunlun")


def apply_grammar_bitmask(
    self,
    logits: torch.Tensor,
    input_batch: InputBatch,
    grammar_req_ids: list[str],
    grammar_bitmask: np.ndarray,
) -> None:
    if not grammar_req_ids:
        return

    # Asynchronously copy the bitmask to GPU.
    with torch.cuda.stream(self.copy_stream):
        bitmask = async_copy_to_gpu(
            grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
        )

    # Construct bitmask -> logits mapping. The kernel takes a packed index
    # `req_idx * mask_stride + position_idx` together with the per-request
    # logits prefix sums, rather than raw logits row indices.
    # Use the upstream 0.28 mapping so adaptive verification and bonus-token
    # slots obey the same contract as its Triton kernel.
    from vllm.v1.worker.gpu.structured_outputs import _build_grammar_mapping

    # Native ABI uses the bitmask row width as the mapping radix. Re-encode
    # the upstream (request, position) pairs with that radix; it need not
    # equal the upstream scheduled-position stride.
    mask_stride = bitmask.shape[1]
    if self.mask_stride > mask_stride:
        bitmask = torch.nn.functional.pad(
            bitmask, (0, self.mask_stride - mask_stride)
        )
        mask_stride = self.mask_stride
    mapping = _build_grammar_mapping(
        input_batch.req_ids, grammar_req_ids, input_batch.cu_num_logits_np,
        input_batch.num_draft_tokens_per_req, self.num_bonus_tokens, mask_stride,
    )

    # Asynchronously copy the mapping to GPU.
    with torch.cuda.stream(self.copy_stream):
        logits_indices = torch.tensor(
            mapping, dtype=torch.int32, device="cpu", pin_memory=True
        )
        logits_indices = self.logits_indices[: len(mapping)].copy_(
            logits_indices, non_blocking=True
        )

    # Ensure all async copies are complete before launching the kernel.
    current_stream = torch.cuda.current_stream()
    current_stream.wait_stream(self.copy_stream)

    num_masks = bitmask.shape[0]
    assert num_masks == len(mapping)
    # The native op supports half/bfloat16. For fp32 logits, generate only
    # the exclusion mask in a half scratch tensor; retain every fp32 value.
    native_logits = (
        torch.zeros_like(logits, dtype=torch.float16)
        if logits.dtype == torch.float32 else logits
    )
    torch.ops.xspeedgate_ops.apply_grammar_bitmask(
        native_logits,
        bitmask,
        logits_indices,
        input_batch.cu_num_logits,
        mask_stride,
    )

    if native_logits is not logits:
        logits.masked_fill_(torch.isneginf(native_logits), -float("inf"))

    # Ensure the copy stream waits for the device tensors to finish being used
    # before it re-uses or deallocates them
    self.copy_stream.wait_stream(current_stream)


apply_grammar_bitmask._kunlun_patched = True
