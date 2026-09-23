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
    mask_stride = bitmask.shape[1]
    mapping: list[int] = []
    req_ids = input_batch.req_ids
    cu_num_logits = input_batch.cu_num_logits_np.tolist()
    req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
    for grammar_req_id in grammar_req_ids:
        req_idx = req_id_to_idx[grammar_req_id]
        base = req_idx * mask_stride
        q_len = cu_num_logits[req_idx + 1] - cu_num_logits[req_idx]
        mapping.extend(range(base, base + q_len))

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
    torch.ops.xspeedgate_ops.apply_grammar_bitmask(
        logits,
        bitmask,
        logits_indices,
        input_batch.cu_num_logits,
        mask_stride,
    )

    # Ensure the copy stream waits for the device tensors to finish being used
    # before it re-uses or deallocates them
    self.copy_stream.wait_stream(current_stream)


apply_grammar_bitmask._kunlun_patched = True
