# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun torch-native replacement for ``vllm.v1.worker.gpu.structured_outputs``.

The V2 model runner has its own grammar-bitmask kernel
(``_apply_grammar_bitmask_kernel``), separate from the V1 path patched in
``vllm_kunlun/v1/structured_output/utils.py``. Launching it on Kunlun XPU fails
with ``Triton Error [CUDA]: CUDA_ERROR_NOT_SUPPORTED``, and the worker process
cannot rely on ``HAS_TRITON`` being False (Triton finds an active driver once
``torch_xmlir`` is initialised), so the launch site is replaced explicitly.

Only ``StructuredOutputsWorker.apply_grammar_bitmask`` is overridden; the
upstream class (buffers, sizing) is reused verbatim via ``load_upstream``. The
upstream side copy-stream is dropped: the H2D copies are issued on the current
stream instead, which keeps the ordering trivially correct on XPU.
"""

import logging

import numpy as np
import torch
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

from vllm_kunlun.v1.worker.gpu._upstream import load_upstream

logger = logging.getLogger("vllm_kunlun")

_up = load_upstream("vllm.v1.worker.gpu.structured_outputs")

StructuredOutputsWorker = _up.StructuredOutputsWorker

_BITS_PER_WORD = 32


def _apply_grammar_bitmask(
    self,
    logits: torch.Tensor,
    input_batch,
    grammar_req_ids: list[str],
    grammar_bitmask: np.ndarray,
) -> None:
    """torch-native replacement of ``_apply_grammar_bitmask_kernel``.

    ``grammar_bitmask`` packs one bit per token id into int32 words; a zero bit
    means the token is disallowed and its logit must become ``-inf``. Row ``i``
    of the bitmask applies to logits row ``mapping[i]``.
    """
    if not grammar_req_ids:
        return

    num_masks = grammar_bitmask.shape[0]
    bitmask = async_copy_to_gpu(grammar_bitmask, out=self.grammar_bitmask[:num_masks])

    # Construct bitmask -> logits mapping (identical to upstream).
    mapping: list[int] = []
    req_ids = input_batch.req_ids
    cu_num_logits = input_batch.cu_num_logits_np.tolist()
    req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
    for grammar_req_id in grammar_req_ids:
        req_idx = req_id_to_idx[grammar_req_id]
        mapping.extend(range(cu_num_logits[req_idx], cu_num_logits[req_idx + 1]))
    assert num_masks == len(mapping)

    rows = async_copy_to_gpu(
        np.asarray(mapping, dtype=np.int32), out=self.logits_indices[: len(mapping)]
    ).to(torch.long)

    # Unpack the bitmask: bit j of word w covers token w * 32 + j. int32 right
    # shift is arithmetic, but the low bit of the result is still the wanted
    # bit, so sign-bit words are handled correctly.
    shifts = torch.arange(_BITS_PER_WORD, dtype=torch.int32, device=bitmask.device)
    bits = (bitmask.unsqueeze(-1) >> shifts) & 1
    vocab_size = logits.shape[-1]
    allowed = bits.view(num_masks, -1)[:, :vocab_size].to(torch.bool)

    selected = logits.index_select(0, rows)
    selected.masked_fill_(~allowed, float("-inf"))
    logits.index_copy_(0, rows, selected)


if not getattr(StructuredOutputsWorker, "_kunlun_v2_patched", False):
    StructuredOutputsWorker.apply_grammar_bitmask = _apply_grammar_bitmask
    StructuredOutputsWorker._kunlun_v2_patched = True
    logger.info(
        "[KunlunPlugin] V2 StructuredOutputsWorker patched (torch-native bitmask)"
    )
