# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op override for ``vllm.v1.worker.gpu.structured_outputs``.

The V2 model runner has its own grammar-bitmask kernel
(``_apply_grammar_bitmask_kernel``), separate from the V1 path patched in
``vllm_kunlun/v1/structured_output/utils.py``. Launching it on Kunlun XPU fails
with ``Triton Error [CUDA]: CUDA_ERROR_NOT_SUPPORTED``, and the worker process
cannot rely on ``HAS_TRITON`` being False (Triton finds an active driver once
``torch_xmlir`` is initialised), so the launch site is replaced with the Kunlun
native op ``torch.ops.xspeedgate_ops.apply_grammar_bitmask``.

Only ``StructuredOutputsWorker.apply_grammar_bitmask`` is overridden; the rest of
the upstream class (buffers, sizing) is left untouched. The upstream side
copy-stream is dropped: the H2D copies are issued on the current stream instead,
which keeps the ordering trivially correct on XPU.

The native op takes the packed bitmask, a per-mask encoded index
``logits_indices[m] = req_idx * mask_stride + position_idx`` (it recovers
``req_idx``/``position_idx`` by div/mod and maps to logits row
``cu_num_logits[req_idx] + position_idx`` on-device), the device
``cu_num_logits`` prefix-sum, and ``mask_stride`` = the bitmask row's int32-word
count. Bit semantics match upstream: set bit == allowed, clear bit => the
token's logit is set to ``-inf``. It supports float16/bfloat16 logits, which is
exactly what reaches this call site -- ``apply_grammar_bitmask`` runs on the raw
model logits before the sampler's fp32 copy (``sample/sampler.py:157``).
"""

import logging

import numpy as np
import torch
import vllm.v1.worker.gpu.structured_outputs as _up
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

logger = logging.getLogger("vllm_kunlun")


def _apply_grammar_bitmask(
    self,
    logits: torch.Tensor,
    input_batch,
    grammar_req_ids: list[str],
    grammar_bitmask: np.ndarray,
) -> None:
    """Native-op replacement of ``_apply_grammar_bitmask_kernel``.

    Row ``m`` of ``grammar_bitmask`` applies to the ``m``-th grammar-constrained
    logit position. The native op needs those positions encoded as
    ``req_idx * mask_stride + position_idx``; the same host-side walk over
    ``grammar_req_ids`` upstream already does (it materialises
    ``cu_num_logits_np`` with ``.tolist()``) builds the encoding here, so no new
    host sync is introduced.
    """
    if not grammar_req_ids:
        return

    num_masks = grammar_bitmask.shape[0]
    bitmask = async_copy_to_gpu(grammar_bitmask, out=self.grammar_bitmask[:num_masks])
    # mask_stride = int32 words per bitmask row (== cdiv(vocab_size, 32)); also
    # the encoding step for logits_indices.
    mask_stride = bitmask.shape[1]

    # Encode per-mask indices: req_idx * mask_stride + position_idx.
    req_ids = input_batch.req_ids
    cu_num_logits = input_batch.cu_num_logits_np.tolist()
    req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
    encoded: list[int] = []
    for grammar_req_id in grammar_req_ids:
        req_idx = req_id_to_idx[grammar_req_id]
        num_logits = cu_num_logits[req_idx + 1] - cu_num_logits[req_idx]
        # The native op recovers (req_idx, position_idx) by div/mod on
        # ``mask_stride``, so the encoding is only invertible while every
        # request's constrained-position count stays below it. This holds
        # comfortably in practice (per-step positions are tiny, mask_stride is
        # ~vocab/32); assert it so a violation fails loudly instead of silently
        # aliasing into the next request's rows.
        assert num_logits < mask_stride, (
            f"grammar position count {num_logits} >= mask_stride {mask_stride}; "
            "logits_indices encoding would alias across requests"
        )
        encoded.extend(req_idx * mask_stride + p for p in range(num_logits))
    assert num_masks == len(encoded)

    logits_indices = async_copy_to_gpu(
        np.asarray(encoded, dtype=np.int32),
        out=self.logits_indices[: len(encoded)],
    )

    torch.ops.xspeedgate_ops.apply_grammar_bitmask(
        logits,
        bitmask,
        logits_indices,
        input_batch.cu_num_logits,
        mask_stride,
    )


_up.StructuredOutputsWorker.apply_grammar_bitmask = _apply_grammar_bitmask
logger.info("[KunlunPlugin] V2 StructuredOutputsWorker patched (xspeedgate_ops native)")
