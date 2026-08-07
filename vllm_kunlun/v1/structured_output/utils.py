"""Kunlun replacement for vllm.v1.structured_output.utils.apply_grammar_bitmask.

Upstream's GPU branch calls ``xgr.apply_token_bitmask_inplace`` with
``backend="auto"`` (the default), which routes XPU tensors to the
torch_compile path -- that path needs libcuda.so and raises
``CUDA_ERROR_NOT_SUPPORTED`` on Kunlun XPU. Force ``backend="torch_native"``
instead.

The replacement also rebinds the symbol in any already-imported consumer
(e.g. ``vllm.v1.worker.gpu_model_runner`` that does
``from vllm.v1.structured_output.utils import apply_grammar_bitmask`` at
module top level), since attribute lookup on the upstream module alone
would not reach those bound names.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch
import vllm.v1.structured_output.utils as _upstream
from vllm.utils.import_utils import LazyLoader
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    import xgrammar as xgr
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")


_XPU_BACKEND = "torch_native"


def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    logits: torch.Tensor,
) -> None:
    """Same as upstream, but forces ``backend='torch_native'`` for XPU."""
    grammar_bitmask = grammar_output.grammar_bitmask

    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    struct_out_req_ids = set(grammar_output.structured_output_request_ids)
    for batch_index, req_id in enumerate(input_batch.req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    out_indices: list[int] = []
    sorted_bitmask = np.full(
        shape=(logits.shape[0], grammar_bitmask.shape[1]),
        fill_value=-1,
        dtype=grammar_bitmask.dtype,
    )
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
                out_indices.append(bitmask_index)
        cumulative_index += 1 + num_spec_tokens

    grammar_bitmask = torch.from_numpy(sorted_bitmask).to(
        logits.device, non_blocking=True
    )

    skip_out_indices = len(out_indices) == logits.shape[0]

    if not logits.is_cpu:
        index_tensor = None
        if not skip_out_indices:
            pin_memory = is_pin_memory_available()
            index_tensor = torch.tensor(
                out_indices, dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            index_tensor = index_tensor.to(logits.device, non_blocking=True)

        xgr.apply_token_bitmask_inplace(
            logits, grammar_bitmask, indices=index_tensor, backend=_XPU_BACKEND
        )
        return

    # CPU path is unchanged from upstream: defer so future fixes flow in.
    if _ORIGINAL is None:
        raise RuntimeError("original apply_grammar_bitmask is unavailable")
    _ORIGINAL(scheduler_output, grammar_output, input_batch, logits)


# Idempotent monkey-patch: safe under fork() and re-import.
_ORIGINAL = getattr(_upstream, "apply_grammar_bitmask", None)
logger = logging.getLogger("vllm_kunlun")

if not getattr(_ORIGINAL, "_kunlun_patched", False):
    _upstream.apply_grammar_bitmask = apply_grammar_bitmask
    apply_grammar_bitmask._kunlun_patched = True  # type: ignore[attr-defined]

    rebind_count = 0
    for module in list(sys.modules.values()):
        if module is None or module is _upstream:
            continue
        # ``__dict__.get`` rather than ``getattr``: a real ``from ... import``
        # binding always lands in the module's own ``__dict__``, while
        # ``getattr`` would fall through to module-level ``__getattr__``
        # hooks. transformers registers ~192 backward-compat alias modules
        # (``models.*.image_processing_*``) whose catch-all
        # ``__getattr__`` logs a warning and eagerly imports the real
        # module for *any* attribute name -- probing them costs ~0.75s and
        # ~38MB per process and floods the log.
        if module.__dict__.get("apply_grammar_bitmask") is _ORIGINAL:
            try:
                setattr(module, "apply_grammar_bitmask", apply_grammar_bitmask)
                rebind_count += 1
            except Exception:
                pass

    logger.info(
        "[KunlunPlugin] apply_grammar_bitmask patched "
        "in vllm_kunlun/v1/structured_output/utils.py, rebound=%s",
        rebind_count,
    )
