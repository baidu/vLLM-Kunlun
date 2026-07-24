"""Kunlun replacement for vllm.v1.structured_output.utils.apply_grammar_bitmask.

Upstream's GPU branch calls ``xgr.apply_token_bitmask_inplace`` with
``backend="auto"`` (the default), which routes XPU tensors to the
torch_compile path -- that path needs libcuda.so and raises
``CUDA_ERROR_NOT_SUPPORTED`` on Kunlun XPU. Use Kunlun's native bitmask op by
default, with xgrammar ``backend="torch_native"`` retained as an explicit
fallback.

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

import kunlun_ops
import numpy as np
import torch
import vllm.v1.structured_output.utils as _upstream
from vllm.utils.import_utils import LazyLoader
from vllm.utils.platform_utils import is_pin_memory_available

import vllm_kunlun.platforms.envs as kunlun_envs

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
    """Apply grammar masks with Kunlun's native op on XPU by default."""
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

    mask_capacity = grammar_bitmask.shape[-1] * 32
    native_vocab_size = min(logits.shape[-1], mask_capacity)
    real_vocab_size = min(
        input_batch.vocab_size,
        native_vocab_size,
    )
    valid_bits_in_last_word = real_vocab_size % 32
    if valid_bits_in_last_word:
        # Mark padding bits in the final real-vocab word as permitted.
        padding_mask_u32 = (
            ~((1 << valid_bits_in_last_word) - 1) & 0xFFFFFFFF
        )
        padding_mask = np.asarray(padding_mask_u32, dtype=np.uint32).view(
            np.int32
        )
        last_word = real_vocab_size // 32
        np.bitwise_or(
            sorted_bitmask[:, last_word],
            padding_mask,
            out=sorted_bitmask[:, last_word],
        )
    first_padding_word = (real_vocab_size + 31) // 32
    native_mask_words = (native_vocab_size + 31) // 32
    sorted_bitmask[:, first_padding_word:native_mask_words] = -1

    grammar_bitmask = torch.from_numpy(sorted_bitmask).to(
        logits.device, non_blocking=True
    )

    skip_out_indices = len(out_indices) == logits.shape[0]

    if not logits.is_cpu:
        # The wheel uses vocab_size as the physical row stride. Native masking
        # is safe when it spans the complete logits row; otherwise retain the
        # stride-aware xgrammar fallback.
        native_shape_supported = (
            native_vocab_size == logits.shape[-1] and logits.is_contiguous()
        )
        if (
            not kunlun_envs.VLLM_KUNLUN_DISABLE_GRAMMAR_BITMASK
            and native_shape_supported
        ):
            kunlun_ops.apply_token_bitmask_inplace(
                logits, grammar_bitmask, vocab_size=native_vocab_size
            )
            return

        # Strict A/B fallback to the previous XPU path. Unlike the native op,
        # xgrammar needs explicit indices to skip unstructured request rows.
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
        if getattr(module, "apply_grammar_bitmask", None) is _ORIGINAL:
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
