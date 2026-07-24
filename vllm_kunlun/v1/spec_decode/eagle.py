# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kunlun XPU replacements for EAGLE control kernels.

The vLLM module uses CUDA Triton kernels through their ``kernel[grid](...)``
interface.  The small adapter below preserves that call convention while
dispatching the serial EAGLE3 path to graph-safe ``kunlun_ops`` kernels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import kunlun_ops
import torch


PADDING_SLOT_ID = -1


class _KernelAdapter:
    """Expose a Python launcher through Triton's ``kernel[grid]`` syntax."""

    _kunlun_patched = True

    def __init__(self, launcher: Callable[..., None]) -> None:
        self._launcher = launcher

    def __getitem__(self, _grid: Any) -> Callable[..., None]:
        return self._launcher


def _check_ret(op_name: str, ret: int) -> None:
    if ret != 0:
        raise RuntimeError(f"kunlun_ops.{op_name} failed with ret={ret}")


def _prepare_next_token_padded(
    sampled_token_ids: torch.Tensor,
    discard_request_mask: torch.Tensor,
    backup_next_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    vocab_size: int,
    num_sampled_tokens_per_req: int,
    num_reqs: int,
    _stride_sampled_token_ids: int,
    **_kwargs: Any,
) -> None:
    if discard_request_mask.dtype is not torch.bool:
        raise TypeError(
            "eagle_prepare_next_token_padded requires a bool "
            "discard_request_mask"
        )
    if not sampled_token_ids.is_contiguous():
        raise ValueError("sampled_token_ids must be contiguous")
    if discard_request_mask.numel() < num_reqs:
        raise ValueError(
            "discard_request_mask is smaller than num_reqs: "
            f"{discard_request_mask.numel()} < {num_reqs}"
        )
    if backup_next_token_ids.numel() < num_reqs:
        raise ValueError(
            "backup_next_token_ids is smaller than num_reqs: "
            f"{backup_next_token_ids.numel()} < {num_reqs}"
        )

    # vLLM keeps both tensors in max-batch-sized persistent buffers and the
    # Triton reference only indexes their first ``num_reqs`` elements.  The
    # Kunlun op enforces an exact numel match, so pass zero-copy prefix views
    # rather than the full-capacity buffers.
    discard_request_mask = discard_request_mask[:num_reqs]
    backup_next_token_ids = backup_next_token_ids[:num_reqs]
    _check_ret(
        "eagle_prepare_next_token_padded",
        kunlun_ops.eagle_prepare_next_token_padded(
            sampled_token_ids,
            discard_request_mask,
            backup_next_token_ids,
            next_token_ids,
            valid_sampled_tokens_count,
            vocab_size,
            num_sampled_tokens_per_req,
            num_reqs,
        ),
    )


def _prepare_inputs_padded(
    cu_num_draft_tokens: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    query_start_loc_gpu: torch.Tensor,
    token_indices_to_sample: torch.Tensor,
    num_rejected_tokens_gpu: torch.Tensor,
    num_reqs: int,
    **_kwargs: Any,
) -> None:
    _check_ret(
        "eagle_prepare_inputs_padded_v2",
        kunlun_ops.eagle_prepare_inputs_padded_v2(
            cu_num_draft_tokens,
            valid_sampled_tokens_count,
            query_start_loc_gpu,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        ),
    )


def _step_update_slot_mapping_and_metadata(
    positions_1d: torch.Tensor,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_model_len: int,
    out_clamped_positions: torch.Tensor,
    out_slot_mapping: torch.Tensor,
    input_batch_size: int | None = None,
) -> None:
    batch_size = positions_1d.shape[0]
    if input_batch_size is None:
        input_batch_size = batch_size
    if input_batch_size < batch_size:
        raise ValueError(
            f"input_batch_size={input_batch_size} is smaller than "
            f"batch_size={batch_size}"
        )
    if block_table_tensor.dtype is not torch.int32:
        raise TypeError("EAGLE block_table_tensor must be int32")
    if not block_table_tensor.is_contiguous():
        raise ValueError("EAGLE block_table_tensor must be contiguous")

    _check_ret(
        "eagle_step_slot_mapping_metadata",
        kunlun_ops.eagle_step_slot_mapping_metadata(
            positions_1d,
            block_table_tensor,
            seq_lens,
            out_clamped_positions,
            out_slot_mapping,
            block_size,
            max_model_len,
            batch_size,
            input_batch_size,
        ),
    )


_step_update_slot_mapping_and_metadata._kunlun_patched = True  # type: ignore[attr-defined]


def patch_spec_decode_utils(module: Any) -> None:
    """Patch a fully imported ``vllm.v1.spec_decode.utils`` module."""

    module.eagle_prepare_next_token_padded_kernel = _KernelAdapter(
        _prepare_next_token_padded
    )
    module.eagle_prepare_inputs_padded_kernel = _KernelAdapter(
        _prepare_inputs_padded
    )
    module.eagle_step_update_slot_mapping_and_metadata = (
        _step_update_slot_mapping_and_metadata
    )
