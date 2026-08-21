"""
Copyright (c) 2026 Baidu, Inc. All Rights Reserved.

This file is a part of the vllm-kunlun project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import vllm_kunlun._kunlun  # noqa: F401


def _reference_prepare_next_token_ids(
    sampled_token_ids: torch.Tensor,
    discard_request_indices: torch.Tensor,
    num_discarded_requests: int,
    backup_next_token_ids: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_sampled_token_ids_gpu = sampled_token_ids.clone()
    if num_discarded_requests > 0:
        idx = discard_request_indices[:num_discarded_requests]
        if idx.device != valid_sampled_token_ids_gpu.device:
            idx = idx.to(valid_sampled_token_ids_gpu.device, non_blocking=True)
        if idx.dtype != torch.long:
            idx = idx.to(torch.long)
        if idx.numel() > 0:
            valid_sampled_token_ids_gpu.index_fill_(0, idx, -1)

    max_gen_len = sampled_token_ids.shape[-1]
    if max_gen_len == 1:
        valid_mask = torch.ones_like(valid_sampled_token_ids_gpu, dtype=torch.bool)
    else:
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < vocab_size
        )

    valid_sampled_tokens_count = valid_mask.sum(dim=1)
    last_valid_indices = valid_sampled_tokens_count - 1
    last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)
    selected_tokens = torch.gather(
        valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
    ).squeeze(1)
    next_token_ids = torch.where(
        last_valid_indices != -1,
        selected_tokens,
        backup_next_token_ids[: valid_sampled_token_ids_gpu.shape[0]],
    )
    return next_token_ids, valid_sampled_tokens_count


def test_eagle_prepare_next_token_ids_without_discards_matches_reference():
    sampled = torch.tensor([[1, 3, 4], [5, 8, 9]], dtype=torch.int64)
    discard = torch.tensor([1, 0], dtype=torch.int32)
    backup = torch.tensor([10, 11], dtype=torch.int64)

    result = torch.ops._C.eagle_prepare_next_token_ids_padded(
        sampled, discard, 0, backup, 100
    )

    expected = _reference_prepare_next_token_ids(sampled, discard, 0, backup, 100)
    torch.testing.assert_close(result[0], expected[0])
    torch.testing.assert_close(result[1], expected[1])


def test_eagle_prepare_next_token_ids_handles_partial_discards():
    sampled = torch.tensor([[3, 4], [7, 8], [9, 10]], dtype=torch.int64)
    discard = torch.tensor([1, 2, 0], dtype=torch.int32)
    backup = torch.tensor([20, 21, 22], dtype=torch.int64)

    result = torch.ops._C.eagle_prepare_next_token_ids_padded(
        sampled, discard, 2, backup, 100
    )

    expected = _reference_prepare_next_token_ids(sampled, discard, 2, backup, 100)
    torch.testing.assert_close(result[0], expected[0])
    torch.testing.assert_close(result[1], expected[1])


def test_eagle_prepare_next_token_ids_handles_all_discards():
    sampled = torch.tensor([[3, 4], [7, 8]], dtype=torch.int64)
    discard = torch.tensor([0, 1], dtype=torch.int32)
    backup = torch.tensor([30, 31], dtype=torch.int64)

    result = torch.ops._C.eagle_prepare_next_token_ids_padded(
        sampled, discard, 2, backup, 100
    )

    expected = _reference_prepare_next_token_ids(sampled, discard, 2, backup, 100)
    torch.testing.assert_close(result[0], expected[0])
    torch.testing.assert_close(result[1], expected[1])


def test_eagle_prepare_next_token_ids_treats_single_token_rows_as_valid():
    sampled = torch.tensor([[-1], [999]], dtype=torch.int64)
    discard = torch.tensor([], dtype=torch.int32)
    backup = torch.tensor([40, 41], dtype=torch.int64)

    result = torch.ops._C.eagle_prepare_next_token_ids_padded(
        sampled, discard, 0, backup, 100
    )

    expected = _reference_prepare_next_token_ids(sampled, discard, 0, backup, 100)
    torch.testing.assert_close(result[0], expected[0])
    torch.testing.assert_close(result[1], expected[1])


def test_eagle_prepare_next_token_ids_filters_invalid_tokens_and_falls_back():
    sampled = torch.tensor([[-1, 2, 3], [101, 105, 2], [-1, -1, -1]], dtype=torch.int64)
    discard = torch.tensor([2], dtype=torch.int32)
    backup = torch.tensor([50, 51, 52], dtype=torch.int64)

    result = torch.ops._C.eagle_prepare_next_token_ids_padded(
        sampled, discard, 1, backup, 100
    )

    expected = _reference_prepare_next_token_ids(sampled, discard, 1, backup, 100)
    torch.testing.assert_close(result[0], expected[0])
    torch.testing.assert_close(result[1], expected[1])


class _BackupNextTokenIds:
    def __init__(self, size: int):
        self.np = np.zeros(size, dtype=np.int64)
        self.gpu = torch.zeros(size, dtype=torch.int64)

    def copy_to_gpu(self, size: int) -> None:
        self.gpu[:size] = torch.from_numpy(self.np[:size]).to(self.gpu.dtype)


class _Request:
    def __init__(self, token_id: int):
        self.token_id = token_id

    def get_token_id(self, _: int) -> int:
        return self.token_id


def test_prepare_next_token_ids_padded_uses_cpp_op():
    from vllm_kunlun.v1.sample.spec_decode import eagle as eagle_module

    sampled = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    discard = torch.tensor([1], dtype=torch.int32)
    expected_next = torch.tensor([3, 77], dtype=torch.int64)
    expected_counts = torch.tensor([3, 0], dtype=torch.int64)

    proposer = SimpleNamespace(backup_next_token_ids=_BackupNextTokenIds(2))
    common_attn_metadata = SimpleNamespace(seq_lens_cpu=torch.tensor([4, 5]))
    gpu_input_batch = SimpleNamespace(num_reqs=2, req_ids=["a", "b"], vocab_size=100)
    requests = {"a": _Request(70), "b": _Request(77)}

    with patch.object(
        torch.ops._C,
        "eagle_prepare_next_token_ids_padded",
        return_value=(expected_next, expected_counts),
        create=True,
    ) as mocked:
        result = eagle_module.prepare_next_token_ids_padded(
            proposer,
            common_attn_metadata,
            sampled,
            requests,
            gpu_input_batch,
            discard,
            1,
        )

    assert proposer.backup_next_token_ids.np[:2].tolist() == [70, 77]
    mocked.assert_called_once()
    torch.testing.assert_close(result[0], expected_next)
    torch.testing.assert_close(result[1], expected_counts)
