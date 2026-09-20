# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-device tests for the Kunlun native-op sampling wrappers.

``apply_temperature`` / ``gumbel_sample`` / ``apply_min_p`` now forward to
``torch.ops.xspeedgate_ops.*``, which register for CUDA/XLA/XPU only (Kunlun XPU
presents as CUDA via ``torch_xmlir``), so these are gated on device
availability.

Dtype note: on the live path ``pos = input_batch.positions[...]`` is int64
(input_batch.py:24) and ``seed`` is int64, and the native ``gumbel_sample``
enforces ``pos.scalar_type() == kInt64``. The wrapper flattens the op's ``[n,
1]`` output back to the 1-D shape the samplers expect (sampler.py:235,
compute_topk_logprobs), which is asserted here.
"""

import pytest
import torch

_CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not _CUDA,
    reason="MRV2 native ops (torch.ops.xspeedgate_ops.*) require Kunlun XPU",
)

DEVICE = "cuda"

if _CUDA:
    from vllm_kunlun.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
    from vllm_kunlun.v1.worker.gpu.sample.min_p import apply_min_p


def test_apply_temperature_keeps_greedy_rows_unchanged():
    logits = torch.tensor([[2.0, 4.0], [2.0, 4.0]], dtype=torch.float32, device=DEVICE)
    apply_temperature(
        logits,
        torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
        torch.tensor([0.0, 2.0], dtype=torch.float32, device=DEVICE),
    )
    assert torch.equal(logits[0].cpu(), torch.tensor([2.0, 4.0]))
    assert torch.equal(logits[1].cpu(), torch.tensor([1.0, 2.0]))


def test_gumbel_sample_is_argmax_for_greedy_requests():
    result = gumbel_sample(
        torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.float32, device=DEVICE),
        torch.tensor([0], dtype=torch.int32, device=DEVICE),
        torch.tensor([0.0], dtype=torch.float32, device=DEVICE),
        torch.tensor([17], dtype=torch.int64, device=DEVICE),
        torch.tensor([3], dtype=torch.int64, device=DEVICE),
        apply_temperature=True,
    )
    # Wrapper flattens the native op's [n, 1] output to 1-D.
    assert result.dim() == 1
    assert torch.equal(result.cpu(), torch.tensor([1]))


def test_gumbel_sample_is_reproducible_for_same_inputs():
    args = dict(
        logits=torch.zeros((2, 4), dtype=torch.float32, device=DEVICE),
        expanded_idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
        temperature=torch.tensor([1.0, 1.0], dtype=torch.float32, device=DEVICE),
        seed=torch.tensor([10, 20], dtype=torch.int64, device=DEVICE),
        pos=torch.tensor([0, 1], dtype=torch.int64, device=DEVICE),
        apply_temperature=False,
    )
    assert torch.equal(gumbel_sample(**args).cpu(), gumbel_sample(**args).cpu())


def test_apply_min_p_filters_logits_below_relative_threshold():
    logits = torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.float32, device=DEVICE)
    apply_min_p(
        logits,
        torch.tensor([0], dtype=torch.int32, device=DEVICE),
        torch.tensor([0.5], dtype=torch.float32, device=DEVICE),
    )
    assert torch.isneginf(logits[0, 0])
    assert torch.isneginf(logits[0, 2])
    assert logits[0, 1] == 4.0


def test_apply_min_p_zero_is_noop():
    logits = torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.float32, device=DEVICE)
    original = logits.clone()
    apply_min_p(
        logits,
        torch.tensor([0], dtype=torch.int32, device=DEVICE),
        torch.tensor([0.0], dtype=torch.float32, device=DEVICE),
    )
    assert torch.equal(logits, original)
