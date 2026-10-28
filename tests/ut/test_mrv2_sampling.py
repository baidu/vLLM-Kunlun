import torch

from vllm_kunlun.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
from vllm_kunlun.v1.worker.gpu.sample.min_p import apply_min_p


def test_apply_temperature_keeps_greedy_rows_unchanged():
    logits = torch.tensor([[2.0, 4.0], [2.0, 4.0]])
    apply_temperature(logits, torch.tensor([0, 1]), torch.tensor([0.0, 2.0]))
    assert torch.equal(logits[0], torch.tensor([2.0, 4.0]))
    assert torch.equal(logits[1], torch.tensor([1.0, 2.0]))


def test_gumbel_sample_is_argmax_for_greedy_requests():
    result = gumbel_sample(
        torch.tensor([[1.0, 4.0, 2.0]]),
        torch.tensor([0]),
        torch.tensor([0.0]),
        torch.tensor([17]),
        torch.tensor([3]),
        apply_temperature=True,
    )
    assert torch.equal(result, torch.tensor([1]))


def test_gumbel_sample_is_reproducible_for_same_inputs():
    args = dict(
        logits=torch.zeros((2, 4)),
        expanded_idx_mapping=torch.tensor([0, 1]),
        temperature=torch.tensor([1.0, 1.0]),
        seed=torch.tensor([10, 20]),
        pos=torch.tensor([0, 1]),
        apply_temperature=False,
    )
    assert torch.equal(gumbel_sample(**args), gumbel_sample(**args))


def test_apply_min_p_filters_logits_below_relative_threshold():
    logits = torch.tensor([[1.0, 4.0, 2.0]])
    apply_min_p(logits, torch.tensor([0]), torch.tensor([0.5]))
    assert torch.isneginf(logits[0, 0])
    assert torch.isneginf(logits[0, 2])
    assert logits[0, 1] == 4.0


def test_apply_min_p_zero_is_noop():
    logits = torch.tensor([[1.0, 4.0, 2.0]])
    original = logits.clone()
    apply_min_p(logits, torch.tensor([0]), torch.tensor([0.0]))
    assert torch.equal(logits, original)
