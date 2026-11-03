import pytest
import torch

from vllm_kunlun.v1.attention.backends.gdn_attn import build_spec_pad_indices


def test_uniform_lengths_returns_none():
    """No padding needed when every spec request has exactly `width` tokens."""
    lens = torch.tensor([4, 4, 4], dtype=torch.int64)
    assert build_spec_pad_indices(lens, 4) is None


def test_single_clipped_request():
    """A request clipped to 3 tokens is padded to 4 and back."""
    lens = torch.tensor([3], dtype=torch.int64)
    out = build_spec_pad_indices(lens, 4)
    assert out is not None
    gather, unpad = out

    assert gather.tolist() == [0, 1, 2, 2]  # last real token replicated
    assert unpad.tolist() == [0, 1, 2]
    assert gather.dtype == torch.int32 and unpad.dtype == torch.int32


def test_mixed_lengths_round_trip():
    """gather[unpad] must map back onto the real tokens in order."""
    lens = torch.tensor([3, 4, 1], dtype=torch.int64)
    width = 4
    gather, unpad = build_spec_pad_indices(lens, width)

    assert gather.numel() == lens.numel() * width
    assert unpad.numel() == int(lens.sum().item())
    # real token i (flattened order) must survive pad -> unpad unchanged
    assert gather[unpad].tolist() == list(range(int(lens.sum().item())))


@pytest.mark.parametrize("width", [2, 4])
def test_padding_never_reads_out_of_range(width):
    lens = torch.tensor([1, width, width - 1], dtype=torch.int64)
    gather, _ = build_spec_pad_indices(lens, width)
    assert int(gather.max().item()) < int(lens.sum().item())
