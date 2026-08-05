import pytest
import torch

from vllm_kunlun.ops.deepseek_v4_cache import (
    dequantize_and_gather_k_cache_pytorch,
)


def test_gather_plain_bf16_cache_across_physical_blocks():
    cache = torch.empty((3, 4, 512), dtype=torch.bfloat16)
    for block in range(3):
        for position in range(4):
            cache[block, position].fill_(100 * block + position)

    out = torch.zeros((1, 3, 512), dtype=torch.bfloat16)
    dequantize_and_gather_k_cache_pytorch(
        out=out,
        k_cache=cache,
        seq_lens=torch.tensor([6]),
        gather_lens=torch.tensor([3]),
        block_table=torch.tensor([[2, 0]], dtype=torch.int32),
        block_size=4,
        offset=0,
    )

    assert out[0, :, 0].float().tolist() == [203.0, 0.0, 1.0]


def test_gather_packed_fp8_cache_layout():
    block_size = 2
    token_data_size = 576
    scale_size = 8
    cache = torch.zeros((1, block_size * (token_data_size + scale_size)), dtype=torch.uint8)

    token_offset = token_data_size
    cache[0, token_offset : token_offset + 448] = 0x38  # E4M3 1.0
    rope = cache[0, token_offset + 448 : token_offset + token_data_size]
    rope.view(torch.bfloat16).fill_(2.0)
    scale_offset = block_size * token_data_size + scale_size
    cache[0, scale_offset : scale_offset + 7] = 127  # UE8M0 scale 1.0

    out = torch.zeros((1, 1, 512), dtype=torch.bfloat16)
    dequantize_and_gather_k_cache_pytorch(
        out=out,
        k_cache=cache,
        seq_lens=torch.tensor([2]),
        gather_lens=torch.tensor([1]),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        block_size=block_size,
        offset=0,
    )

    torch.testing.assert_close(out[0, 0, :448].float(), torch.ones(448))
    torch.testing.assert_close(out[0, 0, 448:].float(), torch.full((64,), 2.0))


def test_gather_rejects_unsupported_plain_cache_dtype():
    cache = torch.zeros((1, 1, 512), dtype=torch.float64)
    out = torch.zeros((1, 1, 512), dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="does not support dtype"):
        dequantize_and_gather_k_cache_pytorch(
            out=out,
            k_cache=cache,
            seq_lens=torch.tensor([1]),
            gather_lens=None,
            block_table=torch.tensor([[0]], dtype=torch.int32),
            block_size=1,
            offset=0,
        )


def test_gather_rejects_invalid_physical_block():
    cache = torch.zeros((1, 1, 512), dtype=torch.bfloat16)
    out = torch.zeros((1, 1, 512), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="Invalid physical block"):
        dequantize_and_gather_k_cache_pytorch(
            out=out,
            k_cache=cache,
            seq_lens=torch.tensor([1]),
            gather_lens=None,
            block_table=torch.tensor([[-1]], dtype=torch.int32),
            block_size=1,
            offset=0,
        )
