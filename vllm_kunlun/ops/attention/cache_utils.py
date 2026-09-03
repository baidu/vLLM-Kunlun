"""Correctness fallbacks for DeepSeek V4 packed KV cache operations."""

import torch


_TOKEN_FP8_DIM = 448
_TOKEN_BF16_DIM = 64
_TOKEN_SCALE_DIM = 8
_QUANT_BLOCK_SIZE = 64
_TOKEN_DATA_SIZE = _TOKEN_FP8_DIM + _TOKEN_BF16_DIM * 2
_NUM_QUANT_BLOCKS = _TOKEN_FP8_DIM // _QUANT_BLOCK_SIZE


def _decode_e4m3(raw: torch.Tensor, use_fnuz: bool) -> torch.Tensor:
    bits = raw.to(torch.int32)
    sign = 1.0 - 2.0 * ((bits >> 7) & 1).float()
    exponent = ((bits >> 3) & 0xF).float()
    mantissa = (bits & 0x7).float()

    if use_fnuz:
        magnitude = torch.where(
            exponent == 0,
            mantissa * (2.0**-10),
            torch.pow(2.0, exponent - 8.0) * (1.0 + mantissa / 8.0),
        )
        decoded = sign * magnitude
        return torch.where(bits == 0x80, torch.nan, decoded)

    magnitude = torch.where(
        exponent == 0,
        mantissa * (2.0**-9),
        torch.pow(2.0, exponent - 7.0) * (1.0 + mantissa / 8.0),
    )
    decoded = sign * magnitude
    return torch.where((bits & 0x7F) == 0x7F, torch.nan, decoded)


def dequantize_and_gather_k_cache_pytorch(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    use_fnuz: bool = False,
) -> None:
    """Gather DeepSeek V4 plain BF16 or packed FP8 cache pages."""
    if out.shape[-1] != _TOKEN_FP8_DIM + _TOKEN_BF16_DIM:
        raise ValueError(f"DeepSeek V4 cache output must have 512 columns, got {out.shape[-1]}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")

    if k_cache.shape[0] == 0:
        raise ValueError("V4 cache must contain at least one physical block")

    plain_cache = k_cache.dtype != torch.uint8
    if plain_cache:
        supported_dtypes = (torch.bfloat16, torch.float16, torch.float32)
        if k_cache.dtype not in supported_dtypes:
            raise NotImplementedError(
                f"Kunlun plain-row V4 cache does not support dtype {k_cache.dtype}"
            )
        if k_cache.ndim < 3 or k_cache.shape[1] < block_size:
            raise ValueError(
                "Plain-row V4 cache must have shape [num_blocks, block_size, ...]"
            )
        if k_cache[0, 0].numel() < out.shape[-1]:
            raise ValueError("Plain-row V4 cache row is smaller than the 512-column output")
    else:
        required_page_bytes = block_size * (_TOKEN_DATA_SIZE + _TOKEN_SCALE_DIM)
        if k_cache[0].numel() < required_page_bytes:
            raise ValueError(
                f"Packed V4 cache page needs {required_page_bytes} bytes, "
                f"got {k_cache[0].numel()}"
            )

    num_reqs = seq_lens.shape[0]
    if out.shape[0] < num_reqs or block_table.shape[0] < num_reqs:
        raise ValueError("Output and block table must contain every request")
    if gather_lens is not None and gather_lens.shape[0] < num_reqs:
        raise ValueError("gather_lens must contain every request")

    seq_lens_cpu = seq_lens.detach().cpu().tolist()
    gather_lens_cpu = (
        gather_lens.detach().cpu().tolist() if gather_lens is not None else None
    )
    block_table_cpu = block_table.detach().cpu()

    def physical_block_for(batch_idx: int, logical_block: int) -> int:
        if logical_block >= block_table_cpu.shape[1]:
            raise ValueError(f"Logical block {logical_block} is outside the block table")
        physical_block = int(block_table_cpu[batch_idx, logical_block])
        if physical_block < 0 or physical_block >= k_cache.shape[0]:
            raise ValueError(f"Invalid physical block {physical_block}")
        return physical_block

    for batch_idx, seq_len in enumerate(seq_lens_cpu):
        gather_len = (
            int(gather_lens_cpu[batch_idx])
            if gather_lens_cpu is not None
            else int(seq_len)
        )
        seq_len = int(seq_len)
        if gather_len < 0 or gather_len > seq_len:
            raise ValueError(
                f"gather length must be in [0, {seq_len}], got {gather_len}"
            )
        if offset + gather_len > out.shape[1]:
            raise ValueError("Gathered rows do not fit in the output workspace")
        start_pos = seq_len - gather_len

        if plain_cache:
            position = start_pos
            output_idx = offset
            while position < seq_len:
                logical_block = position // block_size
                position_in_block = position % block_size
                physical_block = physical_block_for(batch_idx, logical_block)
                rows_in_block = min(block_size - position_in_block, seq_len - position)
                token_rows = k_cache[
                    physical_block,
                    position_in_block : position_in_block + rows_in_block,
                ].reshape(rows_in_block, -1)
                out[
                    batch_idx, output_idx : output_idx + rows_in_block
                ].copy_(token_rows[:, : out.shape[-1]].to(out.dtype))
                position += rows_in_block
                output_idx += rows_in_block
            continue

        for gather_idx in range(gather_len):
            position = start_pos + gather_idx
            logical_block = position // block_size
            position_in_block = position % block_size
            physical_block = physical_block_for(batch_idx, logical_block)

            block_bytes = k_cache[physical_block].reshape(-1).view(torch.uint8)
            data_offset = position_in_block * _TOKEN_DATA_SIZE
            scale_offset = (
                block_size * _TOKEN_DATA_SIZE
                + position_in_block * _TOKEN_SCALE_DIM
            )

            token_bytes = block_bytes[data_offset : data_offset + _TOKEN_DATA_SIZE]
            fp8_bytes = token_bytes[:_TOKEN_FP8_DIM]
            bf16_values = token_bytes[_TOKEN_FP8_DIM:].view(torch.bfloat16)
            scale_bytes = block_bytes[
                scale_offset : scale_offset + _NUM_QUANT_BLOCKS
            ]

            fp8_values = _decode_e4m3(fp8_bytes, use_fnuz)
            scales = torch.pow(2.0, scale_bytes.float() - 127.0).repeat_interleave(
                _QUANT_BLOCK_SIZE
            )
            output_row = out[batch_idx, offset + gather_idx]
            output_row[:_TOKEN_FP8_DIM] = (fp8_values * scales).to(out.dtype)
            output_row[_TOKEN_FP8_DIM:] = bf16_values
