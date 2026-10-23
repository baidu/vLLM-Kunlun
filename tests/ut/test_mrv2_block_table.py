# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-table wrapper dispatch and output views on CPU."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def block_table(load_kunlun_override, native_ops):
    module, upstream = load_kunlun_override(
        "v1/worker/gpu/block_table.py",
        "vllm.v1.worker.gpu.block_table",
        PAD_SLOT_ID=-1,
        BlockTables=type("BlockTables", (), {}),
    )
    return module, upstream


def test_gather_dispatches_heterogeneous_groups_once(block_table, native_ops):
    module, _ = block_table
    sources = [torch.zeros((4, width), dtype=torch.int32) for width in [17, 33]]
    targets = [torch.full_like(value, 777) for value in sources]
    counts = torch.zeros((2, 4), dtype=torch.int32)
    mapping = torch.tensor([2, 0], dtype=torch.int32)
    owner = SimpleNamespace(
        block_tables=[SimpleNamespace(gpu=value) for value in sources],
        input_block_tables=targets,
        num_blocks=SimpleNamespace(gpu=counts),
    )
    native_ops.gather_block_tables = Mock()

    result = module._gather_block_tables(owner, mapping, 3)

    native_ops.gather_block_tables.assert_called_once()
    (
        src,
        dst,
        actual_counts,
        actual_mapping,
        padded,
    ) = native_ops.gather_block_tables.call_args.args
    assert all(actual is wanted for actual, wanted in zip(src, sources))
    assert dst is targets
    assert actual_counts is counts and actual_mapping is mapping
    assert padded == 3
    assert [value.shape for value in result] == [(3, 17), (3, 33)]
    assert all(
        value.data_ptr() == target.data_ptr() for value, target in zip(result, targets)
    )


def test_empty_batch_slot_mapping_forwards_arguments_and_returns_view(
    block_table, native_ops
):
    module, _ = block_table
    sources = [torch.zeros((4, width), dtype=torch.int32) for width in [17, 33]]
    slots = torch.full((2, 8), 777, dtype=torch.int64)
    sizes = torch.tensor([16, 32], dtype=torch.int32)
    mapping = torch.empty(0, dtype=torch.int32)
    starts = torch.tensor([0], dtype=torch.int32)
    positions = torch.empty(0, dtype=torch.int64)
    owner = SimpleNamespace(
        block_tables=[SimpleNamespace(gpu=value) for value in sources],
        slot_mappings=slots,
        block_sizes_tensor=sizes,
        cp_rank=1,
        cp_size=2,
        cp_interleave=3,
    )

    def compute(*args):
        args[4][:, : args[6]].fill_(module.PAD_SLOT_ID)

    native_ops.compute_slot_mappings = Mock(side_effect=compute)
    result = module._compute_slot_mappings(owner, mapping, starts, positions, 3)

    native_ops.compute_slot_mappings.assert_called_once()
    args = native_ops.compute_slot_mappings.call_args.args
    assert all(actual is wanted for actual, wanted in zip(args[0], sources))
    assert all(
        actual is wanted
        for actual, wanted in zip(args[1:6], [mapping, starts, positions, slots, sizes])
    )
    assert args[6:] == (3, 1, 2, 3)
    assert result.shape == (2, 3) and result.data_ptr() == slots.data_ptr()
    assert bool((result == -1).all())
    assert bool((slots[:, 3:] == 777).all())


def test_registration_and_staged_write_order(block_table):
    module, upstream = block_table
    assert upstream.BlockTables.gather_block_tables is module._gather_block_tables
    assert upstream.BlockTables.compute_slot_mappings is module._compute_slot_mappings
    assert upstream.BlockTables.apply_staged_writes is module._apply_staged_writes
    calls = []
    owner = SimpleNamespace(
        block_tables=[
            SimpleNamespace(apply_write=lambda: calls.append("write")) for _ in range(2)
        ],
        num_blocks=SimpleNamespace(copy_to_uva=lambda: calls.append("uva")),
    )
    module._apply_staged_writes(owner)
    assert calls == ["write", "write", "uva"]
