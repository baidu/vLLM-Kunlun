from types import SimpleNamespace

import torch

from vllm_kunlun.v1.worker.gpu import block_table


def test_gather_block_tables_dispatches_all_heterogeneous_groups_once(monkeypatch):
    block_tables = [
        SimpleNamespace(gpu=torch.zeros((4, 17), dtype=torch.int32)),
        SimpleNamespace(gpu=torch.zeros((4, 33), dtype=torch.int32)),
    ]
    input_block_tables = [
        torch.zeros((4, 17), dtype=torch.int32),
        torch.zeros((4, 33), dtype=torch.int32),
    ]
    num_blocks = torch.zeros((2, 4), dtype=torch.int32)
    owner = SimpleNamespace(
        block_tables=block_tables,
        input_block_tables=input_block_tables,
        num_blocks=SimpleNamespace(gpu=num_blocks),
        num_kv_cache_groups=2,
    )
    calls = []

    def fake_gather(src, dst, counts, idx_mapping, num_reqs_padded):
        calls.append((src, dst, counts, idx_mapping, num_reqs_padded))
        return dst

    monkeypatch.setattr(
        torch.ops.xspeedgate_ops,
        "gather_block_tables",
        fake_gather,
    )

    idx_mapping = torch.tensor([2, 0], dtype=torch.int32)
    result = block_table._gather_block_tables(owner, idx_mapping, 3)

    assert len(calls) == 1
    src, dst, counts, actual_mapping, actual_padded = calls[0]
    assert [tensor.shape[1] for tensor in src] == [17, 33]
    assert dst == input_block_tables
    assert counts is num_blocks
    assert actual_mapping is idx_mapping
    assert actual_padded == 3
    assert [tensor.shape for tensor in result] == [
        torch.Size((3, 17)),
        torch.Size((3, 33)),
    ]


def test_compute_slot_mappings_delegates_empty_batch_to_native_op(monkeypatch):
    slot_mappings = torch.full((2, 8), 777, dtype=torch.int64)
    block_sizes = torch.tensor([16, 32], dtype=torch.int32)
    owner = SimpleNamespace(
        block_tables=[
            SimpleNamespace(gpu=torch.zeros((4, 17), dtype=torch.int32)),
            SimpleNamespace(gpu=torch.zeros((4, 33), dtype=torch.int32)),
        ],
        slot_mappings=slot_mappings,
        block_sizes_tensor=block_sizes,
        cp_rank=0,
        cp_size=1,
        cp_interleave=1,
    )
    calls = []

    def fake_compute(*args):
        calls.append(args)
        args[4].fill_(block_table.PAD_SLOT_ID)
        return args[4]

    monkeypatch.setattr(
        torch.ops.xspeedgate_ops,
        "compute_slot_mappings",
        fake_compute,
    )

    result = block_table._compute_slot_mappings(
        owner,
        torch.empty(0, dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.empty(0, dtype=torch.int64),
        num_tokens_padded=3,
    )

    assert len(calls) == 1
