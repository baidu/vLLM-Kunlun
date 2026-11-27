# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise single-slot GDN layout selection and state handoff on CPU."""

from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@dataclass(frozen=True)
class MambaSpec:
    block_size: int
    shapes: tuple
    dtypes: tuple
    page_size_padded: int | None
    num_speculative_blocks: int
    mamba_cache_mode: str = "none"


@pytest.fixture
def target(load_kunlun_override, stub_module, stub_module_tree):
    operators = stub_module(
        "kunlun_ops",
        gdn_sdnn_supports_current_device=Mock(return_value=True),
        fused_gdn_commit_chunk_verify=Mock(),
        gdn_commit_accepted=Mock(),
        gdn_clear_pending=Mock(),
        gdn_align_precopy=Mock(),
        gdn_align_postprocess=Mock(),
    )
    recurrent, _ = load_kunlun_override("ops/fla/fused_recurrent.py", "xspeedgate_ops")
    stub_module(
        "vllm_kunlun.ops.fla.fused_recurrent",
        gdn_sdnn_is_available=recurrent.gdn_sdnn_is_available,
    )

    class Builder:
        def __class_getitem__(cls, metadata_type):
            return cls

        def _init_reorder_batch_threshold(self, *args):
            pass

    stub_module_tree("vllm.config", VllmConfig=object)
    stub_module_tree("vllm.logger", init_logger=lambda name: Mock())
    stub_module_tree(
        "vllm.v1.kv_cache_interface", AttentionSpec=object, MambaSpec=MambaSpec
    )
    stub_module_tree(
        "vllm.v1.attention.backend",
        AttentionBackend=object,
        AttentionCGSupport=SimpleNamespace(UNIFORM_BATCH="uniform"),
        AttentionMetadataBuilder=Builder,
        CommonAttentionMetadata=SimpleNamespace,
    )

    def split_decodes_and_prefills(metadata, decode_threshold):
        lengths = metadata.query_start_loc_cpu.diff()
        num_decodes = ((lengths > 0) & (lengths <= decode_threshold)).sum().item()
        prefill_lengths = lengths[lengths > decode_threshold]
        return (
            num_decodes,
            prefill_lengths.numel(),
            num_decodes,
            prefill_lengths.sum().item(),
        )

    stub_module_tree(
        "vllm.v1.attention.backends.utils",
        NULL_BLOCK_ID=0,
        compute_causal_conv1d_metadata=Mock(return_value=(None, None, None)),
        mamba_get_block_table_tensor=lambda table, *args: table,
        split_decodes_and_prefills=split_decodes_and_prefills,
    )
    stub_module_tree(
        "vllm.model_executor.layers.fla.ops.index",
        prepare_chunk_indices=Mock(return_value=torch.empty((0, 2), dtype=torch.int32)),
        prepare_chunk_offsets=Mock(return_value=torch.zeros(1, dtype=torch.int32)),
    )
    stub_module_tree("vllm.model_executor.layers.fla.ops.utils", FLA_CHUNK_SIZE=64)
    backend, _ = load_kunlun_override(
        "v1/attention/backends/gdn_attn.py", "vllm.v1.attention.backends.gdn_attn"
    )

    spec = MambaSpec(
        block_size=128,
        shapes=((4, 512), (8, 128, 128)),
        dtypes=(torch.float16, torch.float16),
        page_size_padded=2**20,
        num_speculative_blocks=3,
    )
    config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(dtype=torch.float16),
        attention_config=SimpleNamespace(backend=None),
        speculative_config=SimpleNamespace(method="mtp"),
    )
    dimensions = SimpleNamespace(
        num_spec=3, key_heads=4, value_heads=8, head_dims=(128, 128)
    )

    def select(cache_spec=spec):
        return backend.get_gdn_sdnn_cache_spec(cache_spec, config, **vars(dimensions))

    return SimpleNamespace(
        dimensions=dimensions,
        config=config,
        spec=spec,
        select=select,
        operators=operators,
        builder_cls=backend.GDNAttentionMetadataBuilder,
    )


@pytest.mark.parametrize(
    ("method", "architecture"),
    [
        ("mtp", "Qwen3_5MTP"),
        ("dflash", "DFlash2DraftModel"),
        ("dflash", "DFlash2T1ResidualDraftModel"),
    ],
)
@pytest.mark.parametrize("k", [1, 2, 3, 7])
@pytest.mark.parametrize("mode", ["none", "align"])
def test_supported_targets(target, method, architecture, k, mode):
    target.config.speculative_config = SimpleNamespace(
        method=method, draft_model_config=SimpleNamespace(architectures=[architecture])
    )
    target.dimensions.num_spec = k

    result = target.select(replace(target.spec, mamba_cache_mode=mode))

    assert result.num_speculative_blocks == 0
    assert result.shapes[2:] == (
        (4, k + 1, 128),
        (8, k + 1, 128),
        (8, k + 1),
        (1,),
    )
    assert result.dtypes[2:] == (
        torch.float16,
        torch.float16,
        torch.float32,
        torch.int32,
    )


@pytest.mark.parametrize(
    ("method", "architectures"),
    [
        ("ngram", []),
        ("eagle", ["DFlash2DraftModel"]),
        ("eagle3", []),
        ("draft_model", []),
        ("dflash", ["DFlashDraftModel"]),
        ("dflash", []),
        ("dflash", None),
    ],
)
def test_other_proposers_keep_the_original_layout(target, method, architectures):
    target.config.speculative_config = SimpleNamespace(
        method=method, draft_model_config=SimpleNamespace(architectures=architectures)
    )
    assert target.select() is target.spec


@pytest.mark.parametrize(
    "changes",
    [
        {"mamba_cache_mode": "all"},
        {"page_size_padded": None},
        {"page_size_padded": 16384},
        {"dtypes": (torch.float16, torch.float32)},
    ],
)
def test_incompatible_cache_layout(target, changes):
    spec = replace(target.spec, **changes)
    assert target.select(spec) is spec


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("config", "use_v2_model_runner", False),
        ("dimensions", "num_spec", 0),
        ("dimensions", "num_spec", 8),
        ("dimensions", "head_dims", (64, 128)),
        ("dimensions", "head_dims", (128, 64)),
        ("dimensions", "value_heads", 56),
        ("dimensions", "key_heads", 3),
        ("config", "speculative_config", None),
        ("model", "dtype", torch.bfloat16),
        ("attention", "backend", "TREE_ATTN"),
        ("speculative", "speculative_token_tree", [(0,), (1,)]),
        ("speculative", "tree_choices", [[1, 2]]),
        ("speculative", "use_tree_attention", True),
    ],
)
def test_incompatible_runner_shape_or_dtype(target, owner, field, value):
    objects = {
        "config": target.config,
        "dimensions": target.dimensions,
        "model": target.config.model_config,
        "attention": target.config.attention_config,
        "speculative": target.config.speculative_config,
    }
    setattr(objects[owner], field, value)
    assert target.select() is target.spec


@pytest.mark.parametrize(
    "missing_symbol",
    [
        "gdn_sdnn_supports_current_device",
        "fused_gdn_commit_chunk_verify",
        "gdn_commit_accepted",
        "gdn_clear_pending",
    ],
)
def test_old_kunlun_ops_keep_the_original_layout(target, missing_symbol):
    delattr(target.operators, missing_symbol)
    assert target.select() is target.spec


def test_non_xpu3_keeps_the_original_layout(target):
    target.operators.gdn_sdnn_supports_current_device.return_value = False
    assert target.select() is target.spec


@pytest.mark.parametrize("missing", ["gdn_align_precopy", "gdn_align_postprocess"])
def test_align_requires_both_migration_kernels(target, missing):
    delattr(target.operators, missing)
    spec = replace(target.spec, mamba_cache_mode="align")
    assert target.select(spec) is spec
    assert target.select().num_speculative_blocks == 0


def _make_builder(
    target, *, full_graph=False, capture_size=0, mode="none", compact=True
):
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3, parallel_drafting=False
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: full_graph),
            max_cudagraph_capture_size=capture_size,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=SimpleNamespace(mamba_cache_mode=mode),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    spec = replace(target.spec, mamba_cache_mode=mode)
    if compact:
        spec = target.select(spec)
    return target.builder_cls(spec, [], cfg, torch.device("cpu"))


def _build_ordinary_decode(target, *, full_graph, capture_size):
    builder = _make_builder(target, full_graph=full_graph, capture_size=capture_size)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=cu,
        query_start_loc_cpu=cu,
        seq_lens=cu[1:] + 7,
        compute_num_computed_tokens=lambda: torch.tensor([7, 8], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=capture_size if full_graph else 2,
        max_query_len=1,
        max_seq_len=9,
        block_table_tensor=torch.tensor([[5], [3]], dtype=torch.int32),
        slot_mapping=torch.zeros(capture_size if full_graph else 2, dtype=torch.int64),
    )
    return builder, metadata


def test_mtp_eager_handoff(target):
    builder, metadata = _build_ordinary_decode(target, full_graph=False, capture_size=0)
    counts = torch.tensor([3, 2], dtype=torch.int32)

    result = builder.build(0, metadata, counts, torch.full((2,), -1, dtype=torch.int32))

    assert builder.decode_cudagraph_max_bs == 0
    assert builder.num_accepted_tokens.shape == (4,)
    assert result.non_spec_commit_slots.tolist() == [5, 3]
    assert result.non_spec_accepted.tolist() == [3, 2]


def test_mtp_graph_handoff_keeps_stable_buffers(target):
    builder, metadata = _build_ordinary_decode(target, full_graph=True, capture_size=4)
    counts = torch.tensor([3, 2], dtype=torch.int32)
    no_drafts = torch.full((2,), -1, dtype=torch.int32)

    first = builder.build(0, metadata, counts, no_drafts)
    assert first.non_spec_commit_slots.tolist() == [5, 3, -1, -1]
    assert first.non_spec_accepted.tolist() == [3, 2, 0, 0]

    metadata.block_table_tensor.copy_(torch.tensor([[4], [6]], dtype=torch.int32))
    counts.copy_(torch.tensor([2, 1], dtype=torch.int32))
    second = builder.build(0, metadata, counts, no_drafts)

    assert (
        first.non_spec_commit_slots.data_ptr()
        == second.non_spec_commit_slots.data_ptr()
    )
    assert first.non_spec_accepted.data_ptr() == second.non_spec_accepted.data_ptr()
    assert second.non_spec_commit_slots.tolist() == [4, 6, -1, -1]
    assert second.non_spec_accepted.tolist() == [2, 1, 0, 0]


def _common_metadata(
    query_lens, slots, *, mode="none", context_lens=None, num_actual_tokens=None
):
    lengths = torch.tensor(query_lens, dtype=torch.int32)
    cu = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()))
    context = torch.tensor(
        context_lens if context_lens is not None else [7] * len(query_lens),
        dtype=torch.int32,
    )
    table = torch.tensor(slots, dtype=torch.int32).view(-1, 1)
    if mode == "align":
        # Snapshot columns must not widen the compact single running state.
        table = torch.cat(
            (table, torch.full((len(slots), 3), 123, dtype=torch.int32)), 1
        )
    num_tokens = int(cu[-1]) if num_actual_tokens is None else num_actual_tokens
    return SimpleNamespace(
        query_start_loc=cu,
        query_start_loc_cpu=cu,
        seq_lens=context + lengths,
        compute_num_computed_tokens=lambda: context,
        num_reqs=len(query_lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=int((context + lengths).max()),
        block_table_tensor=table,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
    )


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("variable_lengths", [False, True])
def test_spec_decode_reuses_batch_data_and_keeps_group_slots(
    target, mode, full_graph, variable_lengths
):
    builders = [
        _make_builder(target, full_graph=full_graph, capture_size=12, mode=mode)
        for _ in range(2)
    ]
    lengths = [3, 2, 0] if variable_lengths else [4, 4, 0]
    common = [
        _common_metadata(
            lengths,
            slots,
            mode=mode,
            num_actual_tokens=12 if full_graph else None,
        )
        for slots in ([5, 0, 77], [0, 9, 88])
    ]
    drafts = torch.tensor([n - 1 for n in lengths], dtype=torch.int32)
    counts = torch.tensor([3, 1, 1], dtype=torch.int32)
    use_graph_buffers = full_graph and not variable_lengths
    first_pointers = None
    for step in range(2):
        if step:
            for item, slots in zip(common, ([0, 6, 77], [7, 0, 88])):
                item.block_table_tensor[:, 0].copy_(torch.tensor(slots))
            counts[:2].copy_(torch.tensor([2, 3]))
        result0 = builders[0].build(0, common[0], counts, drafts)
        result1 = builders[1].reuse_metadata(result0, common[1])
        rows = 12 if use_graph_buffers else 2
        for result, item in zip((result0, result1), common):
            slots = item.block_table_tensor[:2, 0].tolist()
            assert result.spec_state_indices_tensor.shape == (rows, 1)
            assert result.spec_state_indices_tensor[:2, 0].tolist() == slots
            assert result.spec_conv_state_indices_tensor.tolist() == [
                slot if slot else -1 for slot in slots
            ] + [-1] * (rows - 2)
            assert result.num_accepted_tokens.tolist() == counts[:2].tolist() + [1] * (
                rows - 2
            )
            assert result.spec_state_indices_tensor.is_contiguous()
        assert result0.num_accepted_tokens is result1.num_accepted_tokens
        assert result0.spec_query_start_loc is result1.spec_query_start_loc
        assert result0.spec_sequence_masks is result1.spec_sequence_masks
        if variable_lengths:
            assert result0.spec_pad_gather_idx is result1.spec_pad_gather_idx
            assert result0.spec_pad_gather_idx is not None
        pointers = [
            (
                result.spec_state_indices_tensor.data_ptr(),
                result.spec_conv_state_indices_tensor.data_ptr(),
                result.num_accepted_tokens.data_ptr(),
            )
            for result in (result0, result1)
        ]
        assert pointers[0][0] != pointers[1][0]
        assert pointers[0][1] != pointers[1][1]
        if use_graph_buffers:
            for result in (result0, result1):
                assert torch.count_nonzero(result.spec_state_indices_tensor[2:]) == 0
            if first_pointers is not None:
                assert pointers == first_pointers
            first_pointers = pointers


@pytest.mark.parametrize("mode", ["none", "align"])
def test_spec_graph_capture_metadata_can_be_reused(target, mode):
    builders = [
        _make_builder(target, full_graph=True, capture_size=12, mode=mode)
        for _ in range(2)
    ]
    common = [
        _common_metadata([4, 4, 0], slots, mode=mode, num_actual_tokens=12)
        for slots in ([5, 0, 77], [0, 9, 88])
    ]
    result0 = builders[0].build_for_cudagraph_capture(common[0])
    result1 = builders[1].reuse_metadata(result0, common[1])
    assert result0.num_accepted_tokens is result1.num_accepted_tokens
    assert result1.num_accepted_tokens.tolist() == [4, 4] + [1] * 10
    assert result0.spec_conv_state_indices_tensor.tolist() == [5, -1] + [-1] * 10
    assert result1.spec_conv_state_indices_tensor.tolist() == [-1, 9] + [-1] * 10


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("live", [1, 2])
def test_decode_handoff_reuses_counts_and_keeps_stable_group_slots(
    target, mode, full_graph, live
):
    builders = [
        _make_builder(target, full_graph=full_graph, capture_size=4, mode=mode)
        for _ in range(2)
    ]
    common = [
        _common_metadata(
            [1, 1, 0],
            slots,
            mode=mode,
            num_actual_tokens=4 if full_graph else None,
        )
        for slots in ([5, 0, 77], [0, 9, 88])
    ]
    drafts = torch.full((3,), -1, dtype=torch.int32)
    first_results = None
    for step in range(2):
        if step:
            for item, slots in zip(common, ([0, 6, 77], [7, 0, 88])):
                item.block_table_tensor[:, 0].copy_(torch.tensor(slots))
        counts = torch.tensor([2, 1] if step else [3, 2], dtype=torch.int32)[:live]
        result0 = builders[0].build(0, common[0], counts, drafts)
        result1 = builders[1].reuse_metadata(result0, common[1])
        rows = 4 if full_graph else live
        for result, item in zip((result0, result1), common):
            slots = item.block_table_tensor[:live, 0].tolist()
            assert result.non_spec_commit_slots.tolist() == [
                slot if slot else -1 for slot in slots
            ] + [-1] * (rows - live)
            assert result.non_spec_accepted.tolist() == counts.tolist() + [0] * (
                rows - live
            )
            assert result.non_spec_commit_live == live
            if full_graph:
                assert result.non_spec_query_start_loc.tolist() == [0, 1, 2, 2, 2]
                assert (
                    result.non_spec_state_indices_tensor.tolist()
                    == item.block_table_tensor[:2, 0].tolist() + [0, 0]
                )
        assert (
            result0.non_spec_commit_slots.data_ptr()
            != result1.non_spec_commit_slots.data_ptr()
        )
        assert result0.non_spec_accepted is result1.non_spec_accepted
        if first_results is not None:
            for previous, current in zip(first_results, (result0, result1)):
                assert (
                    previous.non_spec_commit_slots.data_ptr()
                    == current.non_spec_commit_slots.data_ptr()
                )
                assert (
                    previous.non_spec_accepted.data_ptr()
                    == current.non_spec_accepted.data_ptr()
                )
                assert torch.equal(
                    previous.non_spec_commit_slots, current.non_spec_commit_slots
                )
                assert torch.equal(
                    previous.non_spec_accepted, current.non_spec_accepted
                )
        first_results = (result0, result1)


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("compact", [False, True])
def test_decode_over_capture_capacity_keeps_eager_metadata(target, mode, compact):
    builders = [
        _make_builder(
            target, full_graph=True, capture_size=2, mode=mode, compact=compact
        )
        for _ in range(2)
    ]
    common = [
        _common_metadata([1, 1, 0], slots, mode=mode, num_actual_tokens=4)
        for slots in ([5, 0, 77], [0, 9, 88])
    ]
    result0 = builders[0].build(
        0,
        common[0],
        torch.tensor([3, 2], dtype=torch.int32),
        torch.full((3,), -1, dtype=torch.int32),
    )
    result1 = builders[1].reuse_metadata(result0, common[1])
    for result, item in zip((result0, result1), common):
        assert result.non_spec_query_start_loc.tolist() == [0, 1, 2, 2]
        assert (
            result.non_spec_state_indices_tensor.tolist()
            == item.block_table_tensor[:, 0].tolist()
        )
        if compact:
            assert result.non_spec_accepted.tolist() == [3, 2]
            assert result.non_spec_commit_slots.numel() == 2


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize(
    ("lengths", "context", "drafts", "accepted", "non_spec_rows", "history_rows"),
    [
        pytest.param([3, 2, 0], [0, 0, 9], None, [3, 2, 1], [0, 1, 2], [], id="fresh"),
        pytest.param([3, 2], [7, 8], None, [3, 2], [0, 1], [0, 1], id="history"),
        pytest.param(
            [3, 2, 2, 0],
            [7, 0, 9, 10],
            None,
            [3, 2, 1, 4],
            [0, 1, 2, 3],
            [0, 2],
            id="mixed",
        ),
        pytest.param(
            [4, 3, 1, 0],
            [7, 0, 9, 10],
            [3, -1, -1, -1],
            [3, 2, 1, 4],
            [1, 2, 3],
            [2],
            id="spec-and-prefill",
        ),
        pytest.param(
            [4, 3, 1],
            [7, 8, 9],
            [3, -1, -1],
            [3, 2, 1],
            [1, 2],
            [1, 2],
            id="spec-and-history",
        ),
        pytest.param(
            [3, 2, 0], [0, 0, 9], None, None, [0, 1, 2], [], id="fresh-without-handoff"
        ),
        pytest.param(
            [3, 2], [7, 8], None, None, [0, 1], [0, 1], id="history-without-handoff"
        ),
        pytest.param(
            [3, 2, 2, 0],
            [7, 0, 9, 10],
            None,
            None,
            [0, 1, 2, 3],
            [0, 2],
            id="mixed-without-handoff",
        ),
    ],
)
def test_prefill_reuses_batch_data_and_selects_each_groups_history(
    target, mode, lengths, context, drafts, accepted, non_spec_rows, history_rows
):
    builders = [_make_builder(target, mode=mode) for _ in range(2)]
    group_slots = ([5, 0, 7, 77], [0, 9, 0, 88])
    common = [
        _common_metadata(
            lengths, slots[: len(lengths)], mode=mode, context_lens=context
        )
        for slots in group_slots
    ]
    counts = torch.tensor(accepted, dtype=torch.int32) if accepted is not None else None
    draft_counts = (
        torch.tensor(drafts, dtype=torch.int32) if drafts is not None else None
    )
    result0 = builders[0].build(0, common[0], counts, draft_counts)
    result1 = builders[1].reuse_metadata(result0, common[1])
    assert result0.num_prefills > 0
    assert result0.non_spec_accepted is result1.non_spec_accepted
    assert result0.prefill_fresh_mask is result1.prefill_fresh_mask
    assert result0.has_initial_state_cpu is result1.has_initial_state_cpu
    assert result0.chunk_indices is result1.chunk_indices
    for result, slots in zip((result0, result1), group_slots):
        expected_slots = [slots[row] for row in non_spec_rows]
        assert result.non_spec_state_indices_tensor.tolist() == expected_slots
        assert result.non_spec_state_indices_tensor_cpu.tolist() == expected_slots
        assert result.non_spec_state_indices_tensor.is_contiguous()
        assert result.non_spec_state_indices_tensor_cpu.is_contiguous()
        if history_rows:
            assert result.prefill_zero_state is None
            assert result.prefill_state_slots.tolist() == expected_slots
            assert result.prefill_state_slots.is_contiguous()
            if len(history_rows) != len(non_spec_rows):
                assert result.prefill_fresh_mask.flatten().tolist() == [
                    row not in history_rows for row in non_spec_rows
                ]
            else:
                assert result.prefill_fresh_mask is None
            if accepted is not None:
                assert result.non_spec_accepted.tolist() == [
                    accepted[row] for row in history_rows
                ]
                assert result.non_spec_commit_slots.tolist() == [
                    slots[row] if slots[row] else -1 for row in history_rows
                ]
                assert result.non_spec_commit_slots.is_contiguous()
        else:
            assert result.prefill_zero_state is result0.prefill_zero_state
            assert result.prefill_zero_state.shape == (len(non_spec_rows), 8, 128, 128)
            assert torch.count_nonzero(result.prefill_zero_state) == 0
            assert result.prefill_state_slots is None
        if not history_rows or accepted is None:
            assert result.non_spec_accepted is None
            assert result.non_spec_commit_slots is None
    if history_rows:
        assert (
            result0.prefill_state_slots.data_ptr()
            != result1.prefill_state_slots.data_ptr()
        )
        if accepted is not None:
            assert (
                result0.non_spec_commit_slots.data_ptr()
                != result1.non_spec_commit_slots.data_ptr()
            )
