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
    stub_module_tree(
        "vllm.v1.attention.backends.utils",
        NULL_BLOCK_ID=0,
        compute_causal_conv1d_metadata=Mock(),
        mamba_get_block_table_tensor=lambda table, *args: table,
        split_decodes_and_prefills=lambda metadata, decode_threshold: (
            metadata.num_reqs,
            0,
            metadata.num_reqs,
            0,
        ),
    )
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
def test_supported_targets(target, method, architecture, k):
    target.config.speculative_config = SimpleNamespace(
        method=method, draft_model_config=SimpleNamespace(architectures=[architecture])
    )
    target.dimensions.num_spec = k

    result = target.select()

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
        {"mamba_cache_mode": "align"},
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
    ],
)
def test_old_kunlun_ops_keep_the_original_layout(target, missing_symbol):
    delattr(target.operators, missing_symbol)
    assert target.select() is target.spec


def test_non_xpu3_keeps_the_original_layout(target):
    target.operators.gdn_sdnn_supports_current_device.return_value = False
    assert target.select() is target.spec


def _build_ordinary_decode(target, *, full_graph, capture_size):
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3, parallel_drafting=False
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: full_graph),
            max_cudagraph_capture_size=capture_size,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    builder = target.builder_cls(target.select(), [], cfg, torch.device("cpu"))
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
