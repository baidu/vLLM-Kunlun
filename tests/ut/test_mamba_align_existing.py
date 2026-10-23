# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba registration, cached layouts and request-slot adaptation on CPU."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch


@pytest.fixture
def mamba(load_kunlun_override, stub_module_tree, native_ops):
    class Context:
        def initialize_from_forward_context(self, *args, **kwargs):
            self.is_initialized = True

    layout = stub_module_tree(
        "vllm.model_executor.layers.mamba.mamba_utils",
        is_conv_state_dim_first=lambda: False,
    )
    (module, upstream) = load_kunlun_override(
        "v1/worker/mamba_utils.py",
        "vllm.v1.worker.mamba_utils",
        MambaSpecDecodeGPUContext=Context,
        MambaCopyBuffers=type("MambaCopyBuffers", (), {}),
    )
    return SimpleNamespace(module=module, upstream=upstream, layout=layout)


def make_context(mamba, capacity=8):
    context = mamba.upstream.MambaSpecDecodeGPUContext()
    context.is_initialized = False
    context.num_accepted_tokens_out = torch.zeros(capacity, dtype=torch.int32)
    context.state_elem_sizes = torch.tensor([2, 4], dtype=torch.int32)
    context.state_inner_sizes = torch.tensor([67, 35], dtype=torch.int64)
    context.state_conv_widths = torch.tensor([10, 0], dtype=torch.int32)
    context.state_dim_row_count = torch.tensor([67, 0], dtype=torch.int32)
    context.state_dim_row_stride = torch.tensor([20, 0], dtype=torch.int64)
    context.state_base_addrs = torch.tensor([1000, 2000], dtype=torch.int64)
    context.state_block_strides = torch.tensor([1340, 140], dtype=torch.int64)
    context.state_group_indices = torch.zeros(2, dtype=torch.int32)
    context.block_table_ptrs = torch.tensor([3000], dtype=torch.int64)
    context.block_table_stride_req = 12
    context.block_size = 128
    context.initialize_from_forward_context()
    return context


def test_real_module_registers_existing_entry_points_once(mamba):
    (module, upstream) = (mamba.module, mamba.upstream)
    context = upstream.MambaSpecDecodeGPUContext
    initializer = context.initialize_from_forward_context
    assert (
        upstream.preprocess_mamba_align_fused_kernel[1,]
        is module._preprocess_mamba_align
    )
    assert context.run_fused_precopy is module._run_fused_precopy
    assert context.run_fused_postprocess_align is module._run_fused_postprocess_align
    assert upstream.batch_memcpy is module.batch_memcpy
    assert (
        upstream.MambaCopyBuffers.create.__func__ is module._mamba_copy_buffers_create
    )
    module._patch_mamba_align_xspeedgate()
    assert context.initialize_from_forward_context is initializer


@pytest.mark.parametrize("dim_first", [False, True])
def test_layout_cache_preserves_original_metadata_and_rebuilds(mamba, dim_first):
    mamba.layout.is_conv_state_dim_first = lambda: dim_first
    context = make_context(mamba)
    (pre, post) = (
        context._kunlun_precopy_metadata,
        context._kunlun_postprocess_metadata,
    )
    expected = (
        ([2, 4], [67, 0], [20, 0]) if dim_first else ([134, 4], [1, 0], [1340, 0])
    )
    assert [value.tolist() for value in pre] == list(expected)
    assert pre[0].dtype == torch.int64 and post[0].dtype == torch.int32
    assert torch.equal(post[0], pre[0].int())
    assert post[1] is pre[1] and post[2] is pre[2]
    assert context.state_elem_sizes.tolist() == [2, 4]
    assert context.state_dim_row_count.tolist() == [67, 0]
    assert context.state_dim_row_stride.tolist() == [20, 0]
    buffers = context._kunlun_align_buffers
    context.initialize_from_forward_context()
    assert (
        context._kunlun_precopy_metadata is pre
        and context._kunlun_align_buffers is buffers
    )
    context.is_initialized = False
    context.state_elem_sizes[0] = 4
    context.initialize_from_forward_context()
    assert context._kunlun_precopy_metadata is not pre
    assert context._kunlun_precopy_metadata[0].tolist() == (
        [4, 4] if dim_first else [268, 4]
    )


def test_preprocess_uses_existing_operator_and_preserves_buffers(mamba, native_ops):
    arrays = [torch.zeros(8, dtype=torch.int32) for _ in range(7)]
    native_ops.preprocess_mamba_align_fused_kernel = Mock()
    kernel = mamba.upstream.preprocess_mamba_align_fused_kernel
    kernel[1,](*arrays, 3, MAMBA_BLOCK_SIZE=128, BLOCK_SIZE=256)
    native_ops.preprocess_mamba_align_fused_kernel.assert_called_once()
    args = native_ops.preprocess_mamba_align_fused_kernel.call_args.args
    assert all((actual is wanted for (actual, wanted) in zip(args[:7], arrays)))
    assert args[7:] == (3, 128)


def test_precopy_uses_cached_metadata_and_active_mapping(mamba, native_ops):
    context = make_context(mamba)
    mapping = torch.tensor([6, 0, -1, -1], dtype=torch.int32)
    (state, source, bias) = [torch.zeros(8, dtype=torch.int32) for _ in range(3)]
    native_ops.precopy_mamba_align_fused_kernel = Mock()
    context.run_fused_precopy(3, state, source, bias, mapping)
    native_ops.precopy_mamba_align_fused_kernel.assert_called_once()
    args = native_ops.precopy_mamba_align_fused_kernel.call_args.args
    expected = [
        state,
        source,
        bias,
        context.block_table_ptrs,
        context.block_table_stride_req,
        context.state_base_addrs,
        context.state_block_strides,
        context._kunlun_precopy_metadata[0],
        context.state_inner_sizes,
        context.state_conv_widths,
        context.state_group_indices,
        context._kunlun_precopy_metadata[1],
        context._kunlun_precopy_metadata[2],
    ]
    assert all((actual is wanted for (actual, wanted) in zip(args[:13], expected)))
    assert args[13].tolist() == [6, 0, -1] and args[13].data_ptr() == mapping.data_ptr()
    assert args[14:] == (3, 1024, True, 1)


@pytest.mark.parametrize(
    "dim_first,mapping_values",
    [(False, [6, -1, 0]), (False, [-1, -1, -1]), (True, [7, 2, 0])],
)
def test_postprocess_packs_slots_and_writes_only_valid_results(
    mamba, native_ops, dim_first, mapping_values
):
    mamba.layout.is_conv_state_dim_first = lambda: dim_first
    context = make_context(mamba)
    accepted = torch.arange(10, 18, dtype=torch.int32)
    state = torch.arange(20, 28, dtype=torch.int32)
    computed = torch.arange(30, 38, dtype=torch.int32)
    (before, old_state, old_computed) = (
        accepted.clone(),
        state.clone(),
        computed.clone(),
    )
    mapping = torch.tensor(mapping_values, dtype=torch.int32)

    def postprocess(*args):
        slots = mapping.clamp_min(0).long()
        assert torch.equal(args[0], before[slots])
        assert torch.equal(args[1], old_state[slots])
        assert torch.equal(args[3], old_computed[slots])
        assert args[0] is args[2] and args[0] is args[4]
        assert (
            args[5] is context.block_table_ptrs
            and args[6] == context.block_table_stride_req
        )
        assert (
            args[7] is context.state_base_addrs
            and args[8] is context.state_block_strides
        )
        assert args[9] is context._kunlun_postprocess_metadata[0]
        assert (
            args[10] is context.state_inner_sizes
            and args[11] is context.state_conv_widths
        )
        assert args[12] is context.state_group_indices
        assert args[13] is context._kunlun_postprocess_metadata[1]
        assert args[14] is context._kunlun_postprocess_metadata[2]
        assert torch.equal(args[15], args[0])
        assert args[16].tolist() == [
            index if slot >= 0 else -1 for (index, slot) in enumerate(mapping_values)
        ]
        assert args[17:] == (3, 128, True, True)
        args[15].copy_(torch.tensor([101, 202, 303], dtype=torch.int32))

    native_ops.postprocess_mamba = Mock(side_effect=postprocess)
    cache = context._kunlun_align_buffers
    accepted_pointer = accepted.data_ptr()
    context.run_fused_postprocess_align(3, accepted, state, computed, mapping)
    native_ops.postprocess_mamba.assert_called_once()
    expected = before.clone()
    for slot, value in zip(mapping_values, [101, 202, 303]):
        if slot >= 0:
            expected[slot] = value
    assert torch.equal(accepted, expected) and accepted.data_ptr() == accepted_pointer
    assert torch.equal(state, old_state) and torch.equal(computed, old_computed)
    assert context._kunlun_align_buffers is cache


def test_empty_or_uninitialized_context_does_not_call_operators(mamba, native_ops):
    context = make_context(mamba)
    empty = torch.empty(0, dtype=torch.int32)
    mamba.module._preprocess_mamba_align(*[empty] * 7, 0, MAMBA_BLOCK_SIZE=128)
    context.run_fused_precopy(0, None, None, None, None)
    context.run_fused_postprocess_align(0, None, None, None, None)
    context.run_fused_postprocess_align(3, empty, empty, empty, torch.full((3,), -1))
    context.is_initialized = False
    context.run_fused_precopy(3, None, None, None, None)
    context.run_fused_postprocess_align(3, None, None, None, None)
    assert vars(native_ops) == {}
