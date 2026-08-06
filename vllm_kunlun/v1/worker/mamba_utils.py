
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import itertools
import logging
from collections.abc import Callable
from math import prod
from typing import Any

import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))

        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    torch.ops.xspeedgate_ops.batch_memcpy(src_ptrs, dst_ptrs, sizes)


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i in range(len(kv_cache_config.kv_cache_groups)):
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
        if isinstance(kv_cache_spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(kv_cache_spec)
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"
    assert all(mamba_specs[0] == spec for spec in mamba_specs)
    return mamba_group_ids, mamba_specs[0]


@dataclasses.dataclass
class MambaCopyBuffers:
    src_ptrs: CpuGpuBuffer
    dst_ptrs: CpuGpuBuffer
    sizes: CpuGpuBuffer
    mamba_group_ids: list[int]
    mamba_spec: MambaSpec
    offset: int = 0

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaCopyBuffers":
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
        entries_per_req = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        ) * len(copy_funcs)
        n = max_num_reqs * entries_per_req
        return cls(
            src_ptrs=make_buffer(n, dtype=torch.int64),
            dst_ptrs=make_buffer(n, dtype=torch.int64),
            sizes=make_buffer(n, dtype=torch.int64),
            mamba_group_ids=mamba_group_ids,
            mamba_spec=mamba_spec,
        )


def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):
    n = copy_bufs.offset
    if n == 0:
        return
    batch_memcpy(
        copy_bufs.src_ptrs.copy_to_gpu(n),
        copy_bufs.dst_ptrs.copy_to_gpu(n),
        copy_bufs.sizes.copy_to_gpu(n),
    )


def reanchor_spec_to_non_spec_states(
    runner: Any,
    scheduler_output: SchedulerOutput,
) -> int:
    """Move accepted speculative candidates to canonical non-spec state slots.

    In Mamba cache mode ``none``, each request owns one canonical block followed
    by speculative candidate blocks. A speculative GDN step writes conv history
    candidate ``a`` at row offset ``a - 1`` of the canonical block and writes the
    temporal candidate to block ``a - 1``. If the next suffix step has no draft,
    the request switches to non-spec kernels, which read only canonical offset 0.
    Re-anchor only those transitions before attention metadata is built.
    """
    if (
        runner.cache_config.mamba_cache_mode != "none"
        or not runner.speculative_config
        or not runner.model_config.is_hybrid
    ):
        return 0

    scheduled_spec = scheduler_output.scheduled_spec_decode_tokens
    transitions: list[tuple[int, str, int]] = []
    for row, req_id in enumerate(runner.input_batch.req_ids):
        accepted = int(runner.num_accepted_tokens.np[row])
        draft_len = len(scheduled_spec.get(req_id, ()))
        is_decode = (
            runner.input_batch.num_computed_tokens_cpu[row]
            >= runner.input_batch.num_prompt_tokens[row]
        )
        current_is_spec = draft_len > 0 and is_decode
        if accepted > 1 and not current_is_spec:
            transitions.append((row, req_id, accepted))

    if not transitions:
        return 0

    copy_funcs = runner.model.get_mamba_state_copy_func()
    mamba_group_ids, mamba_spec = get_mamba_groups(runner.kv_cache_config)
    num_spec = mamba_spec.num_speculative_blocks
    assert len(copy_funcs) == 2, "GDN re-anchor expects conv and temporal states"

    for group_id in mamba_group_ids:
        layer_names = runner.kv_cache_config.kv_cache_groups[group_id].layer_names
        dest_block_ids: list[int] = []
        source_block_ids: list[int] = []
        offsets: list[int] = []
        for _, req_id, accepted in transitions:
            block_ids = runner.requests[req_id].block_ids[group_id]
            assert (
                1 <= accepted <= len(block_ids)
            ), f"accepted count {accepted} exceeds block table for {req_id}"
            dest_block_ids.append(block_ids[0])
            source_block_ids.append(block_ids[accepted - 1])
            offsets.append(accepted - 1)

        first_attention = runner.compilation_config.static_forward_context[
            layer_names[0]
        ]
        first_conv_state = first_attention.kv_cache[0]
        history_len = first_conv_state.shape[1] - num_spec
        assert history_len > 0

        device = first_conv_state.device
        dest = torch.tensor(dest_block_ids, dtype=torch.long, device=device)
        source = torch.tensor(source_block_ids, dtype=torch.long, device=device)
        offset = torch.tensor(offsets, dtype=torch.long, device=device)
        history_rows = torch.arange(history_len, device=device)
        source_rows = offset[:, None] + history_rows[None, :]

        for layer_name in layer_names:
            attention = runner.compilation_config.static_forward_context[layer_name]
            conv_state, temporal_state = attention.kv_cache
            assert conv_state.shape[1] - num_spec == history_len

            conv_history = conv_state[dest[:, None], source_rows].clone()
            conv_state[dest[:, None], history_rows[None, :]] = conv_history

            temporal_candidates = temporal_state.index_select(0, source)
            temporal_state.index_copy_(0, dest, temporal_candidates)

    rows = [row for row, _, _ in transitions]
    runner.input_batch.num_accepted_tokens_cpu[rows] = 1
    runner.num_accepted_tokens.np[rows] = 1
    row_tensor = torch.tensor(rows, dtype=torch.long, device=runner.device)
    runner.num_accepted_tokens.gpu.index_fill_(0, row_tensor, 1)
    return len(transitions)


def patch_gpu_model_runner(module: Any) -> None:
    """Patch the upstream runner at the post-input-preparation boundary."""
    cls = module.GPUModelRunner
    original = cls._prepare_inputs
    if getattr(original, "_kunlun_spec_reanchor_patched", False):
        return

    def _prepare_inputs_with_reanchor(self, scheduler_output, num_scheduled_tokens):
        result = original(self, scheduler_output, num_scheduled_tokens)
        reanchor_spec_to_non_spec_states(self, scheduler_output)
        return result

    _prepare_inputs_with_reanchor._kunlun_spec_reanchor_patched = True
    cls._prepare_inputs = _prepare_inputs_with_reanchor
    logging.getLogger(__name__).info(
        "[KunlunPlugin] GPUModelRunner speculative state re-anchor patched"
    )


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    # We need to clear mamba_state_idx for resumed requests. When requests are
    # force-preempted (e.g., during reset_prefix_cache / KV cache flush),
    # they appear in resumed_req_ids without a corresponding entry in
    # preempted_req_ids, leaving stale mamba_state_idx entries that can
    # point to block indices beyond the new (smaller) block allocation.
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)

    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


def get_hybrid_attention_mamba_layout(
    kv_cache_shape: tuple[int, ...],
    kv_cache_stride: tuple[int, ...],
    kv_cache_spec: AttentionSpec,
    block_dim: int,
    layer_idx: int,
    kernel_block_size: int,
) -> tuple[tuple[int, ...], int]:
    """
    Compute the stride and storage offset for the hybrid attention+mamba layout.

    Args:
        kv_cache_shape: The shape of the KV cache tensor.
        kv_cache_stride: The stride of the KV cache tensor.
        kv_cache_spec: The specification of the KV cache.
        layer_idx: The index of the layer.
        kernel_num_blocks: The number of kernel blocks.
        kernel_block_size: The size of the kernel block.
    Returns:
        A tuple containing the target stride and storage offset.
    """
    target_stride_list = list(kv_cache_stride)
    storage_offset = 0

    attn_pack_size = kv_cache_spec.pack_size
    # block_dim: 0 means (num_blocks, 2, ...); 1 means (2, num_blocks, ...).
    if block_dim != 0:
        # Hybrid attention+mamba uses (2, num_blocks, ...) logical shape but
        # (num_blocks, 2, ...) physical layout.
        assert kv_cache_shape[0] == 2, (
            "Fail to determine whether the layout is "
            "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
            f"a tensor of shape {kv_cache_shape}"
        )
        assert block_dim == 1
        hidden_size = prod(kv_cache_shape[2:])
        target_stride_list[0] = hidden_size
        target_stride_list[1] = 2 * hidden_size
    # When multiple attention layers share one physical KV cache block
    # (attn_pack_size > 1), scale the block-dim stride by attn_pack_size
    # and compute this layer's element offset within the shared block.
    if attn_pack_size > 1:
        target_stride_list[block_dim] *= attn_pack_size
        dtype_size = get_dtype_size(kv_cache_spec.dtype)
        num_element_per_page = kv_cache_spec.page_size_bytes // dtype_size
        num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
        num_element_per_attn_pack = (
            num_element_per_page // num_blocks_per_kv_block // attn_pack_size
        )
        attn_pack_idx = layer_idx % attn_pack_size
        storage_offset = attn_pack_idx * num_element_per_attn_pack
    return tuple(target_stride_list), storage_offset


def postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    If a blocks is converted from partial block to full block in this step, copy the
    state from the block for running state to the new full block.
    """
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = (
            num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        )
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        aligned_new_computed_tokens = (
            new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        )
        # TODO: how to ensure all blocks that cache_blocks called are cached here?
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)
