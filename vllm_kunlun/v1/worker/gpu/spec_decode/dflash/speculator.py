# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native input preparation for the upstream DFlash/DSpark runner."""

from __future__ import annotations

import torch
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as _up

_reported_xsg_buffer = False
_reported_xsg_multi = False
# Per-worker state. DSpark invokes preparation serially for every KV group.
# Values are device tensors whose addresses are stable for the process lifetime.
_multi_pending: dict[
    int, list[tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]]
] = {}
_multi_meta_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}


def prepare_dflash_inputs(
    input_buffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    """Fill official DFlash buffers with the same anchor/query contract."""
    num_reqs = int(input_batch.num_reqs)
    if num_reqs <= 0:
        raise ValueError("DFlash input preparation requires at least one request")

    # Write graph-visible buffers and device KV LOD with the native operator.
    op = torch.ops.xspeedgate_ops.dflash_prepare_inputs
    from vllm_kunlun.v1.worker.gpu.spec_decode.dspark_buffers import (
        get_or_create_kv_lod,
    )

    kv_lod = get_or_create_kv_lod(input_buffers.seq_lens, max_num_reqs)
    global _reported_xsg_buffer
    if not _reported_xsg_buffer:
        import logging

        logging.getLogger("vllm_kunlun").info(
            "[KunlunPlugin] DSpark XSpeedGate input prep buffers: seq_ptr=%d "
            "lod_ptr=%d seq_capacity=%d",
            input_buffers.seq_lens.data_ptr(),
            kv_lod.data_ptr(),
            input_buffers.seq_lens.numel(),
        )
        _reported_xsg_buffer = True

    target_num_tokens = int(input_batch.num_tokens)
    # Two independent implementations are retained. The flat rewrite wins
    # for high-concurrency decode and a single long prefill; request/cluster
    # sharding wins for medium context chunks.
    use_flat = (num_reqs >= 16 and target_num_tokens <= num_reqs * 16) or (
        num_reqs == 1 and target_num_tokens >= 2048
    )
    implementation = 1 if use_flat else 0

    # The official DSpark loop passes consecutive rows of
    # _context_slot_mappings. Defer the early rows and emit every KV-group
    # mapping in one launch on the final row. Common buffers are not used
    # until that official loop has completed.
    multi_op = torch.ops.xspeedgate_ops.dflash_prepare_inputs_multi
    base = getattr(context_slot_mapping, "_base", None)
    multi_layout = False
    row = 0
    num_groups = 0
    pending_key = 0
    if (
        base is not None
        and base.ndim == 2
        and base.is_contiguous()
        and context_slot_mapping.ndim == 1
        and context_slot_mapping.is_contiguous()
        and base.shape[1] == context_slot_mapping.numel()
    ):
        row_stride = base.stride(0)
        relative_offset = context_slot_mapping.storage_offset() - base.storage_offset()
        if (
            row_stride > 0
            and relative_offset >= 0
            and relative_offset % row_stride == 0
        ):
            row = relative_offset // row_stride
            num_groups = int(base.shape[0])
            # A one-row layout is the normal path for Qwen3.5 DSpark:
            # all five draft attention layers share one official KV group.
            # Keep it on the direct single-group operator to avoid pointer
            # metadata setup and the generalized multi-group dispatch.
            multi_layout = 1 < num_groups <= 16 and 0 <= row < num_groups
            pending_key = base.untyped_storage().data_ptr()

    if multi_layout:
        if row == 0:
            _multi_pending[pending_key] = []
        pending = _multi_pending.setdefault(pending_key, [])
        # Official vLLM invokes rows serially in 0..G-1 order. Fail
        # closed if that contract changes: a partial fallback after early
        # rows were deferred would otherwise expose stale slot mappings.
        if len(pending) == row:
            pending.append(
                (
                    block_table,
                    int(block_size),
                    query_slot_mapping,
                    context_slot_mapping,
                )
            )
            if row + 1 < num_groups:
                return
            groups = pending
            _multi_pending.pop(pending_key, None)
            if len(groups) == num_groups:
                bt_ptrs = tuple(x[0].data_ptr() for x in groups)
                bt_strides = tuple(int(x[0].shape[1]) for x in groups)
                block_sizes = tuple(x[1] for x in groups)
                query_ptrs = tuple(x[2].data_ptr() for x in groups)
                context_ptrs = tuple(x[3].data_ptr() for x in groups)
                cache_key = (
                    input_buffers.input_ids.device.index,
                    bt_ptrs,
                    bt_strides,
                    block_sizes,
                    query_ptrs,
                    context_ptrs,
                )
                meta = _multi_meta_cache.get(cache_key)
                if meta is None:
                    device = input_buffers.input_ids.device
                    meta = (
                        torch.tensor(bt_ptrs, dtype=torch.int64, device=device),
                        torch.tensor(bt_strides, dtype=torch.int32, device=device),
                        torch.tensor(block_sizes, dtype=torch.int32, device=device),
                        torch.tensor(query_ptrs, dtype=torch.int64, device=device),
                        torch.tensor(context_ptrs, dtype=torch.int64, device=device),
                    )
                    _multi_meta_cache[cache_key] = meta
                multi_op(
                    input_batch.positions,
                    input_batch.query_start_loc,
                    input_batch.idx_mapping,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    *meta,
                    input_buffers.input_ids,
                    input_buffers.positions,
                    input_buffers.query_start_loc,
                    input_buffers.seq_lens,
                    context_positions,
                    sample_indices,
                    sample_pos,
                    sample_idx_mapping,
                    kv_lod,
                    num_groups,
                    num_reqs,
                    target_num_tokens,
                    parallel_drafting_token_id,
                    num_query_per_req,
                    num_speculative_steps,
                    max_num_reqs,
                    max_num_tokens,
                    max_model_len,
                    sample_from_anchor,
                    implementation,
                )
                global _reported_xsg_multi
                if not _reported_xsg_multi:
                    import logging

                    logging.getLogger("vllm_kunlun").info(
                        "[KunlunPlugin] DSpark fused %d KV-group input "
                        "preparation enabled (implementation=%d)",
                        num_groups,
                        implementation,
                    )
                    _reported_xsg_multi = True
                return
        else:
            expected = len(pending)
            _multi_pending.pop(pending_key, None)
            raise RuntimeError(
                "DSpark KV-group preparation order changed: "
                f"expected row {expected}, received row {row}"
            )

    # Direct path for one KV group or independently allocated group rows.
    write_common = context_slot_mapping.storage_offset() == 0
    op(
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        block_table,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        kv_lod,
        num_reqs,
        target_num_tokens,
        block_size,
        parallel_drafting_token_id,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        sample_from_anchor,
        write_common,
        implementation,
    )
    return


_up.prepare_dflash_inputs = prepare_dflash_inputs
