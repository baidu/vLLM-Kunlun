# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacements for ``vllm.v1.worker.gpu.model_states.mamba_hybrid``.

``MambaHybridModelState.postprocess_state`` launches the Triton
``_scatter_num_accepted_kernel`` / ``_fill_num_accepted_kernel`` upstream.
Kunlun XPU cannot JIT-compile Triton kernels:

  * the scatter path (``num_sampled`` is a tensor) maps to
    ``xspeedgate_ops.scatter_num_accepted_kernel``;
  * the fill path (``num_sampled`` is an int) maps to
    ``xspeedgate_ops.fill_num_accepted``.

``preprocess_state`` and the align branch of ``postprocess_state``
    Prefix caching forces ``mamba_cache_mode == "align"``, and upstream runs that
    state migration as fused GPU kernels driven by raw state addresses
    (``MambaSpecDecodeGPUContext`` -> ``preprocess_mamba_align_fused_kernel`` /
    ``precopy_mamba_align_fused_kernel`` / ``postprocess_mamba_fused_kernel``).
    Those kernels exist only to avoid a CPU-GPU sync around the V1 copy specs, so
    the same semantics are reimplemented natively here: the per-request decisions
    are tensor arithmetic on the state-slot arrays, and every (request, layer,
    state-type) block copy is issued as one ``xspeedgate_ops.batch_memcpy`` per
    mamba group, over int64 addresses built on device. Torch op *count* is what
    costs on this platform (~20us of dispatch each, regardless of shape), so the
    per-state layout is pre-combined into broadcastable columns.

    Nothing picks rows or states on the host: a boolean-mask select would need a
    ``nonzero`` and therefore a CPU-GPU sync every step, which ``--async-
    scheduling`` cannot afford. Rows that must not copy carry a zero byte count
    instead, which ``batch_memcpy`` skips.

    Speculative decoding is *not* covered: it is the only thing that needs
    ``token_bias > 0`` (and the DS conv layout). The bias arithmetic is carried
    here anyway so the copy matches the fused kernels, but since this path no
    longer goes through ``MambaSpecDecodeGPUContext`` (which used to raise for
    it), ``_build_align_groups`` refuses a spec-decode config itself.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging
from typing import NamedTuple

import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    is_conv_state_dim_first,
)
from vllm.v1.kv_cache_interface import MambaSpec

logger = logging.getLogger("vllm_kunlun")


class _AlignGroup(NamedTuple):
    """Per-(layer, state-type) copy metadata for one mamba KV-cache group.

    The same fields ``MambaSpecDecodeGPUContext.initialize_from_forward_context``
    flattens into device arrays, pre-combined so that a copy costs as few torch
    ops as possible: every tensor is int64 ``[num_states, 1]`` and broadcasts
    against the per-request row vectors.
    """

    states: tuple[torch.Tensor, ...]  # keeps the state tensors (and addresses) alive
    base_ptrs: torch.Tensor  # data_ptr of each state tensor
    block_strides: torch.Tensor  # bytes between consecutive state blocks
    size_base: torch.Tensor  # bytes copied at bias 0 (page strides can exceed it)
    bias_scale: torch.Tensor  # bytes the conv window shifts per bias token
    col_shift: torch.Tensor  # 1 where the bias selects the src block column (temporal)


def _scan_mamba_groups(kv_cache_config, cache_config=None):
    """Collect the MambaSpec groups, without assuming they are all identical.

    `MambaHybridModelState._get_mamba_group_info` asserts every mamba group
    shares one MambaSpec. That assumption has never been exercised on this
    platform -- it is only reached in align mode -- so rather than assert, log
    which fields differ and only insist on the one field the align arithmetic
    depends on (``block_size``, which a single state_idx array is shared for).
    """
    group_ids, specs = [], []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        if isinstance(group.kv_cache_spec, MambaSpec):
            group_ids.append(group_id)
            specs.append(group.kv_cache_spec)
    assert specs, "no mamba layers in the model"
    if any(spec != specs[0] for spec in specs[1:]):
        fields = [
            name
            for name in (
                "block_size",
                "page_size_bytes",
                "page_size_padded",
                "shapes",
                "dtypes",
                "mamba_type",
                "mamba_cache_mode",
                "num_speculative_blocks",
            )
            if len({repr(getattr(spec, name, None)) for spec in specs}) > 1
        ]
        logger.warning(
            "[KunlunPlugin] mamba KV-cache groups %s differ in %s; block sizes %s",
            group_ids,
            fields,
            [spec.block_size for spec in specs],
        )
    block_sizes = {spec.block_size for spec in specs}
    if len(block_sizes) > 1:
        raise NotImplementedError(
            f"align mode needs a single mamba block size, got {sorted(block_sizes)}"
        )
    return group_ids, specs[0]


def _build_align_groups(self, kv_cache_config):
    """Flatten the mamba state layout into the device arrays `_align_copy` needs."""
    if self.vllm_config.num_speculative_tokens > 0:
        # The copy below carries the ``token_bias`` arithmetic the fused kernels
        # use for accepted draft tokens, but the rest of hybrid spec decode is
        # unsupported here (see ``mamba_utils.MambaSpecDecodeGPUContext``, which
        # upstream would have instantiated on this path). Fail loudly rather
        # than run an untested combination.
        raise NotImplementedError(
            "speculative decoding on a hybrid Mamba model is not implemented "
            "on Kunlun XPU"
        )
    forward_context = self.vllm_config.compilation_config.static_forward_context
    copy_funcs = self.model.get_mamba_state_copy_func()
    mamba_group_ids, mamba_spec = _scan_mamba_groups(kv_cache_config, self.cache_config)

    def column(values):
        """A [num_states, 1] device column, to broadcast against the row vectors."""
        return torch.tensor(values, dtype=torch.int64, device=self.device).unsqueeze(1)

    groups = []
    for mamba_group_id in mamba_group_ids:
        states: list[torch.Tensor] = []
        base_ptrs: list[int] = []
        block_strides: list[int] = []
        size_base: list[int] = []
        bias_scale: list[int] = []
        col_shift: list[int] = []
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            kv_caches: list[torch.Tensor] = forward_context[layer_name].kv_cache
            for state_type_idx, state in enumerate(kv_caches):
                is_conv = copy_funcs[state_type_idx] is get_conv_copy_spec
                if is_conv:
                    if is_conv_state_dim_first():
                        # A DS conv state needs one copy per dim row to shift
                        # the window; K3 uses the SD layout
                        # (VLLM_SSM_CONV_STATE_LAYOUT unset).
                        raise NotImplementedError(
                            "align mode with the DS conv state layout is not "
                            "implemented on Kunlun XPU"
                        )
                    if state.dim() != 3:
                        raise ValueError(
                            f"expected 3D conv state cache, got {tuple(state.shape)}"
                        )
                elem_size = state.element_size()
                block_elems = state.stride(0) if state.dim() > 1 else state.numel()
                # Conv: the window is `conv_width` positions of `stride(1)`
                # elements, and a bias of n tokens drops n of them. Temporal:
                # the natural (unpadded) block, selected by block column
                # instead of by byte offset.
                inner_bytes = state.stride(1) * elem_size if is_conv else 0
                natural_bytes = (state[0].numel() if state.dim() > 1 else 1) * elem_size
                states.append(state)
                base_ptrs.append(state.data_ptr())
                block_strides.append(block_elems * elem_size)
                size_base.append(
                    state.size(1) * inner_bytes if is_conv else natural_bytes
                )
                bias_scale.append(inner_bytes)
                col_shift.append(0 if is_conv else 1)

        groups.append(
            _AlignGroup(
                states=tuple(states),
                base_ptrs=column(base_ptrs),
                block_strides=column(block_strides),
                size_base=column(size_base),
                bias_scale=column(bias_scale),
                col_shift=column(col_shift),
            )
        )
    return groups, mamba_spec, mamba_group_ids


def _scatter_ones(self, idx_mapping, selected, num_reqs) -> None:
    """Reset ``num_accepted_tokens`` to the neutral 1 on the selected rows.

    A plain ``arr[req_idx] = ...`` cannot express "skip this row": the rows it
    must skip would still have to name some slot, and under PP (where
    idx_mapping carries -1 sentinels) two rows could then name the same slot and
    race. The scatter op already skips negative entries, so hide the unselected
    rows behind the same sentinel.
    """
    torch.ops.xspeedgate_ops.scatter_num_accepted_kernel(
        torch.where(selected, idx_mapping, self._align_skip[:num_reqs]),
        self._align_ones[:num_reqs],
        self.num_accepted_tokens_gpu,
    )


def _align_copy(self, rows, src_cols, dst_cols, biases, should_copy) -> None:
    """Copy one state block per (row, layer, state-type), one memcpy per group.

    The column/bias inputs are device tensors with one entry per batch row.
    `rows` are batch rows (the block tables are batch-ordered) and
    `src_cols`/`dst_cols` are block columns, so `block_table[row, col]` matches
    the fused kernels' addressing. Conv states shift their window by `biases`
    tokens; temporal states use `biases` to select the accepted speculative
    column. Everything stays in torch broadcasts over int64: selecting the rows
    that copy would need a `nonzero`, i.e. a CPU-GPU sync on every step.
    """
    # Rows that must not copy still address the block table, so keep them in
    # range: a bias is only bounded on rows that do copy (postprocess leaves a
    # negative one behind otherwise), while a column is at worst the -1 of a
    # fresh request, which indexes the last column. Those rows then copy zero
    # bytes, which is what ``batch_memcpy`` skips on.
    biases = biases * should_copy

    for group, block_table in zip(self._align_groups, self._mamba_block_tables):
        shift_bytes = biases * group.bias_scale
        src_ids = block_table[rows, src_cols + biases * group.col_shift]
        dst_ids = block_table[rows, dst_cols]
        src_ptrs = src_ids * group.block_strides + (group.base_ptrs + shift_bytes)
        dst_ptrs = dst_ids * group.block_strides + group.base_ptrs
        sizes = (group.size_base - shift_bytes) * should_copy
        torch.ops.xspeedgate_ops.batch_memcpy(
            src_ptrs.reshape(-1), dst_ptrs.reshape(-1), sizes.reshape(-1)
        )


def preprocess_state(
    self,
    input_batch,
    block_tables: tuple[torch.Tensor, ...],
    kv_cache_config,
    num_computed_tokens: torch.Tensor,
) -> None:
    """Native align-mode state migration across block boundaries (V1 semantics).

    Stands in for `preprocess_mamba_align_fused_kernel` +
    `precopy_mamba_align_fused_kernel`: snapshot the pre-advance running block and
    accepted-token bias, advance the running block to the one this step writes
    into, reset the accepted count when a boundary is crossed, then copy each
    request's state into its new window block.
    """
    if not self._align_mode:
        return
    num_reqs = input_batch.num_reqs
    if num_reqs == 0:
        return

    if getattr(self, "_align_groups", None) is None:
        groups, spec, group_ids = _build_align_groups(self, kv_cache_config)
        self._align_groups = groups
        self._align_spec = spec
        self._align_group_ids = group_ids
        self._align_rows = torch.arange(
            self.max_num_reqs, dtype=torch.int64, device=self.device
        )
        self._align_ones = torch.ones(
            self.max_num_reqs,
            dtype=self.num_accepted_tokens_gpu.dtype,
            device=self.device,
        )
        self._align_skip = self._align_ones.new_full((self.max_num_reqs,), -1)
    # `block_tables` are this step's batch-order slices of the persistent
    # input_block_tables, so they are re-read every step: their row count is
    # num_reqs_after_padding and shrinks/grows with the batch. postprocess_state
    # runs on the same batch and needs no more rows than this.
    self._mamba_block_tables = [block_tables[gid] for gid in self._align_group_ids]

    block_size = self._align_spec.block_size
    idx_mapping = input_batch.idx_mapping[:num_reqs]
    # idx_mapping is built from req_id_to_index here (unlike the one
    # postprocess_state gets under PP), so its entries are distinct and hold no
    # -1 sentinels; guard the writes anyway, since a -1 would otherwise land on
    # a real state slot.
    valid = idx_mapping >= 0
    req_idx = idx_mapping.clamp(min=0)

    # index_select over a gather (`arr[req_idx]`) is worth ~60us per call here.
    state_idx = self._mamba_state_idx_gpu.index_select(0, req_idx)
    num_accepted = self.num_accepted_tokens_gpu.index_select(0, req_idx)
    # num_accepted is >= 1 by construction; clamp anyway, a negative bias
    # would make the copy read before the block (upstream clamps too).
    src_off = (num_accepted - 1).clamp(min=0)

    query_start_loc = input_batch.query_start_loc
    computed_after = num_computed_tokens.index_select(0, req_idx) + (
        query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
    )
    # ceil(computed_after / block_size) - 1, in one division: a scheduled
    # request always has computed_after >= 1.
    new_state_idx = (computed_after - 1) // block_size
    self._mamba_state_idx_gpu[req_idx] = torch.where(valid, new_state_idx, state_idx)

    # A request whose running block moved re-reads its state from the new block
    # start, so its accepted count goes back to the neutral 1.
    crossed = valid & (state_idx >= 0) & (state_idx != new_state_idx)
    _scatter_ones(self, idx_mapping, crossed, num_reqs)

    _align_copy(
        self, self._align_rows[:num_reqs], state_idx, new_state_idx, src_off, crossed
    )


def _align_postprocess(
    self, idx_mapping: torch.Tensor, num_computed_tokens: torch.Tensor
) -> None:
    """Native stand-in for `postprocess_mamba_fused_kernel` (align mode).

    When this step's accepted tokens leave the sequence exactly on a block
    boundary, the running state is copied into that full block so a later
    prefix-cache hit can resume from it. `num_computed_tokens` is slot-indexed
    and already holds the post-step count.
    """
    num_reqs = idx_mapping.shape[0]
    block_size = self._align_spec.block_size
    # Under PP, idx_mapping marks filtered rows with -1. Those read slot 0
    # instead; every write below is masked by `valid` and writes the value it
    # just read back, so the slot is left untouched.
    valid = idx_mapping >= 0
    req_idx = idx_mapping.clamp(min=0)

    num_accepted = self.num_accepted_tokens_gpu.index_select(0, req_idx)
    src_col = self._mamba_state_idx_gpu.index_select(0, req_idx)
    new_computed = num_computed_tokens.index_select(0, req_idx)
    running = new_computed - num_accepted + 1
    last_full_block = new_computed // block_size
    aligned = last_full_block * block_size
    needs_copy = valid & (aligned >= running)
    dst_col = last_full_block - 1
    bias = aligned - running

    # A copy that stays inside the running block changes no data, but it does
    # reset the accepted count to the neutral value.
    in_place = src_col == dst_col
    reset = needs_copy & in_place
    _scatter_ones(self, idx_mapping, reset, num_reqs)

    copy = needs_copy & ~(in_place & (bias == 0))
    _align_copy(self, self._align_rows[:num_reqs], src_col, dst_col, bias, copy)


def postprocess_state(
    self,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor | int,
    num_computed_tokens: torch.Tensor | None = None,
) -> None:
    # Chunked prefill does not sample a token, so num_sampled can be 0.
    # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
    num_reqs = idx_mapping.shape[0]
    if not num_reqs:
        return

    if not isinstance(num_sampled, int):
        # idx_mapping may contain -1 sentinels (filtered rows) under PP; the
        # op skips them rather than scattering with a host-side gather.
        torch.ops.xspeedgate_ops.scatter_num_accepted_kernel(
            idx_mapping, num_sampled, self.num_accepted_tokens_gpu
        )
    else:
        # Fill with single value. The op does not clamp, so clamp here exactly
        # like the Triton kernel's caller upstream.
        torch.ops.xspeedgate_ops.fill_num_accepted(
            idx_mapping, max(num_sampled, 1), self.num_accepted_tokens_gpu
        )

    # Align: save the running state to the block-aligned position when
    # spec-decode acceptance leaves the sequence non-block-aligned (mirrors
    # the V1 align postprocess). num_computed_tokens already holds the
    # post-step advanced count. `_align_groups` is created by
    # `preprocess_state`, which always runs earlier in the same step.
    if (
        self._align_mode
        and num_computed_tokens is not None
        and getattr(self, "_align_groups", None) is not None
    ):
        _align_postprocess(self, idx_mapping, num_computed_tokens)


preprocess_state._kunlun_patched = True
postprocess_state._kunlun_patched = True
