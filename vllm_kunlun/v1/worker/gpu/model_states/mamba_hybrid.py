# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P800 MRV2 Mamba state operations.

Acceptance fill/scatter use xspeedgate_ops. Align-mode preprocessing, precopy
and postprocessing all use the native fused ops. SD precopy uses a flat
one-row DS byte-copy view because the installed native SD branch does not
implement window shifts. The physical state layout stays SD. Hybrid speculative
decoding and physical DS-conv serving remain unsupported here.
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
    """Retain state tensors referenced by native copy metadata."""

    states: tuple[torch.Tensor, ...]


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
    """Collect state tensors and validate supported layouts for native copies."""
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

    groups = []
    for mamba_group_id in mamba_group_ids:
        states: list[torch.Tensor] = []
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
                states.append(state)
        groups.append(_AlignGroup(states=tuple(states)))
    return groups, mamba_spec, mamba_group_ids


def preprocess_state(
    self,
    input_batch,
    block_tables: tuple[torch.Tensor, ...],
    kv_cache_config,
    num_computed_tokens: torch.Tensor,
) -> None:
    """Native align-mode state migration across block boundaries (V1 semantics).

    `preprocess_mamba_align_fused_kernel` snapshots the pre-advance running block
    and accepted-token bias, advances the running block to the one this step
    writes into, and resets the accepted count when a boundary is crossed;
    `precopy_mamba_align_fused_kernel` then copies each request's temporal state
    into its new window block; SD conv uses the same op with flat-row metadata.
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
        self._align_skip = torch.full(
            (self.max_num_reqs,),
            -1,
            dtype=self.num_accepted_tokens_gpu.dtype,
            device=self.device,
        )
    # `block_tables` are this step's batch-order slices of the persistent
    # input_block_tables, so they are re-read every step: their row count is
    # num_reqs_after_padding and shrinks/grows with the batch. postprocess_state
    # runs on the same batch and needs no more rows than this.
    self._mamba_block_tables = [block_tables[gid] for gid in self._align_group_ids]

    block_size = self._align_spec.block_size
    idx_mapping = input_batch.idx_mapping[:num_reqs]
    # idx_mapping is built from req_id_to_index here (unlike the one
    # postprocess_state gets under PP), so its entries are distinct and hold no
    # -1 sentinels; the op skips negative entries anyway.
    req_idx = idx_mapping.clamp(min=0)

    # Fused native decision: writes the pre-advance src column and accepted-token
    # bias into the src_col/src_off buffers, advances _mamba_state_idx_gpu to the
    # new running block, and resets num_accepted on a boundary crossing. Every
    # per-request index stays on device, so there is no CPU-GPU sync per step.
    torch.ops.xspeedgate_ops.preprocess_mamba_align_fused_kernel(
        idx_mapping.to(torch.int32).contiguous(),
        self._mamba_state_idx_gpu,
        num_computed_tokens.to(torch.int32).contiguous(),
        input_batch.query_start_loc.to(torch.int32).contiguous(),
        self.num_accepted_tokens_gpu,
        self._mamba_src_col_gpu,
        self._mamba_src_off_gpu,
        num_reqs,
        block_size,
    )

    # PP sampling can return an older batch after a newer prefill was prepared.
    # Keep tables in request-slot order, then gather the matching postprocess
    # batch rather than reusing the most recently prepared batch rows.
    if getattr(self, "_mamba_slot_tables", None) is None:
        self._mamba_slot_tables = [
            torch.zeros(
                (self.max_num_reqs, t.shape[1]), dtype=t.dtype, device=self.device
            )
            for t in self._mamba_block_tables
        ]
        self._mamba_post_tables = [torch.zeros_like(t) for t in self._mamba_slot_tables]
        self._mamba_post_table_ptrs = torch.tensor(
            [t.data_ptr() for t in self._mamba_post_tables],
            dtype=torch.int64,
            device=self.device,
        )
    for slot_table, batch_table in zip(
        self._mamba_slot_tables, self._mamba_block_tables
    ):
        # preprocess receives distinct live request slots; PP -1 filtering only
        # happens in the later postprocess batch.
        slot_table.index_copy_(0, req_idx.to(torch.int64), batch_table[:num_reqs])

    # Temporal states use their natural layout. SD conv uses a one-row DS view
    # of scalar elements, with its token bias scaled by the contiguous row size.
    layout = _native_align_layout(self)
    if layout["temporal"]:
        if not getattr(self, "_native_precopy_logged", False):
            logger.info(
                "[KunlunPlugin] Mamba align native precopy active: "
                "xspeedgate_ops.precopy_mamba_align_fused_kernel; "
                "temporal states=%d; SD conv uses native flat-row precopy",
                layout["temporal"][0].numel(),
            )
            self._native_precopy_logged = True
        torch.ops.xspeedgate_ops.precopy_mamba_align_fused_kernel(
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            layout["table_ptrs"],
            layout["stride"],
            *layout["temporal"],
            idx_mapping.to(torch.int32).contiguous(),
            num_reqs,
            1024,
            False,
            1,
        )
    for inner_size, conv_metadata in layout["conv_flat"]:
        torch.ops.xspeedgate_ops.precopy_mamba_align_fused_kernel(
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu * inner_size,
            layout["table_ptrs"],
            layout["stride"],
            *conv_metadata,
            idx_mapping.to(torch.int32).contiguous(),
            num_reqs,
            1024,
            True,
            1,
        )
    if layout["conv_flat"] and not getattr(self, "_native_sd_precopy_logged", False):
        logger.info(
            "[KunlunPlugin] Mamba SD precopy active: "
            "xspeedgate_ops.precopy_mamba_align_fused_kernel; "
            "flat scalar-row view; groups=%d; no batch_memcpy",
            len(layout["conv_flat"]),
        )
        self._native_sd_precopy_logged = True


def _native_align_layout(self):
    """Build native byte-copy metadata; retain every referenced state tensor."""
    tables = self._mamba_block_tables
    strides = {table.stride(0) for table in tables}
    if len(strides) != 1 or any(
        table.dtype != torch.int32 or table.stride(1) != 1 for table in tables
    ):
        raise ValueError(
            "native Mamba copy requires int32 tables with a shared row stride"
        )
    signature = tuple((t.data_ptr(), t.stride(0)) for t in tables)
    cached = getattr(self, "_native_align_cached", None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    values = [[] for _ in range(8)]
    temporal = []
    conv_groups = {}
    copy_funcs = self.model.get_mamba_state_copy_func()
    for group_id, group in enumerate(self._align_groups):
        for i, state in enumerate(group.states):
            conv = copy_funcs[i % len(copy_funcs)] is get_conv_copy_spec
            row = [
                state.data_ptr(),
                state.stride(0) * state.element_size(),
                state.element_size(),
                state.stride(1) if conv else state[0].numel(),
                state.size(1) if conv else 0,
                group_id,
                0,
                0,
            ]
            if conv:
                if (
                    state.dim() != 3
                    or state.stride(2) != 1
                    or state.stride(1) != state.size(2)
                ):
                    raise ValueError("native SD precopy requires contiguous token rows")
                conv_groups.setdefault(state.size(2), []).append(len(values[0]))
            else:
                temporal.append(len(values[0]))
            for target, value in zip(values, row):
                target.append(value)
    if len(values[0]) > 512 or len(tables) > 16:
        raise ValueError("native Mamba metadata exceeds the installed op limits")
    dtypes = (
        torch.int64,
        torch.int64,
        torch.int32,
        torch.int64,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int64,
    )
    metadata = [
        torch.tensor(v, dtype=dt, device=self.device) for v, dt in zip(values, dtypes)
    ]
    temporal_metadata = [m[temporal].contiguous() for m in metadata] if temporal else []
    if temporal_metadata:
        temporal_metadata[2] = temporal_metadata[2].to(torch.int64)
    # Keep the public postprocess metadata unchanged. Only precopy sees the
    # synthetic DS layout: one scalar row, width W*D, bias b*D. Thus its source
    # byte offset is b*D*elem_size and copy size is (W-b)*D*elem_size, exactly SD.
    precopy_dtypes = list(dtypes)
    precopy_dtypes[2] = torch.int64
    conv_flat = []
    for inner_size, indices in conv_groups.items():
        flat_values = [[column[i] for i in indices] for column in values]
        flat_values[3] = [1] * len(indices)
        flat_values[4] = [values[4][i] * inner_size for i in indices]
        flat_values[6] = [1] * len(indices)
        flat_values[7] = [values[4][i] * inner_size * values[2][i] for i in indices]
        conv_flat.append(
            (
                inner_size,
                [
                    torch.tensor(v, dtype=dt, device=self.device)
                    for v, dt in zip(flat_values, precopy_dtypes)
                ],
            )
        )
    layout = dict(
        table_ptrs=torch.tensor(
            [t.data_ptr() for t in tables], dtype=torch.int64, device=self.device
        ),
        stride=next(iter(strides)),
        all=metadata,
        temporal=temporal_metadata,
        conv_flat=conv_flat,
    )
    self._native_align_cached = (signature, layout)
    return layout


def _align_postprocess(self, idx_mapping, num_computed_tokens):
    """Gather slot data into dense batch rows before calling the native op.

    Native per-request metadata is bounded by num_reqs. Real request slots can
    exceed that bound (e.g. a one-request batch in slot 25), so passing the full
    slot buffers and idx_mapping directly is unsafe. Preserve PP skip rows with
    -1 in a dense identity mapping, then scatter the output counts back.
    """
    layout = _native_align_layout(self)
    num_reqs = idx_mapping.shape[0]
    req_idx = idx_mapping.clamp(min=0).to(torch.int64)
    accepted = self.num_accepted_tokens_gpu.index_select(0, req_idx)
    accepted_out = accepted.clone()
    state_idx = self._mamba_state_idx_gpu.index_select(0, req_idx)
    computed = num_computed_tokens.index_select(0, req_idx)
    native_mapping = torch.where(
        idx_mapping >= 0,
        self._align_rows[:num_reqs].to(torch.int32),
        self._align_skip[:num_reqs],
    )
    for slot_table, post_table in zip(self._mamba_slot_tables, self._mamba_post_tables):
        post_table[:num_reqs].copy_(slot_table.index_select(0, req_idx))
    if not getattr(self, "_native_postprocess_logged", False):
        logger.info(
            "[KunlunPlugin] Mamba align native postprocess active: "
            "xspeedgate_ops.postprocess_mamba; states=%d; dense batch mapping",
            layout["all"][0].numel(),
        )
        self._native_postprocess_logged = True
    torch.ops.xspeedgate_ops.postprocess_mamba(
        accepted,
        state_idx,
        computed,
        computed,
        computed,
        self._mamba_post_table_ptrs,
        self._mamba_post_tables[0].stride(0),
        *layout["all"],
        accepted_out,
        native_mapping,
        num_reqs,
        self._align_spec.block_size,
        True,
        False,
    )
    torch.ops.xspeedgate_ops.scatter_num_accepted_kernel(
        idx_mapping,
        accepted_out,
        self.num_accepted_tokens_gpu,
    )


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
