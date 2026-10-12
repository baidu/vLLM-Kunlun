# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op override for ``vllm.v1.worker.gpu.mm.rope``.

``RopeState.prepare_positions`` is on the live path for every mrope / XD-RoPE
model. Replace its Triton kernel with
``torch.ops.xspeedgate_ops.prepare_rope_positions``. The native op derives its
work extent from ``query_start_loc`` on-device and takes no ``max_model_len``
argument, avoiding the host ``.item()`` synchronization needed by the former
torch-native fallback. That fallback remains the CPU parity oracle in
``tests/ut/test_mrv2_kernels.py``.

The empty-batch guard and the live prefix of a scheduler-padded
``query_start_loc`` are handled here before the native call. These preserve
the upstream no-op behavior while satisfying the native wrapper's exact
``num_reqs + 1`` shape contract.

Kunlun also keeps prefill positions as typed tensors when admitting requests.
Upstream converts every axis to Python lists and reconstructs tensors later,
which is expensive for long multimodal prompts. The two staging overrides
preserve the model-produced positions and delta, batch the upload, and retain
writes made through the standard staged-buffer interface.

The remaining state behavior (``read_prefill_positions``,
``update_prefill_positions``, ``get_positions``, and ``get_rope_state``) is
reused from upstream.
"""

import logging

import numpy as np
import torch
import vllm.v1.worker.gpu.mm.rope as _up
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

logger = logging.getLogger("vllm_kunlun")


def _init_prefill_positions(
    self, req_idx, model, prefill_token_ids, mm_features
) -> None:
    if self.has_delta:
        positions, delta = model.get_mrope_input_positions(
            prefill_token_ids, mm_features
        )
        self.prefill_delta.np[req_idx] = delta
    else:
        positions = model.get_xdrope_input_positions(prefill_token_ids, mm_features)

    # Snapshot just as stage_write did, preserving non-contiguous positions
    # and all axes for multimodal/XD-RoPE inputs. Nothing is regenerated from
    # token indices, so this also covers nonzero deltas and pruned positions.
    positions = positions.to(device="cpu", dtype=torch.int32, copy=True).contiguous()
    writes = getattr(self, "_kunlun_prefill_writes", None)
    if writes is None:
        writes = self._kunlun_prefill_writes = []
    writes.append((req_idx, positions))


def _apply_staged_writes(self) -> None:
    writes = getattr(self, "_kunlun_prefill_writes", ())
    if writes:
        rows = []
        lengths = []
        for req_idx, positions in writes:
            rows.extend(range(self.num_dims * req_idx, self.num_dims * (req_idx + 1)))
            lengths.extend([positions.shape[1]] * self.num_dims)
        target = self.prefill_positions
        indices = target.write_indices.copy_to_uva(rows)
        starts = target.write_starts.copy_to_uva([0] * len(rows))
        cu_lens = target.write_cu_lens.copy_to_uva(np.cumsum(lengths, dtype=np.int32))
        # One typed upload and one write kernel for all newly admitted requests.
        # Avoid both scalar-list conversion and one H2D call per request/axis.
        cpu = torch.cat([positions.reshape(-1) for _, positions in writes])
        contents = async_copy_to_gpu(cpu, device=self.device)
        torch.ops.xspeedgate_ops.apply_write(
            target.gpu, indices, starts, contents, cu_lens
        )
        writes.clear()
    # Preserve any writes staged through the existing buffer interface.
    self.prefill_positions.apply_write()
    if self.has_delta:
        self.prefill_delta.copy_to_uva()


def _prepare_positions(
    self, idx_mapping, query_start_loc, prefill_lens, num_computed_tokens
) -> None:
    # ``prefill_positions`` is a StagedWriteTensor and ``prefill_delta`` a
    # UvaBackedTensor from gpu.buffer_utils, whose Kunlun patch may swap the
    # UVA views for plain device tensors -- ``.gpu`` is valid either way.
    if idx_mapping.numel() == 0:
        return
    torch.ops.xspeedgate_ops.prepare_rope_positions(
        self.positions,
        self.prefill_positions.gpu,
        self.prefill_delta.gpu,
        idx_mapping,
        query_start_loc[: idx_mapping.numel() + 1],
        prefill_lens,
        num_computed_tokens,
        self.num_dims,
    )


_up.RopeState.init_prefill_positions = _init_prefill_positions
_up.RopeState.apply_staged_writes = _apply_staged_writes
_up.RopeState.prepare_positions = _prepare_positions
logger.info(
    "[KunlunPlugin] V2 RopeState.prepare_positions patched (xspeedgate_ops native)"
)
