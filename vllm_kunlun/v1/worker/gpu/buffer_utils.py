# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacements for ``vllm.v1.worker.gpu.buffer_utils``.

Two things are replaced, both because Kunlun XPU has no UVA aliasing:

``UvaBufferPool.copy_to_uva``
    Upstream gets the device-side publish for free: ``buf.uva`` is a UVA alias
    onto ``buf.cpu``. On this platform ``get_accelerator_view_from_cpu_tensor``
    returns a one-shot H2D copy instead (see
    ``vllm_kunlun/utils/torch_utils.py``), so the host write would never reach
    the device. Publish with an explicit blocking ``copy_``.

``StagedWriteTensor.apply_write``
    Upstream launches the Triton ``_apply_write_kernel``; kept on the
    **torch-native** path (``_apply_write``) rather than the
    ``xspeedgate_ops.apply_write`` op. When two staged writes overlap the same
    element the sequential implementation is deterministic ("last write wins"),
    while the op -- like the Triton kernel it replaces -- leaves the order
    unspecified. The runner's staged writes are disjoint by construction, so
    production never observes the difference.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

from vllm.utils.torch_utils import async_tensor_h2d

logger = logging.getLogger("vllm_kunlun")


def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
    # Round robin to the next buffer.
    self._curr = (self._curr + 1) % self.max_concurrency
    buf = self._uva_bufs[self._curr]
    # CPU-to-CPU copy
    dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
    n = len(x)
    dst[:n] = x
    # Publish the host-side write to the device buffer. Upstream gets this for
    # free (``buf.uva`` aliases ``buf.cpu``); here ``buf.uva`` is a one-shot H2D
    # copy taken at construction, so copy explicitly -- and blocking, so a later
    # step cannot overwrite the pinned source while the copy is still in flight.
    buf.uva[:n].copy_(buf.cpu[:n])
    return buf.uva[:n]


copy_to_uva._kunlun_patched = True


def _apply_write(
    outputs: Sequence[tuple[torch.Tensor, int]],
    write_indices: Sequence[int],
    write_starts: Sequence[int],
    write_contents: torch.Tensor,
    write_cu_lens: Sequence[int],
    write_group_ids: Sequence[int] | None,
) -> None:
    """Pure-torch replacement for the Triton `_apply_write_kernel`.

    For each staged write `pid`, copies `write_contents[cu_start:cu_end]` into the
    target output row. When `write_group_ids` is None all writes target the single
    output in `outputs`; otherwise `write_group_ids[pid]` selects the output tensor
    (KV cache group). Each output is a `(tensor, row_stride)` pair, matching the
    flat pointer arithmetic `row_idx * row_stride + start_idx` of the kernel.
    """
    for pid in range(len(write_indices)):
        cu_start = write_cu_lens[pid - 1] if pid > 0 else 0
        cu_end = write_cu_lens[pid]
        content_len = cu_end - cu_start
        if content_len == 0:
            continue

        group_id = 0 if write_group_ids is None else write_group_ids[pid]
        out_tensor, row_stride = outputs[group_id]

        offset = write_indices[pid] * row_stride + write_starts[pid]
        out_tensor.view(-1)[offset : offset + content_len] = write_contents[
            cu_start:cu_end
        ]


def apply_write(self) -> None:
    n = len(self._staged_write_indices)
    if n == 0:
        return

    indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
    starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
    cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)
    write_contents = async_tensor_h2d(
        self._staged_write_contents, device=self.device, dtype=self.dtype
    )

    # Native path on purpose; see the module docstring. `indices_uva` /
    # `starts_uva` / `cu_lens_uva` are still published so the UVA buffers stay
    # in sync with the staged lists.
    _apply_write(
        [(self.gpu, self.gpu.stride(0))],
        self._staged_write_indices,
        self._staged_write_starts,
        write_contents,
        self._staged_write_cu_lens,
        None,
    )
    self.clear_staged_writes()


apply_write._kunlun_patched = True
