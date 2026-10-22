# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacements for ``vllm.v1.worker.gpu.buffer_utils``.

``UvaBufferPool.copy_to_uva``
    Upstream gets the device-side publish for free: ``buf.uva`` is a UVA alias
    onto ``buf.cpu``. On this platform ``get_accelerator_view_from_cpu_tensor``
    returns a one-shot H2D copy instead (see
    ``vllm_kunlun/utils/torch_utils.py``), so the host write would never reach
    the device. Publish with an explicit blocking ``copy_``.

``StagedWriteTensor.apply_write`` / ``FusedStagedWriter.apply``
    Upstream launches the Triton ``_apply_write_kernel`` in both the single-group
    and the fused multi-group form; Kunlun XPU cannot JIT-compile Triton, so each
    is replaced by the matching ``xspeedgate_ops.apply_write`` mode
    (``MULTI_GROUP=False`` / ``True``). The staged-write bookkeeping and the
    published UVA metadata are left exactly as upstream -- only the kernel launch
    is swapped, so the two bodies stay diff-able against upstream.

The module-level names therefore mirror the replaced attributes (``apply_write``,
``apply``), as ``_mrv2_apply`` binds plugin functions by attribute name.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)
from vllm.utils.torch_utils import async_tensor_h2d

logger = logging.getLogger("vllm_kunlun")

# Matches the ``BLOCK_SIZE`` upstream passes to ``_apply_write_kernel``.
_BLOCK_SIZE = 1024


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


def apply_write(self) -> None:
    n = len(self._staged_write_indices)
    if n == 0:
        return

    indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
    starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
    cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)

    # Special handling for write_contents
    write_contents = async_tensor_h2d(
        self._staged_write_contents, device=self.device, dtype=self.dtype
    )

    # Write diffs to the GPU buffer. The op requires a 2-D output, but a 1-D
    # ``StagedWriteTensor`` (``states.total_len`` / ``num_computed_tokens``) is
    # addressed flat -- ``index * stride(0) + start`` with ``stride(0) == 1`` --
    # which is exactly what an ``(n, 1)`` view expresses (the kernel does raw
    # pointer arithmetic and never clamps to a row width). Assert ``stride(0)``
    # first so ``view`` can never silently return a *copy*, which would drop the
    # writes on the floor.
    output = self.gpu
    if output.dim() == 1:
        assert output.stride(0) == 1, "1-D staged-write output must be contiguous"
        output = output.view(-1, 1)
    torch.ops.xspeedgate_ops.apply_write(
        output,
        indices_uva,
        starts_uva,
        write_contents,
        cu_lens_uva,
        None,
        None,
        None,
        _BLOCK_SIZE,
        False,
    )
    # Clear the staged writes
    self.clear_staged_writes()


def apply(
    self,
    tensors: Sequence,
    output_ptrs: torch.Tensor,
    output_strides: torch.Tensor,
) -> None:
    """Apply and clear the staged writes of `tensors` with one kernel.

    Body identical to ``FusedStagedWriter.apply`` upstream, except the launch:
    the single Triton kernel becomes the ``MULTI_GROUP=True`` mode of
    ``xspeedgate_ops.apply_write``, which resolves each write's target tensor
    from ``output_ptrs``/``output_strides``/``write_group_ids``.
    """
    group_ids: list[int] = []
    indices: list[int] = []
    starts: list[int] = []
    contents: list[int | float] = []
    cu_lens: list[int] = []

    for group_id, t in enumerate(tensors):
        n = len(t._staged_write_indices)
        if n == 0:
            continue

        group_ids.extend([group_id] * n)
        indices.extend(t._staged_write_indices)
        starts.extend(t._staged_write_starts)
        content_base = len(contents)
        contents.extend(t._staged_write_contents)
        cu_lens.extend(content_base + cu_len for cu_len in t._staged_write_cu_lens)

    if not group_ids:
        return

    group_ids_uva = self.group_ids.copy_to_uva(group_ids)
    indices_uva = self.indices.copy_to_uva(indices)
    starts_uva = self.starts.copy_to_uva(starts)
    cu_lens_uva = self.cu_lens.copy_to_uva(cu_lens)
    contents_gpu = async_tensor_h2d(contents, device=self.device, dtype=torch.int32)

    torch.ops.xspeedgate_ops.apply_write(
        None,
        indices_uva,
        starts_uva,
        contents_gpu,
        cu_lens_uva,
        output_ptrs,
        output_strides,
        group_ids_uva,
        _BLOCK_SIZE,
        True,
    )
    for t in tensors:
        t.clear_staged_writes()


copy_to_uva._kunlun_patched = True
apply_write._kunlun_patched = True
apply._kunlun_patched = True
