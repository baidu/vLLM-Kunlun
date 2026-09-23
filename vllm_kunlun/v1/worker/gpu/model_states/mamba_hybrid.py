# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacements for ``vllm.v1.worker.gpu.model_states.mamba_hybrid``.

``MambaHybridModelState.postprocess_state`` launches the Triton
``_scatter_num_accepted_kernel`` / ``_fill_num_accepted_kernel`` upstream.
Kunlun XPU cannot JIT-compile Triton kernels:

  * the scatter path (``num_sampled`` is a tensor) is replaced by its
    xspeedgate_ops equivalent;
  * the fill path (``num_sampled`` is an int) has no op and keeps the torch
    implementation, carried here as ``_fill_num_accepted_torch``.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)

logger = logging.getLogger("vllm_kunlun")


def _fill_num_accepted_torch(
    idx_mapping: torch.Tensor,
    num_accepted: torch.Tensor,
    num_sampled: int,
) -> None:
    """Pure-torch replacement for `_fill_num_accepted_kernel`.

    For each batch row: `num_accepted[idx_mapping[row]] = num_sampled`, skipping
    rows whose idx_mapping entry is a negative (-1) sentinel.
    """
    valid = idx_mapping >= 0
    if not bool(valid.any()):
        return
    idx = idx_mapping[valid].to(torch.int64)
    num_accepted.index_fill_(0, idx, num_sampled)


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
        # Fill with single value.
        _fill_num_accepted_torch(
            idx_mapping, self.num_accepted_tokens_gpu, max(num_sampled, 1)
        )

    # Align: save the running state to the block-aligned position when
    # spec-decode acceptance leaves the sequence non-block-aligned (mirrors
    # the V1 align postprocess). num_computed_tokens already holds the
    # post-step advanced count.
    if (
        self._align_mode
        and num_computed_tokens is not None
        and self._mamba_ctx is not None
    ):
        self._mamba_ctx.run_fused_postprocess_align(
            num_reqs,
            self.num_accepted_tokens_gpu,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            idx_mapping,
        )


postprocess_state._kunlun_patched = True
