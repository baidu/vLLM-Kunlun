# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacements for ``vllm.v1.worker.gpu.sample.gumbel``.

``apply_temperature`` and ``gumbel_sample`` launch the Triton
``_temperature_kernel`` / ``_gumbel_sample_kernel`` upstream. Kunlun XPU cannot
JIT-compile Triton kernels, so both are replaced by their xspeedgate_ops
equivalents.

Preconditions the op enforces (adapted here):
  * ``apply_temperature`` needs a **contiguous fp32** logits tensor and writes in
    place -- pass it straight through, since a defensive ``.contiguous()`` copy
    would silently drop the result. The sampler always hands it the contiguous
    fp32 copy made for the in-place sampling chain.
  * ``gumbel_sample`` needs contiguous fp32 logits (the fp16/bf16 widening is
    exact and this op only reads logits), int64 ``pos``, and an int64
    ``output_processed_logits_col`` (the dspark speculator passes int32 step
    columns). It returns ``[num_tokens, 1]``; flattened to keep the 1-D contract.

The op's RNG is the Philox4x32-10 stream the upstream kernel used, and it stores
the *temperature-applied* logits into ``output_processed_logits`` -- which is
exactly the contract on this vllm version.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import torch

import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)

logger = logging.getLogger("vllm_kunlun")


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    torch.ops.xspeedgate_ops.apply_temperature(
        logits, expanded_idx_mapping, temperature
    )


def gumbel_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    if logits.dtype != torch.float32:
        logits = logits.to(torch.float32)
    if not logits.is_contiguous():
        logits = logits.contiguous()
    if (
        output_processed_logits_col is not None
        and output_processed_logits_col.dtype != torch.int64
    ):
        output_processed_logits_col = output_processed_logits_col.to(torch.int64)
    sampled = torch.ops.xspeedgate_ops.gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos.to(torch.int64),
        apply_temperature,
        output_processed_logits,
        output_processed_logits_col,
        use_fp64,
    )
    return sampled.view(-1)


apply_temperature._kunlun_patched = True
gumbel_sample._kunlun_patched = True
