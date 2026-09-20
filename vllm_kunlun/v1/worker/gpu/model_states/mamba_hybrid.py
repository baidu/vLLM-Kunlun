# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun overrides for ``vllm.v1.worker.gpu.model_states.mamba_hybrid``.

Upstream has three Triton launch sites in this module:

* ``:195`` ``preprocess_mamba_align_fused_kernel``        -- align mode only
* ``:308`` ``_scatter_num_accepted_kernel``               -- every step (live)
* ``MambaSpecDecodeGPUContext.run_fused_precopy`` / ``run_fused_postprocess_align``
  (``:207`` / ``:328``)                                   -- align mode only

Align mode requires prefix caching (``models/config.py:596-600`` forces
``mamba_cache_mode="none"`` otherwise, which leaves ``_align_mode`` False at
``mamba_hybrid.py:86``), so only the scatter kernel is live in a
prefix-caching-off configuration. It is replaced with the Kunlun native op
``torch.ops.xspeedgate_ops.scatter_num_accepted_kernel``, which implements the
same ``-1``-sentinel skip on-device.

We patch the module-level *kernel object* rather than the method that launches
it: ``MambaHybridModelState.postprocess_state`` is defined in the upstream
module, so its ``__globals__`` *is* the upstream module dict, and rebinding the
name there is picked up at call time. That leaves the surrounding logic --
including the ``int`` branch at ``:311-315`` (plain ``index_fill_``, no Triton)
and the align branch -- completely untouched.

The native op requires int32 index/count tensors. On the live path all three
are already int32 (``idx_mapping`` from ``np.int32``, ``num_sampled`` from
``seq_lens.new_ones`` with ``seq_lens`` int32, ``num_accepted_tokens_gpu``
allocated int32 at ``mamba_hybrid.py:79-80``); the two read-only inputs are cast
defensively, while ``num_accepted`` is mutated in place and left as the caller's
tensor. The torch-native ``_kernels.scatter_num_accepted`` is no longer on the
live path -- it remains the CPU parity oracle in ``tests/ut/test_mrv2_kernels.py``.
"""

import logging

import torch
import vllm.v1.worker.gpu.model_states.mamba_hybrid as _up

from vllm_kunlun.v1.worker.gpu._kernels import TorchKernel

logger = logging.getLogger("vllm_kunlun")


def _scatter_num_accepted(idx_mapping, num_sampled, num_accepted) -> None:
    idx32 = (
        idx_mapping if idx_mapping.dtype == torch.int32 else idx_mapping.to(torch.int32)
    )
    ns32 = (
        num_sampled if num_sampled.dtype == torch.int32 else num_sampled.to(torch.int32)
    )
    torch.ops.xspeedgate_ops.scatter_num_accepted_kernel(idx32, ns32, num_accepted)


_up._scatter_num_accepted_kernel = TorchKernel(_scatter_num_accepted)

# ``MambaHybridAttnMetadata.get_extra_attn_kwargs`` (:47-64) decides whether to
# forward ``num_accepted_tokens`` / ``num_decode_draft_tokens_cpu`` by doing an
# isinstance check against the GDN builder class resolved from that module's
# globals (bound at :17 from vllm.v1.attention.backends.gdn_attn).
#
# Kunlun's builder is NOT a subclass of upstream's -- it works today only
# because vllm_kunlun/v1/attention/backends/gdn_attn.py:510-512 monkey-patches
# the upstream module and happens to be imported first (via
# vllm_kunlun/models/qwen3_next.py). If that ordering ever inverts the isinstance
# check silently fails and the spec-decode kwargs are dropped -- wrong results,
# no error. Bind it explicitly so the check is order-independent.
try:
    from vllm_kunlun.v1.attention.backends.gdn_attn import (
        GDNAttentionMetadataBuilder as _KunlunGDNBuilder,
    )

    _up.GDNAttentionMetadataBuilder = _KunlunGDNBuilder
except Exception:
    logger.warning(
        "[KunlunPlugin] could not bind the Kunlun GDN metadata builder into "
        "mamba_hybrid; spec-decode extra attn kwargs may be dropped",
        exc_info=True,
    )

logger.info(
    "[KunlunPlugin] V2 MambaHybridModelState patched "
    "(xspeedgate_ops native scatter_num_accepted)"
)
