# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun replacement for ``vllm.v1.worker.gpu.model_states.mamba_hybrid``.

Upstream has three Triton launch sites in this module:

* ``:195`` ``preprocess_mamba_align_fused_kernel``        -- align mode only
* ``:308`` ``_scatter_num_accepted_kernel``               -- every step (live)
* ``MambaSpecDecodeGPUContext.run_fused_precopy`` / ``run_fused_postprocess_align``
  (``:207`` / ``:328``)                                   -- align mode only

Align mode requires prefix caching (``models/config.py:596-600`` forces
``mamba_cache_mode="none"`` otherwise, which leaves ``_align_mode`` False at
``mamba_hybrid.py:86``), so only the scatter kernel is live in a
prefix-caching-off configuration. It is replaced with a torch-native scatter.

We patch the module-level *kernel object* rather than the method that launches
it: ``MambaHybridModelState.postprocess_state`` is defined in the upstream
module, so its ``__globals__`` *is* the upstream module dict, and rebinding the
name there is picked up at call time. That leaves the surrounding logic --
including the ``int`` branch at ``:311-315`` (plain ``index_fill_``, no Triton)
and the align branch -- completely untouched.
"""

import logging

from vllm_kunlun.v1.worker.gpu._kernels import TorchKernel, scatter_num_accepted
from vllm_kunlun.v1.worker.gpu._upstream import load_upstream, reexport

logger = logging.getLogger("vllm_kunlun")

_up = load_upstream("vllm.v1.worker.gpu.model_states.mamba_hybrid")
reexport(_up, globals())

if not getattr(_up.MambaHybridModelState, "_kunlun_v2_patched", False):
    _up._scatter_num_accepted_kernel = TorchKernel(scatter_num_accepted)

    # ``MambaHybridAttnMetadata.get_extra_attn_kwargs`` (:47-64) decides whether
    # to forward ``num_accepted_tokens`` / ``num_decode_draft_tokens_cpu`` by
    # doing an isinstance check against the GDN builder class resolved from this
    # module's globals (bound at :17 from vllm.v1.attention.backends.gdn_attn).
    #
    # Kunlun's builder is NOT a subclass of upstream's -- it works today only
    # because vllm_kunlun/v1/attention/backends/gdn_attn.py:510-512
    # monkey-patches the upstream module and happens to be imported first (via
    # vllm_kunlun/models/qwen3_next.py). If that ordering ever inverts the
    # isinstance check silently fails and the spec-decode kwargs are dropped --
    # wrong results, no error. Bind it explicitly so the check is
    # order-independent.
    try:
        from vllm_kunlun.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadataBuilder as _KunlunGDNBuilder,
        )

        _up.GDNAttentionMetadataBuilder = _KunlunGDNBuilder
        GDNAttentionMetadataBuilder = _KunlunGDNBuilder
    except Exception:
        logger.warning(
            "[KunlunPlugin] could not bind the Kunlun GDN metadata builder into "
            "mamba_hybrid; spec-decode extra attn kwargs may be dropped",
            exc_info=True,
        )

    _up.MambaHybridModelState._kunlun_v2_patched = True
    logger.info(
        "[KunlunPlugin] V2 MambaHybridModelState patched "
        "(torch-native scatter_num_accepted)"
    )

# Keep this module's name pointing at whatever is installed upstream, so a
# consumer importing it from here sees the patched object.
_scatter_num_accepted_kernel = _up._scatter_num_accepted_kernel
