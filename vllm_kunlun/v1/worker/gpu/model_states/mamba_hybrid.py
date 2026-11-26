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
prefix-caching-off configuration. It is replaced with an XSpeedGate scatter.

We patch the module-level *kernel object* rather than the method that launches
it: ``MambaHybridModelState.postprocess_state`` is defined in the upstream
module, so its ``__globals__`` *is* the upstream module dict, and rebinding the
name there is picked up at call time. That leaves the surrounding logic --
including the ``int`` branch at ``:311-315`` (plain ``index_fill_``, no Triton)
and the align branch -- completely untouched.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import vllm.v1.worker.gpu.model_states.mamba_hybrid as _up
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

import vllm_kunlun.v1.worker.mamba_utils as _kunlun_mamba_utils
from vllm_kunlun.v1.worker.gpu._kernels import TorchKernel

if TYPE_CHECKING:
    from vllm_kunlun.v1.attention.backends.gdn_attn import GDNAttentionMetadata

logger = logging.getLogger("vllm_kunlun")


def build_hybrid_attn_metadata(
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    dcp_local_seq_lens: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    for_cudagraph_capture: bool = False,
    rswa_prefix_lens: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build GDN batch tensors once per compatible KV spec and forward step."""
    seq_lens = seq_lens[:num_reqs]
    if dcp_local_seq_lens is not None:
        dcp_local_seq_lens = dcp_local_seq_lens[:num_reqs]
    if seq_lens_cpu_upper_bound is not None:
        seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound[:num_reqs]

    attn_metadata: dict[str, Any] = {}
    cached_metadata: dict[MambaSpec, GDNAttentionMetadata] = {}
    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
    for i in range(num_kv_cache_groups):
        groups = attn_groups[i]
        if not groups:
            continue
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]
        common_attn_metadata_extra_kwargs = (
            model_specific_attn_metadata.get_extra_common_attn_kwargs(i, num_reqs)
            if model_specific_attn_metadata is not None
            else {}
        )
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=max_seq_len,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=positions,
            rswa_prefix_lens=rswa_prefix_lens,
            **common_attn_metadata_extra_kwargs,
        )
        for attn_group in groups:
            attn_metadata_builder = attn_group.get_metadata_builder(0)
            reuse_key = None
            source_metadata = None
            # GDN subclasses may own additional per-group metadata.
            if type(attn_metadata_builder) is _KunlunGDNBuilder:
                reuse_key = attn_metadata_builder.kv_cache_spec
                source_metadata = cached_metadata.get(reuse_key)
            else:
                # Other builders can change shared device lengths at capture.
                # Keep the same GDN buffer ownership boundaries at replay.
                cached_metadata.clear()

            if source_metadata is not None:
                metadata = attn_metadata_builder.reuse_metadata(
                    source_metadata,
                    common_attn_metadata,
                )
            elif for_cudagraph_capture:
                metadata = attn_metadata_builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            else:
                attn_metadata_extra_kwargs = (
                    model_specific_attn_metadata.get_extra_attn_kwargs(
                        attn_metadata_builder, num_reqs
                    )
                    if model_specific_attn_metadata is not None
                    else {}
                )
                metadata = attn_metadata_builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    **attn_metadata_extra_kwargs,
                )

            if reuse_key is not None and source_metadata is None:
                cached_metadata[reuse_key] = metadata
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = metadata
    return attn_metadata


def _scatter_num_accepted(idx_mapping, num_sampled, num_accepted) -> None:
    if idx_mapping.numel() == 0:
        return
    torch.ops.xspeedgate_ops.scatter_num_accepted_kernel(
        idx_mapping, num_sampled, num_accepted
    )


_up._scatter_num_accepted_kernel = TorchKernel(_scatter_num_accepted)
_up.preprocess_mamba_align_fused_kernel = (
    _kunlun_mamba_utils._up.preprocess_mamba_align_fused_kernel
)

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
# isort: off
from vllm_kunlun.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder as _KunlunGDNBuilder,
)

# isort: on

# This binding is correctness-critical: without it the upstream isinstance check
# drops speculative-decode metadata. Let import-hook dispatch report and retry
# an installation failure instead of continuing with potentially wrong results.
_up.GDNAttentionMetadataBuilder = _KunlunGDNBuilder
_up.build_attn_metadata = build_hybrid_attn_metadata

logger.info(
    "[KunlunPlugin] V2 MambaHybridModelState patched "
    "(XSpeedGate scatter_num_accepted, mamba align and GDN metadata reuse)"
)

# Request slots can be reused in cache mode "none" as well as "align".
_original_add_request = _up.MambaHybridModelState.add_request


def _add_request(self, req_index, new_req_data):
    _original_add_request(self, req_index, new_req_data)
    self.num_accepted_tokens_gpu[req_index] = 1


_up.MambaHybridModelState.add_request = _add_request
