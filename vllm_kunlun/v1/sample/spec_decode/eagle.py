# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun overrides for the EAGLE / MTP speculative-decode proposer."""

import os
from dataclasses import replace

import numpy as np
import torch
from vllm.compilation import monitor
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)

PADDING_SLOT_ID = -1

_orig_prepare_next_token_ids_padded = EagleProposer.prepare_next_token_ids_padded
_orig_prepare_inputs_padded = EagleProposer.prepare_inputs_padded
_orig_update_positions_dependent_metadata = (
    EagleProposer._update_positions_dependent_metadata
)
_orig_eagle_init = EagleProposer.__init__
_orig_load_model = EagleProposer.load_model

# NOTE: the drafter is still restricted to PIECEWISE cudagraphs upstream
# (llm_base_proposer.py::initialize_cudagraph_keys). That was originally a
# *correctness* requirement: ``CUDAGraphWrapper`` never copies runtime inputs
# into persistent buffers, and the drafter rebuilt its attention metadata from
# freshly allocated tensors on every step
# (``build_per_group_and_layer_attn_metadata``), so a FULL graph baked in those
# pointers and replays read stale slot mappings / state indices. Measured effect
# when this was attempted: acceptance rate collapsed from ~60% to 2.95%
# (acceptance length 1.06) while the step got slower.
#
# The metadata addresses are stable now (``KunlunAttentionMetadataBuilder.build``
# stages every lod / seq-len field, host *and* device, into persistent buffers),
# and the kernels were measured to be graph-safe: under capture/replay
# ``speculative_attention`` follows the *device* ``context_lens`` (changing only
# the host mirror has no effect, and replay is bit-exact across block-count
# changes in both directions), and ``reshape_and_cache_flash`` follows the
# runtime slot mapping including the -1 padding. So the opt-in path below exists;
# it is still off by default and must be validated with the per-position
# acceptance rate, not just address stability.


def _is_qwen35_mtp(self) -> bool:
    # 上游会把 draft 的 model_type 归一化：qwen3_next -> "qwen3_next_mtp"，
    # qwen3_5/qwen3_5_moe -> "qwen3_5_mtp"（vllm/config/speculative.py）。同时
    # self.method 会被统一改写成 "mtp"，故不能用它来识别具体模型。
    hf_config = getattr(getattr(self, "draft_model_config", None), "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    return model_type in ("qwen3_5_mtp", "qwen3_next_mtp")


def prepare_next_token_ids_padded(
    self,
    sampled_token_ids: torch.Tensor,
    requests: dict[str, CachedRequestState],
    gpu_input_batch: InputBatch,
    discard_request_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch-native replacement (Qwen3.5-MTP) for the Triton kernel
    ``eagle_prepare_next_token_padded_kernel``。
    """
    if not _is_qwen35_mtp(self):
        return _orig_prepare_next_token_ids_padded(
            self,
            sampled_token_ids,
            requests,
            gpu_input_batch,
            discard_request_mask,
        )

    # Precompute backup token ids
    num_reqs = gpu_input_batch.num_reqs
    self.backup_next_token_ids.np[:num_reqs] = np.array(
        [
            requests[gpu_input_batch.req_ids[i]].get_token_id(
                gpu_input_batch.num_tokens_no_spec[i] - 1
            )
            for i in range(num_reqs)
        ],
        dtype=np.int32,
    )
    self.backup_next_token_ids.copy_to_gpu(num_reqs)

    # Mask out discarded requests' sampled tokens.
    discard_sampled_tokens_req_indices = torch.nonzero(
        discard_request_mask[:num_reqs], as_tuple=False
    ).flatten()

    valid_sampled_token_ids_gpu = sampled_token_ids.clone()

    if discard_sampled_tokens_req_indices.numel() > 0:
        idx = discard_sampled_tokens_req_indices
        if idx.device != valid_sampled_token_ids_gpu.device:
            idx = idx.to(valid_sampled_token_ids_gpu.device, non_blocking=True)
        if idx.dtype != torch.long:
            idx = idx.to(torch.long)
        valid_sampled_token_ids_gpu.index_fill_(0, idx, -1)

    valid_mask = (valid_sampled_token_ids_gpu != -1) & (
        valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
    )
    valid_sampled_tokens_count_long = valid_mask.sum(dim=1)
    valid_sampled_tokens_count = valid_sampled_tokens_count_long.to(torch.int32)

    last_valid_indices = valid_sampled_tokens_count_long - 1
    last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)
    selected_tokens = torch.gather(
        valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
    ).squeeze(1)

    batch_size = valid_sampled_token_ids_gpu.shape[0]
    next_token_ids = torch.where(
        last_valid_indices != -1,
        selected_tokens,
        self.backup_next_token_ids.gpu[:batch_size],
    )

    return next_token_ids, valid_sampled_tokens_count


def prepare_inputs_padded(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    spec_decode_metadata: SpecDecodeMetadata,
    valid_sampled_tokens_count: torch.Tensor,
) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
    """Torch-native replacement (仅 Qwen3.5-MTP) for the Triton kernel
    ``eagle_prepare_inputs_padded_kernel``"""
    if not _is_qwen35_mtp(self):
        return _orig_prepare_inputs_padded(
            self,
            common_attn_metadata,
            spec_decode_metadata,
            valid_sampled_tokens_count,
        )

    num_reqs = common_attn_metadata.num_reqs

    cu_num_draft = spec_decode_metadata.cu_num_draft_tokens
    num_draft = cu_num_draft.clone()
    if num_reqs > 1:
        num_draft[1:] = cu_num_draft[1:] - cu_num_draft[:-1]

    valid_count = valid_sampled_tokens_count.to(num_draft.dtype)
    num_rejected = num_draft + 1 - valid_count
    num_rejected = torch.where(
        num_draft > 0, num_rejected, torch.zeros_like(num_rejected)
    )

    q_last_tok_idx = common_attn_metadata.query_start_loc[1:] - 1
    token_indices_to_sample = (q_last_tok_idx - num_rejected).to(torch.int32)
    num_rejected_tokens_gpu = num_rejected.to(torch.int32)

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    total_num_tokens = query_start_loc_cpu[-1].item()

    spec_common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=common_attn_metadata.query_start_loc,
        seq_lens=common_attn_metadata.seq_lens,
        query_start_loc_cpu=query_start_loc_cpu,
        _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
        _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
        num_reqs=common_attn_metadata.num_reqs,
        num_actual_tokens=total_num_tokens,
        max_query_len=new_query_len_per_req.max().item(),
        max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
        block_table_tensor=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
        causal=True,
        dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
    )

    return (
        spec_common_attn_metadata,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
    )


def _update_positions_dependent_metadata(
    self,
    positions: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,
    batch_size: int,
    input_batch_size: int,
    block_size: int,
) -> torch.Tensor:
    """Torch-native replacement (仅 Qwen3.5-MTP) for the Triton kernel
    ``eagle_step_slot_mapping_metadata_kernel``。
    """
    if not _is_qwen35_mtp(self):
        return _orig_update_positions_dependent_metadata(
            self,
            positions,
            common_attn_metadata,
            batch_size,
            input_batch_size,
            block_size,
        )

    cad = common_attn_metadata
    positions_1d = positions[0] if self.uses_mrope else positions

    # Pick the destination buffer up front so ``positions + 1`` can be written
    # straight into it.
    #
    # PERF: every op here is a separate dispatch, and the drafter runs this once
    # per draft step at batch 1, where the tensors are a handful of elements --
    # so the cost is pure dispatch overhead, not compute. Profiling showed this
    # function alone issuing 167 of the ~1140 eager ops in one ``propose`` (the
    # single largest contributor). The in-place / ``out=`` forms below avoid the
    # ``zeros_like`` / ``full_like`` / ``ones_like`` temporaries and the extra
    # ``copy_`` that the straightforward version needs.
    if self.uses_mrope:
        out_pos = self.mrope_positions[0, :batch_size]
    elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
        out_pos = self.xdrope_positions[0, :batch_size]
    else:
        out_pos = self.positions[:batch_size]

    # positions + 1, with out-of-range entries folded to 0; their slot is set to
    # PADDING_SLOT_ID below, so the value itself does not matter.
    torch.add(positions_1d, 1, out=out_pos)
    exceeds_max = out_pos >= self.max_model_len
    out_pos.masked_fill_(exceeds_max, 0)
    clamped_position = out_pos

    n_blocks_per_req = cad.block_table_tensor.shape[1]
    block_number = clamped_position.div(block_size, rounding_mode="floor")
    block_number.clamp_(max=n_blocks_per_req - 1)
    # gather() returns a fresh tensor, so the slot arithmetic can run in place.
    slot = cad.block_table_tensor.gather(dim=1, index=block_number.view(-1, 1)).view(-1)
    slot = slot.mul_(block_size).add_(clamped_position.remainder(block_size))
    slot.masked_fill_(exceeds_max, PADDING_SLOT_ID)

    # seq_lens += 1, reset to 1 where out of range, then clamp -- in place on
    # cad.seq_lens instead of building a new tensor and copying it back.
    cad.seq_lens.add_(1).masked_fill_(exceeds_max, 1).clamp_(max=self.max_model_len)

    self._slot_mapping_buffer[:batch_size].copy_(slot)
    if input_batch_size > batch_size:
        self._slot_mapping_buffer[batch_size:input_batch_size].fill_(PADDING_SLOT_ID)
    cad.slot_mapping = self._slot_mapping_buffer[:batch_size]

    if self.uses_mrope:
        self.mrope_positions[1:, :batch_size] = out_pos
        positions = self.mrope_positions[:, :batch_size]
    elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
        self.xdrope_positions[1:, :batch_size] = out_pos
        positions = self.xdrope_positions[0, :batch_size]
    else:
        positions = out_pos

    cad.max_seq_len = min(cad.max_seq_len + 1, self.max_model_len)

    if cad._seq_lens_cpu is not None:
        cad._seq_lens_cpu += 1
    if cad._num_computed_tokens_cpu is not None:
        cad._num_computed_tokens_cpu += 1
    if cad.seq_lens_cpu_upper_bound is not None:
        cad.seq_lens_cpu_upper_bound += 1

    return positions


def _patched_eagle_init(self, *args, **kwargs):
    _orig_eagle_init(self, *args, **kwargs)
    if (
        _is_qwen35_mtp(self)
        and getattr(self, "uses_mrope", False)
        and not hasattr(self, "positions")
    ):
        self.positions = self.mrope_positions


EagleProposer.prepare_next_token_ids_padded = prepare_next_token_ids_padded
EagleProposer.prepare_inputs_padded = prepare_inputs_padded
EagleProposer._update_positions_dependent_metadata = (
    _update_positions_dependent_metadata
)
EagleProposer.__init__ = _patched_eagle_init


# --------------------------------------------------------------------------
# Experimental: FULL cudagraphs for the drafter. Off unless
# ``VLLM_KUNLUN_DRAFTER_FULL_CUDAGRAPH=1``.
#
# Upstream gives the drafter no FULL path at all: ``initialize_cudagraph_keys``
# hardcodes PIECEWISE, and ``CUDAGraphWrapper`` is only ever wrapped around the
# *target* model (gpu_model_runner.py). Capture also cannot go through the
# drafter's ``dummy_run``: it calls ``set_forward_context(None, ...)``, so the
# attention layers take the ``attn_metadata is None`` profiling shortcut and a
# graph captured there would contain no attention at all. The proposer holds no
# reference to the runner either, so it cannot synthesize capture-time metadata
# pointing at the runner's persistent ``seq_lens`` / block table -- and those are
# handed to the kernels unstaged, so their addresses must match capture time.
# Hence: capture lazily, on a real drafting step, where the real metadata is.
# --------------------------------------------------------------------------


def _drafter_full_cudagraph_enabled() -> bool:
    return os.getenv("VLLM_KUNLUN_DRAFTER_FULL_CUDAGRAPH", "0") == "1"


def _uniform_decode_num_reqs(attn_metadata) -> int | None:
    """Request count if every attention group is a query-len-1 decode batch.

    ``None`` means "do not use a graph for this call". The first drafter forward
    in ``propose`` is a mixed prefill/decode batch, and ``KunlunImpl.forward``
    picks between the prefill and decode kernels on the host from
    ``num_prefills`` / ``num_decodes``; baking that choice into a graph would
    silently attend over the wrong rows on a later step with a different split.
    """
    if not attn_metadata:
        return None
    groups = (
        attn_metadata.values() if isinstance(attn_metadata, dict) else [attn_metadata]
    )
    num_reqs = None
    for md in groups:
        if getattr(md, "num_prefills", 1) != 0:
            return None
        if getattr(md, "max_decode_seq_len", 0) != 1:
            return None
        block_tables = getattr(md, "block_tables", None)
        if block_tables is None:
            return None
        if num_reqs is not None and block_tables.shape[0] != num_reqs:
            return None
        num_reqs = block_tables.shape[0]
    return num_reqs


class _DrafterCUDAGraphWrapper(CUDAGraphWrapper):
    """FULL cudagraphs for the drafter, captured on the first real step.

    Deliberately does *not* touch the cudagraph dispatcher. The dispatcher only
    sees a token count and always dispatches with ``uniform_decode=False``
    (llm_base_proposer::_determine_batch_execution_and_padding), so it cannot
    tell a draft step from the mixed prefill/decode first pass -- and switching
    its mode to FULL would strip the drafter's PIECEWISE keys, leaving the first
    pass eager. Instead this wrapper triggers on the PIECEWISE mode the
    dispatcher already emits and decides per call:

    * mixed batch -> pass through, so the inner piecewise graphs run as before.
      ``KunlunImpl.forward`` picks between the prefill and decode kernels on the
      host from ``num_prefills``/``num_decodes``, so baking that choice into a
      graph would silently attend over the wrong rows on a later step.
    * uniform query-len-1 decode -> run the whole drafter forward from one FULL
      graph. The runtime mode is flipped to FULL for the duration so the inner
      piecewise wrappers see a mode that is not theirs and pass through, rather
      than trying to capture inside our capture.

    Two more deviations from the base wrapper:

    * The unpadded request count is folded into the cache key. The dispatcher's
      descriptor only carries the *padded* token count, while what the graph
      bakes follows the unpadded count (``block_tables.shape[0]`` becomes the
      kernel's ``batch_num``), so two batch sizes that pad to the same size must
      not share a graph.
    * Capture runs inside ``graph_capture()`` (a non-default stream is required)
      and is followed by an explicit replay: capture executes nothing, so the
      base wrapper's post-capture return value is uninitialized memory. During
      warmup that is harmless, but here the caller is a real drafting step.
    """

    def __init__(self, runnable, vllm_config, device):
        super().__init__(runnable, vllm_config, CUDAGraphMode.FULL)
        self._device = device

    def __call__(self, *args, **kwargs):
        if not is_forward_context_available():
            return self.runnable(*args, **kwargs)

        ctx = get_forward_context()
        if (
            ctx.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
            or ctx.batch_descriptor is None
        ):
            return self.runnable(*args, **kwargs)

        num_reqs = _uniform_decode_num_reqs(ctx.attn_metadata)
        if num_reqs is None:
            return self.runnable(*args, **kwargs)

        prev_desc = ctx.batch_descriptor
        desc = replace(prev_desc, num_reqs=num_reqs, uniform=True)
        entry = self.concrete_cudagraph_entries.get(desc)
        prev_mode = ctx.cudagraph_runtime_mode
        prev_enabled = monitor.cudagraph_capturing_enabled
        ctx.batch_descriptor = desc
        ctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
        try:
            if entry is not None and entry.cudagraph is not None:
                return super().__call__(*args, **kwargs)
            monitor.set_cudagraph_capturing_enabled(True)
            # ``torch.cuda.graph.__enter__`` calls ``torch.cuda.empty_cache()``
            # (torch/cuda/graphs.py). During warmup that is free; in a live
            # process it drops the allocator cache and every following
            # allocation goes back to cudaMalloc. Measured ~40% of the capture
            # cost, plus the aftermath. vLLM's own ``CUDAGraphOptions.gc_disable``
            # does not cover this: it patches ``torch.accelerator.empty_cache``,
            # a different function from the one graphs.py calls. ``gc.collect()``
            # is already gated off by ``torch.compiler.config.force_cudagraph_gc``.
            orig_empty_cache = torch.cuda.empty_cache
            torch.cuda.empty_cache = lambda *a, **kw: None
            try:
                with graph_capture(self._device):
                    super().__call__(*args, **kwargs)
            finally:
                torch.cuda.empty_cache = orig_empty_cache
            entry = self.concrete_cudagraph_entries[desc]
            entry.cudagraph.replay()
            logger.info("[KunlunPlugin] drafter captured a FULL cudagraph %s", desc)
            return entry.output
        finally:
            monitor.set_cudagraph_capturing_enabled(prev_enabled)
            ctx.batch_descriptor = prev_desc
            ctx.cudagraph_runtime_mode = prev_mode


def _patched_load_model(self, target_model) -> None:
    _orig_load_model(self, target_model)
    if not _drafter_full_cudagraph_enabled():
        return
    self.model = _DrafterCUDAGraphWrapper(self.model, self.vllm_config, self.device)
    logger.info("[KunlunPlugin] drafter model wrapped for FULL cudagraphs")


EagleProposer.load_model = _patched_load_model
