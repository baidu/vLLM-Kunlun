# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialize fresh hybrid states before the first target/draft forward."""

import vllm.v1.worker.gpu.model_runner as _up
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_kunlun.registration.compat_patches import _is_dmtp_speculative_config
from vllm_kunlun.v1.worker.mamba_utils import MambaStateZeroer

_original_add_requests = _up.GPUModelRunner.add_requests
_original_execute_model = _up.GPUModelRunner.execute_model


def _add_requests(self, scheduler_output):
    _original_add_requests(self, scheduler_output)
    if self.vllm_config.speculative_config is None:
        return
    context = self.vllm_config.compilation_config.static_forward_context
    state_blocks = []
    for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
        if not isinstance(group.kv_cache_spec, MambaSpec):
            continue
        blocks = {
            b
            for req in scheduler_output.scheduled_new_reqs
            if req.num_computed_tokens == 0
            for b in req.block_ids[gid]
            if b != 0
        }
        if blocks:
            state_blocks.extend(
                (state, blocks)
                for name in group.layer_names
                for state in context[name].kv_cache
            )
    if state_blocks:
        if not hasattr(self, "_kunlun_state_zeroer"):
            self._kunlun_state_zeroer = MambaStateZeroer()
        self._kunlun_state_zeroer.clear(state_blocks)


_up.GPUModelRunner.add_requests = _add_requests


def _execute_model(
    self,
    scheduler_output,
    intermediate_tensors=None,
    dummy_run=False,
    skip_attn_for_dummy_run=False,
    is_profile=False,
):
    spec = self.vllm_config.speculative_config
    manager = self.cudagraph_manager
    scheduled = scheduler_output.num_scheduled_tokens
    drafts = scheduler_output.scheduled_spec_decode_tokens
    # Target FULL graphs capture the speculative GDN path. A prefill with
    # exactly K+1 rows has the same shape but needs the prefill state update.
    # Nonuniform batches already avoid FULL graphs in the upstream dispatcher.
    prefill = (
        not dummy_run
        and manager is not None
        and _is_dmtp_speculative_config(spec)
        and bool(scheduled)
        and all(
            count == spec.num_speculative_tokens + 1 for count in scheduled.values()
        )
        and any(
            len(drafts.get(req_id, ())) != count - 1
            for req_id, count in scheduled.items()
        )
    )
    if not prefill:
        return _original_execute_model(
            self,
            scheduler_output,
            intermediate_tensors,
            dummy_run,
            skip_attn_for_dummy_run,
            is_profile,
        )
    dispatch = manager.dispatch
    had_override = "dispatch" in vars(manager)
    previous = vars(manager).get("dispatch")

    def prefill_dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras):
        # Keep target piecewise/eager execution and its actual prefill metadata.
        # The independent DMTP native and draft graph managers stay enabled.
        return dispatch(num_reqs, num_tokens, None, num_active_loras)

    try:
        manager.dispatch = prefill_dispatch
        return _original_execute_model(
            self,
            scheduler_output,
            intermediate_tensors,
            dummy_run,
            skip_attn_for_dummy_run,
            is_profile,
        )
    finally:
        if had_override:
            manager.dispatch = previous
        else:
            del manager.dispatch


_up.GPUModelRunner.execute_model = _execute_model
