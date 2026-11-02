# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialize fresh hybrid states before the first target/draft forward."""

import vllm.v1.worker.gpu.model_runner as _up
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_kunlun.v1.worker.mamba_utils import MambaStateZeroer

_original_add_requests = _up.GPUModelRunner.add_requests


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
