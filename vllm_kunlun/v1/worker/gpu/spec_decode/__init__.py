# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

import vllm.v1.worker.gpu.spec_decode as _up

_original_init_speculator = _up.init_speculator


def is_dflash2(speculative_config) -> bool:
    if speculative_config is None or speculative_config.method != "dflash":
        return False
    draft = speculative_config.draft_model_config
    return "DFlash2DraftModel" in (draft.architectures or [])


def init_speculator(vllm_config, device):
    if is_dflash2(vllm_config.speculative_config):
        from vllm_kunlun.v1.worker.gpu.spec_decode.dflash2.speculator import (
            DFlash2Speculator,
        )

        return DFlash2Speculator(vllm_config, device)
    return _original_init_speculator(vllm_config, device)


_up.init_speculator = init_speculator
# Also cover consumers imported before the hook was installed.
_runner = sys.modules.get("vllm.v1.worker.gpu.model_runner")
if getattr(_runner, "init_speculator", None) is _original_init_speculator:
    _runner.init_speculator = init_speculator
