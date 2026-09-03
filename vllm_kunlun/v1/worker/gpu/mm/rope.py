# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun replacement for ``vllm.v1.worker.gpu.mm.rope``.

Only ``RopeState.prepare_positions`` launches Triton
(``_prepare_rope_positions_kernel``, launched at ``mm/rope.py:118``). It is on
the live path for every mrope / XD-RoPE model, reached from
``DefaultModelState.prepare_inputs`` (``model_states/default.py:114-119``) --
which includes the whole Qwen3-VL family and the Qwen3.5 hybrid VL
architecture, since ``MambaHybridModelState`` extends ``DefaultModelState``.

Everything else in the module (``init_prefill_positions``,
``apply_staged_writes``, ``read_prefill_positions``,
``update_prefill_positions``, ``get_positions``, ``get_rope_state``) is plain
torch and is reused as-is.
"""

import logging

from vllm_kunlun.v1.worker.gpu._kernels import prepare_rope_positions
from vllm_kunlun.v1.worker.gpu._upstream import load_upstream, reexport

logger = logging.getLogger("vllm_kunlun")

_up = load_upstream("vllm.v1.worker.gpu.mm.rope")
reexport(_up, globals())


def _prepare_positions(
    self, idx_mapping, query_start_loc, prefill_lens, num_computed_tokens
) -> None:
    # ``prefill_positions`` is a StagedWriteTensor and ``prefill_delta`` a
    # UvaBackedTensor from gpu.buffer_utils, which is itself swapped for the
    # Kunlun device-tensor version -- ``.gpu`` is valid either way.
    prepare_rope_positions(
        positions=self.positions,
        prefill_positions=self.prefill_positions.gpu,
        prefill_delta=self.prefill_delta.gpu,
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        prefill_lens=prefill_lens,
        num_computed_tokens=num_computed_tokens,
        num_dims=self.num_dims,
        max_model_len=self.max_model_len,
    )


if not getattr(_up.RopeState, "_kunlun_v2_patched", False):
    _up.RopeState.prepare_positions = _prepare_positions
    _up.RopeState._kunlun_v2_patched = True
    logger.info("[KunlunPlugin] V2 RopeState.prepare_positions patched (torch-native)")
