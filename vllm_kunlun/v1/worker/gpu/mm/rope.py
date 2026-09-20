# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native-op override for ``vllm.v1.worker.gpu.mm.rope``.

Only ``RopeState.prepare_positions`` launches Triton
(``_prepare_rope_positions_kernel``, launched at ``mm/rope.py:118``). It is on
the live path for every mrope / XD-RoPE model, reached from
``DefaultModelState.prepare_inputs`` (``model_states/default.py:114-119``) --
which includes the whole Qwen3-VL family and the Qwen3.5 hybrid VL
architecture, since ``MambaHybridModelState`` extends ``DefaultModelState``.

It is replaced with the Kunlun native op
``torch.ops.xspeedgate_ops.prepare_rope_positions``. The native op derives its
work extent from ``query_start_loc`` on-device and takes NO ``max_model_len``
argument (the upstream kernel only needs strides, and clamps positions
internally), so the torch-native ``_kernels.prepare_rope_positions`` -- which
had to take ``max_model_len`` and pay a ``.item()`` host sync to size its work
-- is no longer on the live path. It remains the CPU parity oracle in
``tests/ut/test_mrv2_kernels.py``.

Everything else in the module (``init_prefill_positions``,
``apply_staged_writes``, ``read_prefill_positions``,
``update_prefill_positions``, ``get_positions``, ``get_rope_state``) is plain
torch and is reused as-is.
"""

import logging

import torch
import vllm.v1.worker.gpu.mm.rope as _up

logger = logging.getLogger("vllm_kunlun")


def _prepare_positions(
    self, idx_mapping, query_start_loc, prefill_lens, num_computed_tokens
) -> None:
    # ``prefill_positions`` is a StagedWriteTensor and ``prefill_delta`` a
    # UvaBackedTensor from gpu.buffer_utils, whose Kunlun patch may swap the
    # UVA views for plain device tensors -- ``.gpu`` is valid either way.
    torch.ops.xspeedgate_ops.prepare_rope_positions(
        self.positions,
        self.prefill_positions.gpu,
        self.prefill_delta.gpu,
        idx_mapping,
        query_start_loc,
        prefill_lens,
        num_computed_tokens,
        self.num_dims,
    )


_up.RopeState.prepare_positions = _prepare_positions
logger.info(
    "[KunlunPlugin] V2 RopeState.prepare_positions patched (xspeedgate_ops native)"
)
