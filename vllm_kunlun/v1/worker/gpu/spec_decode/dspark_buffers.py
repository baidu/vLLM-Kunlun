# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent device metadata shared by DSpark input prep and attention builders."""

from __future__ import annotations

import logging
import threading

import torch

_lock = threading.Lock()
_kv_lod_by_seq_ptr: dict[tuple[str, int | None, int], torch.Tensor] = {}


def _key(seq_lens: torch.Tensor) -> tuple[str, int | None, int]:
    return (seq_lens.device.type, seq_lens.device.index, seq_lens.data_ptr())


def get_or_create_kv_lod(seq_lens: torch.Tensor, capacity: int) -> torch.Tensor:
    """Return an address-stable int32 [capacity + 1] LOD buffer."""
    key = _key(seq_lens)
    with _lock:
        buf = _kv_lod_by_seq_ptr.get(key)
        if buf is None or buf.numel() < capacity + 1:
            buf = torch.empty(capacity + 1, dtype=torch.int32, device=seq_lens.device)
            _kv_lod_by_seq_ptr[key] = buf
            logging.getLogger("vllm_kunlun").info(
                "[KunlunPlugin] DSpark persistent KV LOD allocated: seq_ptr=%d "
                "lod_ptr=%d capacity=%d registry_size=%d",
                seq_lens.data_ptr(),
                buf.data_ptr(),
                capacity,
                len(_kv_lod_by_seq_ptr),
            )
        return buf


def lookup_kv_lod(seq_lens: torch.Tensor) -> torch.Tensor | None:
    return _kv_lod_by_seq_ptr.get(_key(seq_lens))
