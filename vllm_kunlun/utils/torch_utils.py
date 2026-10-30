# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific replacement for ``vllm.utils.torch_utils`` (UVA aliasing).

``get_accelerator_view_from_cpu_tensor`` returns a zero-copy UVA alias of a
pinned CPU tensor upstream. Kunlun XPU has no such aliasing, so it returns a
real H2D **copy** instead. A caller that mutates the CPU tensor *after* this
call therefore does not see the change on the device; callers that rely on
publish-after-write semantics must copy explicitly -- see
``UvaBufferPool.copy_to_uva`` in ``vllm/v1/worker/gpu/buffer_utils.py``, which
publishes with an explicit ``copy_``.

Note the upstream XPU branch (``torch.ops._C.get_xpu_view_from_cpu_tensor``) is
never taken here: this platform reports ``PlatformEnum.OOT``, not ``XPU``, so
``current_platform.is_xpu()`` is False and the fallback below is the live path.

Triggering: post-import hook from ``vllm_kunlun.__init__``; ``_kunlun_patched`` flag.
"""

import logging

import torch

logger = logging.getLogger("vllm_kunlun")


def get_accelerator_view_from_cpu_tensor_torch(
    cpu_tensor: torch.Tensor,
) -> torch.Tensor:
    if cpu_tensor.numel() == 0:
        return torch.empty_like(cpu_tensor, device="cuda")
    if cpu_tensor.is_pinned():
        return cpu_tensor.cuda(non_blocking=True)
    return cpu_tensor.cuda()


def get_accelerator_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        assert cpu_tensor.is_pinned(), "CPU tensor must be pinned"
        return torch.ops._C.get_xpu_view_from_cpu_tensor(cpu_tensor)
    return get_accelerator_view_from_cpu_tensor_torch(cpu_tensor)


get_accelerator_view_from_cpu_tensor._kunlun_patched = True
