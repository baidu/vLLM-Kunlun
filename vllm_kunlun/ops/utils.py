#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
# This file is a part of the vllm-kunlun project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities for Kunlun ops."""

from collections.abc import Sequence

import torch
from vllm.v1.worker.workspace import current_workspace_manager


def allocate_temp_tensors(
    specs: Sequence[tuple[tuple[int, ...], torch.dtype, str]],
) -> list[torch.Tensor]:
    """Allocate scratch tensors from the shared workspace manager."""

    tensors = current_workspace_manager().get_simultaneous(
        *((shape, dtype) for shape, dtype, _ in specs)
    )

    for tensor, (_, _, init) in zip(tensors, specs):
        if init not in ("empty", "zeros", "ones"):
            raise ValueError(
                f"Invalid init value {init!r}; expected one of "
                '{"empty", "zeros", "ones"}.'
            )
        if init == "zeros":
            tensor.zero_()
        elif init == "ones":
            tensor.fill_(1)

    return tensors
