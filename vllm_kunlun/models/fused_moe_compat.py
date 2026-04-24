#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
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
# This file is a part of the vllm-kunlun project.
#

import inspect
from typing import Any

import torch


def make_expert_params_mapping(
    fused_moe_cls: type[Any],
    model: torch.nn.Module,
    **kwargs: Any,
):
    method = fused_moe_cls.make_expert_params_mapping
    if "model" in inspect.signature(method).parameters:
        return method(model, **kwargs)
    return method(**kwargs)
