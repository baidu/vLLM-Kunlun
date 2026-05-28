#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import sys

# ``_custom_ops`` registers torch custom ops via ``@custom_op`` decorators.
# Each decorator may be executed at most ONCE per process. The plugin's
# ``register()`` already loads ``_custom_ops.py`` directly via
# ``spec_from_file_location`` (under a private module name) BEFORE this
# package's ``__init__.py`` is reached, in order to avoid a circular
# import chain through ``vllm.model_executor.layers.fused_moe.*``.
# Skip the second registration when that has already happened.
if "_vllm_kunlun_custom_ops_registration" not in sys.modules:
    import vllm_kunlun.ops._custom_ops  # noqa: F401

import vllm_kunlun.ops.fused_moe.layer  # noqa: E402,F401

# base layers
import vllm_kunlun.ops.layernorm
import vllm_kunlun.ops.linear

# embedding
import vllm_kunlun.ops.rotary_embedding
import vllm_kunlun.ops.vocab_parallel_embedding

# Spec-decode helpers (eagle / dflash) import upstream symbols that may not
# exist on every vllm version (e.g. ``vllm.v1.attention.backends.tree_attn``
# was removed in 0.18.0). They are only used when speculative decoding is
# enabled, so make their import optional.
try:
    import vllm_kunlun.v1.sample.spec_decode.dflash  # noqa: F401
except ImportError as _e:
    import logging as _logging

    _logging.getLogger("vllm_kunlun").debug(
        "[KunlunPlugin] spec_decode.dflash unavailable: %s", _e
    )
try:
    import vllm_kunlun.v1.sample.spec_decode.eagle  # noqa: F401
except ImportError as _e:
    import logging as _logging

    _logging.getLogger("vllm_kunlun").debug(
        "[KunlunPlugin] spec_decode.eagle unavailable: %s", _e
    )

# TODO @xyDong0223 remove v0.16.0
# import vllm_kunlun.ops.mla
