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

"""Kunlun ops package.

Avoid eager side-effect imports here. vLLM 0.19 imports
``vllm.v1.attention.ops.merge_attn_states`` during CLI argument parsing, and our
import hook redirects that path into ``vllm_kunlun.ops.attention``. Pulling in
``vllm_kunlun.ops`` package-wide side effects at that stage forces ``kunlun_ops``
and ``torch_xmlir`` to load before the service is even configured.

The individual submodules still import and register their runtime pieces when
they are imported explicitly by model/attention code.
"""
