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
# This file is a part of the vllm-kunlun project.
#
"""
Kunlun-optimized Rotary Embedding implementations using vLLM's CustomOp.register_oot mechanism.

Design:
- Uses @CustomOp.register_oot to register Kunlun-optimized RotaryEmbedding classes
- These classes automatically replace the default implementations when instantiated
- Since KunlunPlatform uses _enum=PlatformEnum.OOT, dispatch_forward() selects
  forward_oot, so we implement forward_oot

OOT Mechanism:
- When code calls RotaryEmbedding(...), vLLM's CustomOp.__new__ checks op_registry_oot
- If "RotaryEmbedding" is found in OOT registry, it returns KunlunRotaryEmbedding instance instead
- This is the official vLLM way to replace operators without modifying source code
"""

import logging

from vllm_kunlun.ops.rotary_embedding.gemma4_rope import (  # noqa: F401
    Gemma4RotaryEmbedding,
)
from vllm_kunlun.ops.rotary_embedding.kunlun_deepseek_rope import (  # noqa: F401
    KunlunDeepseekScalingRotaryEmbedding,
)
from vllm_kunlun.ops.rotary_embedding.kunlun_mrope import (  # noqa: F401
    KunlunMRotaryEmbedding,
)
from vllm_kunlun.ops.rotary_embedding.kunlun_rope import (  # noqa: F401
    KunlunRotaryEmbedding,
)
from vllm_kunlun.ops.rotary_embedding.utils import Split_Norm_Rope  # noqa: F401

logger = logging.getLogger("vllm_kunlun.ops.rotary_embedding")

# Log that OOT registration is complete
logger.info(
    "[KunlunOOT] Registered KunlunRotaryEmbedding, KunlunMRotaryEmbedding, "
    "KunlunDeepseekScalingRotaryEmbedding via CustomOp.register_oot"
)
