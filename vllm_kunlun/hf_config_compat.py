from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5Config as UpstreamQwen3_5Config,
)
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig as UpstreamQwen3_5MoeConfig,
)

from vllm_kunlun.transformers_utils.configs.qwen3_5 import Qwen3_5Config
from vllm_kunlun.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig

QWEN3_5_CONFIG_TYPES = (Qwen3_5Config, UpstreamQwen3_5Config)
QWEN3_5_MOE_CONFIG_TYPES = (Qwen3_5MoeConfig, UpstreamQwen3_5MoeConfig)
