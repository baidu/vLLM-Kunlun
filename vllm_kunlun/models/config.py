from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import DeepseekV32ForCausalLM

if TYPE_CHECKING:

    from vllm.config import VllmConfig

logger = init_logger(__name__)


@classmethod
def kunlun_verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
    """
    Updated fp8 cache to custom "fp8_ds_mla" format for DeepSeekV32
    """
    hf_config = vllm_config.model_config.hf_config

    # Mirror the check in vllm/model_executor/models/deepseek_v2.py
    is_v32 = hasattr(hf_config, "index_topk")
    assert is_v32

    # For DeepSeekV3.2, we use a custom fp8 format as default (i.e.
    #   "auto")
    cache_config = vllm_config.cache_config
    if cache_config.cache_dtype == "auto" or \
        cache_config.cache_dtype.startswith("fp8"):
        # TODO: When fwd_kvcache_mla supports uint8 kv cache, modify cache_dtype to "fp8_ds_mla"
        # cache_config.cache_dtype = "fp8_ds_mla"
        cache_config.cache_dtype = "auto"
        logger.info("Using custom fp8 kv-cache format for DeepSeekV3.2")
    if cache_config.cache_dtype == "bfloat16":
        cache_config.cache_dtype = "auto"
        logger.info("Using bfloat16 kv-cache for DeepSeekV3.2")
        
DeepseekV32ForCausalLM.verify_and_update_config = kunlun_verify_and_update_config