from vllm import ModelRegistry


def register_model():

    # TODO Remove all of models registration below

    ModelRegistry.register_model(
        "Qwen3NextForCausalLM", "vllm_kunlun.models.qwen3_next:Qwen3NextForCausalLM"
    )

    ModelRegistry.register_model(
        "GptOssForCausalLM", "vllm_kunlun.models.gpt_oss:GptOssForCausalLM"
    )

    ModelRegistry.register_model(
        "SeedOssForCausalLM", "vllm_kunlun.models.seed_oss:SeedOssForCausalLM"
    )

    ModelRegistry.register_model(
        "MiMoV2FlashForCausalLM",
        "vllm_kunlun.models.mimo_v2_flash:MiMoV2FlashForCausalLM",
    )

    ModelRegistry.register_model(
        "GptOssForCausalLM", "vllm_kunlun.models.gpt_oss:GptOssForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM", "vllm_kunlun.models.deepseek_v2:DeepseekV3ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV32ForCausalLM", "vllm_kunlun.models.deepseek_v2:DeepseekV3ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepSeekMTPModel", "vllm_kunlun.models.deepseek_mtp:DeepSeekMTP"
    )

    ModelRegistry.register_model(
        "GlmMoeDsaForCausalLM", "vllm_kunlun.models.deepseek_v2:GlmMoeDsaForCausalLM"
    )

    ModelRegistry.register_model(
        "Qwen3_5MoeForConditionalGeneration",
        "vllm_kunlun.models.qwen3_5:Qwen3_5MoeForConditionalGeneration",
    )

    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration",
        "vllm_kunlun.models.qwen3_5:Qwen3_5ForConditionalGeneration",
    )


def register_quant_method():
    """to do"""
