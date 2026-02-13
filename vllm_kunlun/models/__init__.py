from vllm import ModelRegistry


def register_model():

    # TODO Remove all of models registration below

    # from .demo_model import DemoModel  # noqa: F401

    # ModelRegistry.register_model(
    #     "DemoModel",
    #     "vllm_kunlun.model_executor.models.demo_model:DemoModel")

    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "vllm_kunlun.models.qwen2_vl:Qwen2VLForConditionalGeneration",
    )

    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",
        "vllm_kunlun.models.qwen2_5_vl:Qwen2_5_VLForConditionalGeneration",
    )

    # Remove in v0.15.1
    # ModelRegistry.register_model(
    #     "Qwen3ForCausalLM",
    #     "vllm_kunlun.models.qwen3:Qwen3ForCausalLM")

    # ModelRegistry.register_model(
    #     "Qwen3MoeForCausalLM",
    #     "vllm_kunlun.models.qwen3_moe:Qwen3MoeForCausalLM")

    ModelRegistry.register_model(
        "Qwen3NextForCausalLM", "vllm_kunlun.models.qwen3_next:Qwen3NextForCausalLM"
    )

    ModelRegistry.register_model(
        "GptOssForCausalLM", "vllm_kunlun.models.gpt_oss:GptOssForCausalLM"
    )

    ModelRegistry.register_model(
        "InternLM2ForCausalLM", "vllm_kunlun.models.internlm2:InternLM2ForCausalLM"
    )

    ModelRegistry.register_model(
        "InternVLChatModel", "vllm_kunlun.models.internvl:InternVLChatModel"
    )

    ModelRegistry.register_model(
        "InternS1ForConditionalGeneration",
        "vllm_kunlun.models.interns1:InternS1ForConditionalGeneration",
    )

    # Remove in v0.15.1
    # ModelRegistry.register_model(
    #     "Qwen3VLForConditionalGeneration",
    #     "vllm_kunlun.models.qwen3_vl:Qwen3VLForConditionalGeneration")

    # ModelRegistry.register_model(
    #     "Qwen3VLMoeForConditionalGeneration",
    #     "vllm_kunlun.models.qwen3_vl_moe:Qwen3VLMoeForConditionalGeneration")

    # ModelRegistry.register_model(
    #     "Qwen3OmniMoeForConditionalGeneration",
    #     "vllm_kunlun.models.qwen3_omni_moe_thinker:Qwen3OmniMoeThinkerForConditionalGeneration")

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


def register_quant_method():
    """to do"""
