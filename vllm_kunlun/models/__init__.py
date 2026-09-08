from vllm import ModelRegistry


def register_model():

    # TODO Remove all of models registration below

    # from .demo_model import DemoModel  # noqa: F401

    # ModelRegistry.register_model(
    #     "DemoModel",
    #     "vllm_kunlun.model_executor.models.demo_model:DemoModel")

    ModelRegistry.register_model(
        "Qwen3NextForCausalLM", "vllm_kunlun.models.qwen3_next:Qwen3NextForCausalLM"
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

    # MiniMax-M3 is vendored per backend upstream and selected by platform predicate.
    # Kunlun reports device_type "cuda" with is_cuda_alike/is_cuda/is_rocm/is_xpu/is_cpu
    # all False and is_out_of_tree() True, so the selector hands it the nvidia variant,
    # which needs flashinfer, fmha_sm100 and triton kernels that do not run here.
    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_kunlun.models.minimax_m3:MiniMaxM3SparseForCausalLM",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        "vllm_kunlun.models.minimax_m3:MiniMaxM3SparseForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "MiniMaxM3MTP", "vllm_kunlun.models.minimax_m3:MiniMaxM3MTP"
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
