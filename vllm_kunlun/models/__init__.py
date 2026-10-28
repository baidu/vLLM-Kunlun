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

    ModelRegistry.register_model(
        "Qwen3_5MoeForConditionalGeneration",
        "vllm_kunlun.models.qwen3_5:Qwen3_5MoeForConditionalGeneration",
    )

    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration",
        "vllm_kunlun.models.qwen3_5:Qwen3_5ForConditionalGeneration",
    )

    ModelRegistry.register_model(
        "Qwen4ExpForCausalLM", "vllm_kunlun.models.qwen4_exp:Qwen4ExpForCausalLM"
    )

    # The VL architecture is the multimodal entry point; its vision tower is
    # the same one qwen3_5 uses.
    ModelRegistry.register_model(
        "Qwen4ExpForConditionalGeneration",
        "vllm_kunlun.models.qwen4_exp:Qwen4ExpForConditionalGeneration",
    )

    ModelRegistry.register_model(
        "Qwen4ExpMTP", "vllm_kunlun.models.qwen4_exp:Qwen4ExpMTP"
    )
    _register_qwen4_exp_configs()


def _register_qwen4_exp_configs():
    """Register the qwen4_exp config classes in vLLM's config registry.

    This cannot live in ``registration/qwen4_exp_patches.py``: the registry is
    only importable once vLLM itself has finished importing, which is exactly
    when ``register_model()`` runs.
    """
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from vllm_kunlun.transformers_utils.configs.qwen4_exp import (
        Qwen4ExpConfig,
        Qwen4ExpTextConfig,
    )

    _CONFIG_REGISTRY["qwen4_exp"] = Qwen4ExpConfig
    _CONFIG_REGISTRY["qwen4_exp_text"] = Qwen4ExpTextConfig
