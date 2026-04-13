import importlib
from transformers import PretrainedConfig
from vllm.transformers_utils.config import LazyConfigDict, _CONFIG_REGISTRY


def patch_transformers_utils_configs_for_glm4() -> None:
    """LazyConfigDict resolves class names via getattr(vllm.transformers_utils.configs, name).

    vLLM 自带的 configs 包未必注册 Glm4Moe*，但 transformers 已提供，在此挂到 configs 模块上，
    否则加载 glm4_moe_lite / glm4_moe 会报 AttributeError。
    """
    import vllm.transformers_utils.configs as configs

    fallbacks: list[tuple[str, str, str]] = [
        ("Glm4MoeLiteConfig", "transformers.models.glm4_moe_lite", "Glm4MoeLiteConfig"),
        ("Glm4MoeConfig", "transformers.models.glm4_moe", "Glm4MoeConfig"),
    ]
    for attr_name, mod_path, cls_name in fallbacks:
        if hasattr(configs, attr_name):
            continue
        try:
            mod = importlib.import_module(mod_path)
            setattr(configs, attr_name, getattr(mod, cls_name))
        except Exception:
            pass


_XPU_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = LazyConfigDict(
    chatglm="ChatGLMConfig",
    deepseek_vl_v2="DeepseekVLV2Config",
    deepseek_v3="DeepseekV3Config",
    deepseek_v32="DeepseekV3Config",
    glm_moe_dsa="DeepseekV3Config",
    glm4_moe="Glm4MoeConfig",
    glm4_moe_mtp="Glm4MoeConfig",
    # NOTE: vLLM 0.11's `vllm.transformers_utils.configs` may not expose
    # `Glm4MoeLiteConfig`, causing AttributeError during config parsing.
    # Use `Glm4MoeConfig` (same as non-Flash GLM4.7) to keep the launch path
    # consistent and robust; any extra Flash-specific fields will still be
    # attached as attributes by `PretrainedConfig` via **kwargs.
    glm4_moe_lite="Glm4MoeConfig",
    glm4_moe_lite_mtp="Glm4MoeConfig",
    kimi_vl="KimiVLConfig",
    kimi_k25="KimiK25Config",
    Llama_Nemotron_Nano_VL="Nemotron_Nano_VL_Config",
    RefinedWeb="RWConfig",  # For tiiuae/falcon-40b(-instruct)
    RefinedWebModel="RWConfig",  # For tiiuae/falcon-7b(-instruct)
    jais="JAISConfig",
    mlp_speculator="MLPSpeculatorConfig",
    medusa="MedusaConfig",
    midashenglm="MiDashengLMConfig",
    eagle="EAGLEConfig",
    speculators="SpeculatorsConfig",
    nemotron="NemotronConfig",
    olmo3="Olmo3Config",
    ovis="OvisConfig",
    ultravox="UltravoxConfig",
    step3_vl="Step3VLConfig",
    step3_text="Step3TextConfig",
    qwen3_next="Qwen3NextConfig",
)
