from transformers import PretrainedConfig
from vllm.transformers_utils.config import LazyConfigDict, _CONFIG_REGISTRY

_XPU_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = LazyConfigDict(
    chatglm="ChatGLMConfig",
    deepseek_vl_v2="DeepseekVLV2Config",
    deepseek_v3="DeepseekV3Config",
    deepseek_v32="DeepseekV3Config",
    glm_moe_dsa="DeepseekV3Config",
    kimi_vl="KimiVLConfig",
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
