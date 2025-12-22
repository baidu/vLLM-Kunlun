from vllm import ModelRegistry


def register_model():
    # from .demo_model import DemoModel  # noqa: F401
    from .qwen2_vl import Qwen2VLForConditionalGeneration #noqa: F401
    from .qwen2_5_vl import Qwen2_5_VLForConditionalGeneration #noqa: F401
    from .qwen3 import Qwen3ForCausalLM #noqa: F401
    from .qwen3_moe import Qwen3MoeForCausalLM #noqa: F401
    
    # ModelRegistry.register_model(
    #     "DemoModel",
    #     "vllm_kunlun.model_executor.models.demo_model:DemoModel")

    ModelRegistry.register_model(
        "LlamaForCausalLM",
        "vllm_kunlun.models.llama:LlamaForCausalLM")
    
    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "vllm_kunlun.models.qwen2_vl:Qwen2VLForConditionalGeneration")

    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",
        "vllm_kunlun.models.qwen2_5_vl:Qwen2_5_VLForConditionalGeneration")

    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "vllm_kunlun.models.qwen3:Qwen3ForCausalLM")

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "vllm_kunlun.models.qwen3_moe:Qwen3MoeForCausalLM")
    
    ModelRegistry.register_model(
        "GlmForCausalLM",
        "vllm_kunlun.models.glm:GlmForCausalLM")  

    ModelRegistry.register_model(
        "GptOssForCausalLM",
        "vllm_kunlun.models.gpt_oss:GptOssForCausalLM")   
    ModelRegistry.register_model(
        "InternLM2ForCausalLM",
        "vllm_kunlun.models.internlm2:InternLM2ForCausalLM")   

    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "vllm_kunlun.models.qwen2:Qwen2ForCausalLM")
    
    ModelRegistry.register_model(
        "InternVLChatModel",
        "vllm_kunlun.models.internvl:InternVLChatModel")

    ModelRegistry.register_model(
        "InternS1ForConditionalGeneration",
        "vllm_kunlun.models.interns1:InternS1ForConditionalGeneration")
    
    ModelRegistry.register_model(
        "Glm4MoeForCausalLM",
        "vllm_kunlun.models.glm4_moe:Glm4MoeForCausalLM")
    
    ModelRegistry.register_model(
        "Glm4ForCausalLM",
        "vllm_kunlun.models.glm4:Glm4ForCausalLM")

    ModelRegistry.register_model(
        "Glm4vForConditionalGeneration",
        "vllm_kunlun.models.glm4_1v:Glm4vForConditionalGeneration")

    ModelRegistry.register_model(
        "Glm4vMoeForConditionalGeneration",
        "vllm_kunlun.models.glm4_1v:Glm4vMoeForConditionalGeneration")


def register_quant_method():
    """to do"""
