
def test_demo():
    from vllm import LLM, SamplingParams

    import os

    dir_path = "./Qwen3-0.6B"

    if os.path.isdir(dir_path):
        print(os.path.abspath(dir_path))
    else:
        print("目录不存在")

    llm = LLM(model="/home/model/Qwen3-0.6B", compilation_config="{"splitting_ops": ["vllm.unified_attention", "vllm.unified_attention_with_output", "vllm.unified_attention_with_output_kunlun", "vllm.mamba_mixer2","vllm.mamba_mixer","vllm.short_conv","vllm.linear_attention", "vllm.plamo2_mamba_mixer", "vllm.gdn_attention","vllm.sparse_attn_indexer"]}")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
    )

    outputs = llm.generate(
        ["介绍一下 vLLM"],
        sampling_params,
    )
    print(outputs[0].outputs[0].text)
    assert len(outputs[0].outputs[0].text) > 0


    