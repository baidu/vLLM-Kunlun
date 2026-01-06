
def test_demo():
    from vllm import LLM, SamplingParams

    llm = LLM(model="/home/models/Qwen3-1.7B")

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


    