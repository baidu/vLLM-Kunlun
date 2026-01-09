# Single XPU (Qwen3-8B)

## Run vllm-kunlun on Single XPU

Setup environment using container:

```bash
# !/bin/bash
# rundocker.sh
XPU_NUM=8
DOCKER_DEVICE_CONFIG=""
if [ $XPU_NUM -gt 0 ]; then
    for idx in $(seq 0 $((XPU_NUM-1))); do
        DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpu${idx}:/dev/xpu${idx}"
    done
    DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpuctrl:/dev/xpuctrl"
fi

export build_image="xxxxxxxxxxxxxxxxx"

docker run -itd ${DOCKER_DEVICE_CONFIG} \
    --net=host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --tmpfs /dev/shm:rw,nosuid,nodev,exec,size=32g \
    --cap-add=SYS_PTRACE \
    -v /home/users/vllm-kunlun:/home/vllm-kunlun \
    -v /usr/local/bin/xpu-smi:/usr/local/bin/xpu-smi \
    --name "$1" \
    -w /workspace \
    "$build_image" /bin/bash
```

### Offline Inference on Single XPU

Start the server in a container:

```{code-block} bash
from vllm import LLM, SamplingParams

def main():

    model_path = "/models/Qwen3-8B"

    llm_params = {
        "model": model_path,
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "dtype": "float16",
        "enable_chunked_prefill": False,
        "distributed_executor_backend": "mp",
    }

    llm = LLM(**llm_params)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "tell a joke"
                }
            ]
        }
    ]

    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        stop_token_ids=[181896]
    )

    outputs = llm.chat(messages, sampling_params=sampling_params)

    response = outputs[0].outputs[0].text
    print("=" * 50)
    print("Input content:", messages)
    print("Model response:\n", response)
    print("=" * 50)

if __name__ == "__main__":
    main()

```

:::::

If you run this script successfully, you can see the info shown below:

```bash
==================================================
Input content: [{'role': 'user', 'content': [{'type': 'text', 'text': 'tell a joke'}]}]
Model response:
 <think>

Okay, the user asked me to tell a joke. First, I need to consider the user's needs. They might just want to relax or need some entertainment. Next, I need to choose a suitable joke that is not too complicated, easy to understand, and also interesting.


The user might expect the joke to be in Chinese, so I need to ensure that the joke conforms to the language habits and cultural background of Chinese. I need to avoid sensitive topics, such as politics, religion, or anything that might cause misunderstanding. Then, I have to consider the structure of the joke, which usually involves a setup and an unexpected ending to create humor.

For example, I could tell a light-hearted story about everyday life, such as animals or common scenarios. For instance, the story of a turtle and a rabbit racing, but with a twist. However, I need to ensure that the joke is of moderate length and not too long, so the user doesn't lose interest. Additionally, I should pay attention to using colloquial language and avoid stiff or complex sentence structures.

I might also need to check if this joke is common to avoid repetition. If the user has heard something similar before, I may need to come up with a different angle.
==================================================
```

### Online Serving on Single XPU

Start the vLLM server on a single XPU:

```{code-block} bash
python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 9000 \
      --model /models/Qwen3-8B\
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 1 \
      --dtype float16 \
      --max_num_seqs 128 \
      --max_num_batched_tokens 32768 \
      --max-seq-len-to-capture 32768 \
      --block-size 128 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name Qwen3-8B \
      --compilation-config '{"splitting_ops": ["vllm.unified_attention_with_output_kunlun",
            "vllm.unified_attention", "vllm.unified_attention_with_output",
            "vllm.mamba_mixer2"]}' \
```

If your service start successfully, you can see the info shown below:

```bash
(APIServer pid=118459) INFO:     Started server process [118459]
(APIServer pid=118459) INFO:     Waiting for application startup.
(APIServer pid=118459) INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:9000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-8B",
        "prompt": "What is your name?",
        "max_tokens": 100,
        "temperature": 0
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"cmpl-80ee8b893dc64053947b0bea86352faa","object":"text_completion","created":1763015742,"model":"Qwen3-8B","choices":[{"index":0,"text":" is the S, and ,","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":5,"total_tokens":12,"completion_tokens":7,"prompt_tokens_details":null},"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
(APIServer pid=54567) INFO:     127.0.0.1:60338 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=54567) INFO 11-13 14:35:48 [loggers.py:123] Engine 000: Avg prompt throughput: 0.5 tokens/s, Avg generation throughput: 0.7 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```
