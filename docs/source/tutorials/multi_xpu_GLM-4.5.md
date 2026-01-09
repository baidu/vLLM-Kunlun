# Multi XPU (GLM-4.5)

## Run vllm-kunlun on multi XPU

Setup environment using container:

```bash
docker run -itd \
        --net=host \
        --cap-add=SYS_PTRACE --security-opt=seccomp=unconfined \
        --ulimit=memlock=-1 --ulimit=nofile=120000 --ulimit=stack=67108864 \
        --shm-size=128G \
        --privileged \
        --name=glm-vllm-01011 \
        -v ${PWD}:/data \
        -w /workspace \
        -v /usr/local/bin/:/usr/local/bin/ \
        -v /lib/x86_64-linux-gnu/libxpunvidia-ml.so.1:/lib/x86_64-linux-gnu/libxpunvidia-ml.so.1 \
        iregistry.baidu-int.com/hac_test/aiak-inference-llm:xpu_dev_20251113_221821 bash

docker exec -it glm-vllm-01011 /bin/bash
```

### Offline Inference on multi XPU

Start the server in a container:

```{code-block} bash
   :substitutions:
import os
from vllm import LLM, SamplingParams

def main():

    model_path = "/data/GLM-4.5"

    llm_params = {
        "model": model_path,
        "tensor_parallel_size": 8,
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
                    "text": "Hello, who are you?"
                }
            ]
        }
    ]

    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.7,
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
Input content: [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello, who are you?'}]}]
Model response:
 <think>
Well, the user asked a rather direct question about identity. This question seems simple, but there could be several underlying intentions—perhaps they are testing my reliability for the first time, or they simply want to confirm the identity of the conversational partner. From the common positioning of AI assistants, the user has provided a clear and flat way to define identity while leaving room for potential follow-up questions.\n\nThe user used "you" instead of "your", which leans towards a more informal tone, so the response style can be a bit more relaxed. However, since this is the initial response, it is better to maintain a moderate level of professionalism. Mentioning
==================================================
```

### Online Serving on Single XPU

Start the vLLM server on a single XPU:

```{code-block} bash
python -m vllm.entrypoints.openai.api_server \
      --host localhost \
      --port 8989 \
      --model /data/GLM-4.5 \
      --gpu-memory-utilization 0.95 \
      --trust-remote-code \
      --max-model-len 131072 \
      --tensor-parallel-size 8 \
      --dtype float16 \
      --max_num_seqs 128 \
      --max_num_batched_tokens 4096 \
      --max-seq-len-to-capture 4096 \
      --block-size 128 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name GLM-4.5 \
      --compilation-config '{"splitting_ops": ["vllm.unified_attention_with_output_kunlun", "vllm.unified_attention", "vllm.unified_attention_with_output", "vllm.mamba_mixer2"]}'  > log_glm_plugin.txt 2>&1 &
```

If your service start successfully, you can see the info shown below:

```bash
(APIServer pid=51171) INFO:     Started server process [51171]
(APIServer pid=51171) INFO:     Waiting for application startup.
(APIServer pid=51171) INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8989/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-4.5",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-6af7318de7394bc4ae569e6324a162fa","object":"chat.completion","created":1763101638,"model":"GLM-4.5","choices":[{"index":0,"message":{"role":"assistant","content":"\n<think>The user asked, \"Hello, who are you?\" This is a question about my identity. First, I need to confirm the user's intent. They might be using this service for the first time or have never interacted with similar AI assistants before, so they want to know my background and capabilities.\n\nNext, I should ensure my answer is clear and friendly, focusing on key points: who I am, who developed me, and what I can do. I should avoid technical jargon and keep the response conversational so it's easy to understand.\n\nAdditionally, the user may have potential needs, such as wanting to know what I am capable of.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"length","stop_reason":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":11,"total_tokens":111,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"kv_tr
```

Logs of the vllm server:

```bash
(APIServer pid=54567) INFO:     127.0.0.1:60338 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=54567) INFO 11-13 14:35:48 [loggers.py:123] Engine 000: Avg prompt throughput: 0.5 tokens/s, Avg generation throughput: 0.7 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```
