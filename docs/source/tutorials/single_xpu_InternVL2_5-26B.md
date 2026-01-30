# Single XPU (InternVL2_5-26B)

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

export build_image="iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.32"


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

```bash
from vllm import LLM, SamplingParams

def main():

    model_path = "/models/InternVL2_5-26B"

    llm_params = {
        "model": model_path,
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "dtype": "float16",
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "distributed_executor_backend": "mp",
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.9,
    }

    llm = LLM(**llm_params)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "你好！你是谁？"
                }
            ]
        }
    ]

    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=0.7,
        top_k=50,
        top_p=0.9
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
Input content: [{'role': 'user', 'content': [{'type': 'text', 'text': '你好！你是谁？'}]}]
Model response:
 你好！我是一个由人工智能驱动的助手，旨在帮助回答问题、提供信息和解决日常问题。请问有什么我可以帮助你的？
==================================================
```

### Online Serving on Single XPU

Start the vLLM server on a single XPU:

```bash
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 9988 \
    --model /models/InternVL2_5-26B \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --max_num_seqs 128 \
    --max_num_batched_tokens 32768 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --distributed-executor-backend mp \
    --served-model-name InternVL2_5-26B \
    --compilation-config '{"splitting_ops": ["vllm.unified_attention",
                                                "vllm.unified_attention_with_output",
                                                "vllm.unified_attention_with_output_kunlun",
                                                "vllm.mamba_mixer2",
                                                "vllm.mamba_mixer",
                                                "vllm.short_conv",
                                                "vllm.linear_attention",
                                                "vllm.plamo2_mamba_mixer",
                                                "vllm.gdn_attention",
                                                "vllm.sparse_attn_indexer"]}
                                                #Version 0.11.0
```

If your service start successfully, you can see the info shown below:

```bash
(APIServer pid=157777) INFO:     Started server process [157777]
(APIServer pid=157777) INFO:     Waiting for application startup.
(APIServer pid=157777) INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:9988/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "InternVL2_5-26B",
        "prompt": "你好！你是谁?",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"cmpl-23a24afd616d4a47910aeeccb20921ed","object":"text_completion","created":1768891222,"model":"InternVL2_5-26B","choices":[{"index":0,"text":" 你有什么问题吗?\n\n你好！我是书生·AI，很高兴能与你交流。请问有什么我可以帮助你的吗？无论是解答问题、提供信息还是其他方面的帮助，我都会尽力而为。请告诉我你的需求。","logprobs":null,"finish_reason":"stop","stop_reason":92542,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":6,"total_tokens":53,"completion_tokens":47,"prompt_tokens_details":null},"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
(APIServer pid=161632) INFO:     127.0.0.1:56708 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=161632) INFO 01-20 14:40:25 [loggers.py:127] Engine 000: Avg prompt throughput: 0.6 tokens/s, Avg generation throughput: 4.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=161632) INFO 01-20 14:40:35 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

Input an image for testing.Here,a python script is used:

```python
import requests
import base64
API_URL = "http://localhost:9988/v1/chat/completions"
MODEL_NAME = "InternVL2_5-26B"
IMAGE_PATH = "/images.jpeg"
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
base64_image = encode_image(IMAGE_PATH)
payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "你好！请描述一下这张图片。"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 300,
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 50
}
response = requests.post(API_URL, json=payload)
print(response.json())
```

If you query the server successfully, you can see the info shown below (client):

```bash
{'id': 'chatcmpl-9aeab6044795458da04f2fdcf1d0445d', 'object': 'chat.completion', 'created': 1768891349, 'model': 'InternVL2_5-26B', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '你好！这张图片上有一个黄色的笑脸表情符号，双手合十，旁边写着“Hugging Face”。这个表情符号看起来很开心，似乎在表示拥抱或欢迎。', 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'tool_calls': [], 'reasoning_content': None}, 'logprobs': None, 'finish_reason': 'stop', 'stop_reason': 92542, 'token_ids': None}], 'service_tier': None, 'system_fingerprint': None, 'usage': {'prompt_tokens': 790, 'total_tokens': 827, 'completion_tokens': 37, 'prompt_tokens_details': None}, 'prompt_logprobs': None, 'prompt_token_ids': None, 'kv_transfer_params': None}
```

Logs of the vllm server:

```bash
(APIServer pid=161632) INFO:     127.0.0.1:58686 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=161632) INFO 01-20 14:42:35 [loggers.py:127] Engine 000: Avg prompt throughput: 79.0 tokens/s, Avg generation throughput: 3.7 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=161632) INFO 01-20 14:42:45 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```
