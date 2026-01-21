# Multi XPU (Qwen2.5-VL-32B)

## Run vllm-kunlun on Multi XPU

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

### Offline Inference on Multi XPU

Start the server in a container:

```bash
from vllm import LLM, SamplingParams

def main():

    model_path = "/models/Qwen2.5-VL-32B-Instruct"

    llm_params = {
        "model": model_path,
        "tensor_parallel_size": 2,
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
 你好！我是通义千问，阿里巴巴集团旗下的超大规模语言模型。你可以叫我Qwen。我能够回答问题、创作文字，比如写故事、写公文、写邮件、写剧本、逻辑推理、编程等等，还能表达观点，玩游戏等。有什么我可以帮助你的吗？ 😊
==================================================
```

### Online Serving on Multi XPU

Start the vLLM server on a multi XPU:

```bash
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 9988 \
    --model /models/Qwen2.5-VL-32B-Instruct \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --max_num_seqs 128 \
    --max_num_batched_tokens 32768 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --distributed-executor-backend mp \
    --served-model-name Qwen2.5-VL-32B-Instruct \
    --compilation-config '{"splitting_ops": ["vllm.unified_attention", 
                                                "vllm.unified_attention_with_output",
                                                "vllm.unified_attention_with_output_kunlun",
                                                "vllm.mamba_mixer2",
                                                "vllm.mamba_mixer", 
                                                "vllm.short_conv", 
                                                "vllm.linear_attention", 
                                                "vllm.plamo2_mamba_mixer", 
                                                "vllm.gdn_attention", 
                                                "vllm.sparse_attn_indexer"]}'
                                                #Version 0.11.0 
```

If your service start successfully, you can see the info shown below:

```bash
(APIServer pid=110552) INFO:     Started server process [110552]
(APIServer pid=110552) INFO:     Waiting for application startup.
(APIServer pid=110552) INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:9988/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2.5-VL-32B-Instruct",
        "prompt": "你好！你是谁?",
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"cmpl-9784668ac5bc4b4e975d0aa5ee8377c6","object":"text_completion","created":1768898088,"model":"Qwen2.5-VL-32B-Instruct","choices":[{"index":0,"text":" 你好！我是通义千问，阿里巴巴集团旗下的超大规模语言模型。你可以回答问题、创作文字，如写故事、公文、邮件、剧本等，还能表达\n","logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":5,"total_tokens":45,"completion_tokens":40,"prompt_tokens_details":null},"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
(APIServer pid=110552) INFO 01-20 16:34:48 [loggers.py:127] Engine 000: Avg prompt throughput: 0.5 tokens/s, Avg generation throughput: 0.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=110552) INFO:     127.0.0.1:17988 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=110552) INFO 01-20 16:34:58 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3.4 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=110552) INFO 01-20 16:35:08 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

Input an image for testing.Here,a python script is used:

```python
import requests
import base64

API_URL = "http://localhost:9988/v1/chat/completions"
MODEL_NAME = "Qwen2.5-VL-32B-Instruct"
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
{'id': 'chatcmpl-9857119aed664a3e8f078efd90defdca', 'object': 'chat.completion', 'created': 1768898198, 'model': 'Qwen2.5-VL-32B-Instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '你好！这张图片展示了一个标志，内容如下：\n\n1. **左侧图标**：\n   - 一个黄色的圆形笑脸表情符号。\n   - 笑脸的表情非常开心，眼睛眯成弯弯的形状，嘴巴张开露出牙齿，显得非常愉快。\n   - 笑脸的双手在胸前做出拥抱的动作，手掌朝外，象征着“拥抱”或“友好的姿态”。\n\n2. **右侧文字**：\n   - 文字是英文单词：“Hugging Face”。\n   - 字体为黑色，字体风格简洁、现代，看起来像是无衬线字体（sans-serif）。\n\n3. **整体设计**：\n   - 整个标志的设计非常简洁明了，颜色对比鲜明（黄色笑脸和黑色文字），背景为纯白色，给人一种干净、友好的感觉。\n   - 笑脸和文字之间的间距适中，布局平衡。\n\n这个标志可能属于某个品牌或组织，名字为“Hugging Face”，从设计来看，它传达了一种友好、开放和积极的形象。', 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'tool_calls': [], 'reasoning_content': None}, 'logprobs': None, 'finish_reason': 'stop', 'stop_reason': None, 'token_ids': None}], 'service_tier': None, 'system_fingerprint': None, 'usage': {'prompt_tokens': 95, 'total_tokens': 311, 'completion_tokens': 216, 'prompt_tokens_details': None}, 'prompt_logprobs': None, 'prompt_token_ids': None, 'kv_transfer_params': None}
```

Logs of the vllm server:

```bash
(APIServer pid=110552) INFO:     127.0.0.1:19378 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=110552) INFO 01-20 16:36:49 [loggers.py:127] Engine 000: Avg prompt throughput: 9.5 tokens/s, Avg generation throughput: 21.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=110552) INFO 01-20 16:36:59 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```