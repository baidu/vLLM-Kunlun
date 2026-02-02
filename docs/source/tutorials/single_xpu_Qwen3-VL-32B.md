# Single XPU (Qwen3-VL-32B)

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

    model_path = "/models/Qwen3-VL-32B"

    llm_params = {
        "model": model_path,
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "dtype": "float16",
        "enable_chunked_prefill": False,
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
                    "text": "tell a joke"
                }
            ]
        }
    ]

    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=1.0,
        top_k=50,
        top_p=1.0
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
 Why don’t skeletons fight each other?
Because they don’t have the guts! 🦴😄
==================================================
```

### Online Serving on Single XPU

Start the vLLM server on a single XPU:

```bash
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 9988 \
    --model /models/Qwen3-VL-32B \
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
    --served-model-name Qwen3-VL-32B \
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
(APIServer pid=109442) INFO:     Started server process [109442]
(APIServer pid=109442) INFO:     Waiting for application startup.
(APIServer pid=109442) INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:9988/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-VL-32B",
        "prompt": "你好！你是谁?",
        "max_tokens": 100,
        "temperature": 0
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"cmpl-4f61fe821ff34f23a91baade5de5103e","object":"text_completion","created":1768876583,"model":"Qwen3-VL-32B","choices":[{"index":0,"text":" 你好！我是通义千问，是阿里云研发的超大规模语言模型。我能够回答问题、创作文字、编程等，还能根据你的需求进行多轮对话。有什么我可以帮你的吗？😊\n\n（温馨提示：我是一个AI助手，虽然我尽力提供准确和有用的信息，但请记得在做重要决策时，最好结合专业意见或进一步核实信息哦！）","logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":5,"total_tokens":90,"completion_tokens":85,"prompt_tokens_details":null},"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
(APIServer pid=109442) INFO:     127.0.0.1:19962 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=109442) INFO 01-20 10:36:28 [loggers.py:127] Engine 000: Avg prompt throughput: 0.5 tokens/s, Avg generation throughput: 8.5 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=109442) INFO 01-20 10:36:38 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=109442) INFO 01-20 10:43:23 [chat_utils.py:560] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
(APIServer pid=109442) INFO 01-20 10:43:28 [loggers.py:127] Engine 000: Avg prompt throughput: 9.0 tokens/s, Avg generation throughput: 6.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.5%, Prefix cache hit rate: 0.0%
```

Input an image for testing.Here,a python script is used:

```python
import requests
import base64
API_URL = "http://localhost:9988/v1/chat/completions"
MODEL_NAME = "Qwen3-VL-32B"
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
    "temperature": 0.1
}
response = requests.post(API_URL, json=payload)
print(response.json())
```

If you query the server successfully, you can see the info shown below (client):

```bash
{'id': 'chatcmpl-4b42fe46f2c84991b0af5d5e1ffad9ba', 'object': 'chat.completion', 'created': 1768877003, 'model': 'Qwen3-VL-32B', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '你好！这张图片展示的是“Hugging Face”的标志。\n\n图片左侧是一个黄色的圆形表情符号（emoji），它有着圆圆的眼睛、张开的嘴巴露出微笑，双手合拢在脸颊两侧，做出一个拥抱或欢迎的姿态，整体传达出友好、温暖和亲切的感觉。\n\n图片右侧是黑色的英文文字“Hugging Face”，字体简洁现代，与左侧的表情符号相呼应。\n\n整个标志设计简洁明了，背景为纯白色，突出了标志本身。这个标志属于Hugging Face公司，它是一家知名的开源人工智能公司，尤其在自然语言处理（NLP）领域以提供预训练模型（如Transformers库）和模型托管平台而闻名。\n\n整体来看，这个标志通过可爱的表情符号和直白的文字，成功传达了公司“拥抱”技术、开放共享、友好的品牌理念。', 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'tool_calls': [], 'reasoning_content': None}, 'logprobs': None, 'finish_reason': 'stop', 'stop_reason': None, 'token_ids': None}], 'service_tier': None, 'system_fingerprint': None, 'usage': {'prompt_tokens': 90, 'total_tokens': 266, 'completion_tokens': 176, 'prompt_tokens_details': None}, 'prompt_logprobs': None, 'prompt_token_ids': None, 'kv_transfer_params': None}
```

Logs of the vllm server:

```bash
(APIServer pid=109442) INFO:     127.0.0.1:26854 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=109442) INFO 01-20 10:43:38 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 10.7 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=109442) INFO 01-20 10:43:48 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```
