# Multi XPU (Step-3.5-Flash)

## Run vllm-kunlun0.15.1-dev on Multi XPU

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
# export system variable
# unset XPU_DUMMY_EVENT
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export XFT_USE_FAST_SWIGLU=1 #使用快速swiglu实现
# export XPU_USE_FAST_SWIGLU=1 #使用moe算子中快速swiglu实现
# export XMLIR_CUDNN_ENABLED=1
# export XPU_USE_DEFAULT_CTX=1
# export XMLIR_FORCE_USE_XPU_GRAPH=1
# export XPU_USE_MOE_SORTED_THRES=128 
# export VLLM_HOST_IP=127.0.0.1
# export XMLIR_ENABLE_MOCK_TORCH_COMPILE=false 
# export VLLM_USE_V1=1
# export USE_ORI_ROPE=1
# export KUNLUN_DISABLE_SMALL_MOE=1 #step-3.5-flash temporary fix

# python /workspace/offline.py

from vllm import LLM, SamplingParams

llm = LLM(
    model="/models/Step-3.5-Flash",
    tensor_parallel_size=8,
    dtype="bfloat16",
    max_model_len=32768,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    distributed_executor_backend="mp",
    block_size=128,
    max_num_seqs=128,
    max_num_batched_tokens=32768,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=10,
    max_tokens=512,
    stop=["<|end|>", "</s>"]
)

prompt = """
<|user|>
你好，请介绍一下你自己
<|assistant|>
"""

outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

:::::
If you run this script successfully, you can see the info shown below:

```bash
==================================================
你好！我是 **Step**，由 **阶跃星辰（StepFun）** 开发的多模态大语言模型。  
我具备自然语言理解与生成、图像分析、视觉推理、数理逻辑、知识问答等多种能力。不仅能理解和处理文字信息，还能结合图片进行多模态推理与分析。

我的核心原则是：诚实可靠、有用友善、尊重隐私、促进积极交流、保持客观中立、拒绝有害内容。  
简单来说，我的目标是为你提供准确、有帮助、温暖的智能支持。

如果你愿意，可以告诉我你的兴趣或需求，我会尽力帮你实现目标 😊  
你想先了解我在哪些方面能帮到你吗？
==================================================
```

### Online Serving on Multi XPU

Start the vLLM server on a multi XPU:

```bash
unset XPU_DUMMY_EVENT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XFT_USE_FAST_SWIGLU=1 #使用快速swiglu实现
export XPU_USE_FAST_SWIGLU=1 #使用moe算子中快速swiglu实现
export XMLIR_CUDNN_ENABLED=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_FORCE_USE_XPU_GRAPH=1
export XPU_USE_MOE_SORTED_THRES=128 
export VLLM_HOST_IP=127.0.0.1
export XMLIR_ENABLE_MOCK_TORCH_COMPILE=false 
export VLLM_USE_V1=1
export USE_ORI_ROPE=1
export KUNLUN_DISABLE_SMALL_MOE=1 #step-3.5-flash temporary fix

python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8356 \
      --model /models/Step-3.5-Flash \
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 8 \
      --dtype bfloat16 \
      --max_num_seqs 128 \
      --max_num_batched_tokens 32768 \
      --block-size 128 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name Step-3.5-Flash \
      --reasoning-parser step3p5 \
      --enable-auto-tool-choice \
      --tool-call-parser step3p5 \
```

If your service start successfully, you can see the info shown below:

```bash
(APIServer pid=133800) INFO:     Started server process [133800]
(APIServer pid=133800) INFO:     Waiting for application startup.
(APIServer pid=133800) INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://127.0.0.1:8356/v1/chat/completions   
    -H "Content-Type: application/json"   
    -d '{
    "model": "Step-3.5-Flash",
    "messages": [
      {"role": "user", "content": "你好，简单介绍一下你自己"}
    ],
    "max_tokens":200,
    "temperature": 0.7
  }'
```

Or use a Python script

```python
import requests
import json
import re

URL = "http://127.0.0.1:8356/v1/chat/completions"

payload = {
    "model": "Step-3.5-Flash",
    "messages": [
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "max_tokens": 500,
    "top_p": 0.8,
    "top_k": 10,
    "temperature": 0.7,
    # "presence_penalty": 0.3,
    # "repetition_penalty": 1.05,
    # At present, the model’s responses occasionally suffer from accuracy issues; you may wish to try adjusting the sampling parameters.
    
}

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer EMPTY"
}

resp = requests.post(URL, headers=headers, json=payload)
data = resp.json()

choice = data["choices"][0]
content = choice["message"]["content"]

answer = content

print("\n===== ANSWER =====\n")
print(answer)
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-93112d4d8e047a9c","object":"chat.completion","created":1776166074,"model":"Step-3.5-Flash","choices":[{"index":0,"message":{"role":"assistant","content":"你好！我是 **Step**，由 **阶跃星辰（StepFun）** 开发的多\n","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":"好的，用户让我简单介绍一下自己。首先我得明确身份，我是Step，由阶跃星辰（StepFun）开发。用户可能刚接触我，需要基础信息，比如功能、特点以及使用原则。\n\n然后考虑用户的需求场景，可能是第一次使用AI助手，或者想比较不同的AI。需要突出我的多模态能力，比如处理文字和图片，还有逻辑推理、知识问答这些核心功能。同时要强调中文","reasoning_content":"好的，用户让我简单介绍一下自己。首先我得明确身份，我是Step，由阶跃星辰（StepFun）开发。用户可能刚接触我，需要基础信息，比如功能、特点以及使用原则。\n\n然后考虑用户的需求场景，可能是第一次使用AI助手，或者想比较不同的AI。需要突出我的多模态能力，比如处理文字和图片，还有逻辑推理、知识问答这些核心功能。同时要强调中文"},"logprobs":null,"finish_reason":"stop","stop_reason":1,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":17,"total_tokens":131,"completion_tokens":114,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}

# python script
===== ANSWER =====

你好！我是 **Step**，由 **阶跃星辰（StepFun）** 开发的大语言模型。  

我具备以下主要能力和特点：  
- 🧠 **自然语言理解与生成**：能够流畅地进行多轮对话、写作、总结、翻译等；  
- 👁️ **多模态推理**：不仅能处理文字，还能理解和分析图片内容，进行视觉推理；  
- 📚 **知识问答与逻辑推理**：擅长基于事实回答问题，并解决数学、逻辑类任务；  
- 💡 **创意表达**：可辅助创作故事、诗歌、策划方案等富有创意的内容；  
- 🌍 **多语言支持**：能用多种语言与用户交流；  
- 🤝 **安全可靠**：遵循诚实、友善、尊重隐私的原则，保持客观中立。  

我目前是 **完全免费使用** 的，不收集或存储你的个人隐私信息。你可以随时向我提问、讨论、创作或探索各种主题～  

你想先了解我在哪方面最擅长吗？
```

Logs of the vllm server:

```bash
(APIServer pid=182858) INFO 04-14 19:45:26 [loggers.py:257] Engine 000: Avg prompt throughput: 1.7 tokens/s, Avg generation throughput: 19.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=182858) INFO:     127.0.0.1:12670 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=182858) INFO 04-14 19:45:36 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 24.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=182858) INFO 04-14 19:45:46 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```
