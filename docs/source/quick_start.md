# Quickstart

## Prerequisites

### Supported Devices

- Kunlun3 P800

## Setup environment using container

:::::{tab-set}
::::{tab-item} Ubuntu

```{code-block} bash
   :substitutions:
#!/bin/bash
XPU_NUM=8
DOCKER_DEVICE_CONFIG=""
if [ $XPU_NUM -gt 0 ]; then
    for idx in $(seq 0 $((XPU_NUM-1))); do
        DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpu${idx}:/dev/xpu${idx}"
    done
    DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpuctrl:/dev/xpuctrl"
fi
export build_image="xxxxx"
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

::::
:::::

Start docker:

```bash
#start
bash ./rundocker.sh <container_name>
#Enter container
docker exec -it <container_name> bash
```

The default working directory is `/workspace`. With the fully provisioned environment image we provide, you can quickly start developing and running tasks within this directory.

## Set up system environment

```
#Set environment
chmod +x /workspace/vllm-kunlun/setup_env.sh && source /workspace/vllm-kunlun/setup_env.sh
```

## Usage

You can start the service quickly using the script below.

:::::{tab-set}
::::{tab-item} Offline Batched Inference

With vLLM installed, you can start generating texts for list of input prompts (i.e. offline batch inferencing).

Try to run below Python script directly or use `python3` shell to generate texts:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```python
import os
from vllm import LLM, SamplingParams

def main():

    model_path = "models/Qwen3-VL-30B-A3B-Instruct"

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="float16",
        distributed_executor_backend="mp",
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        block_size=128,
        max_num_seqs=128,
        max_num_batched_tokens=32768,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        served_model_name="Qwen3-VL",
        compilation_config={
            "splitting_ops": [
                "vllm.unified_attention",
                "vllm.unified_attention_with_output",
                "vllm.unified_attention_with_output_kunlun",
                "vllm.mamba_mixer2",
                "vllm.mamba_mixer",
                "vllm.short_conv",
                "vllm.linear_attention",
                "vllm.plamo2_mamba_mixer",
                "vllm.gdn_attention",
                "vllm.sparse_attn_indexer",
            ]
        },
    )

    # === test chat ===
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello, what can you do?"}]
        }
    ]

    sampling = SamplingParams(
        max_tokens=200,
        temperature=0.8,
        top_k=50,
        top_p=1.0,
    )

    print("开始推理...")
    outputs = llm.chat(messages, sampling_params=sampling)

    print("模型输出：\n")
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()

```

::::

::::{tab-item} OpenAI Completions API

vLLM can also be deployed as a server that implements the OpenAI API protocol. Run
the following command to start the vLLM server with the
[Qwen3-VL-30B-A3B-Instruct]model:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8356 \
      --model models/Qwen3-VL-30B-A3B-Instruct \
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 1 \
      --dtype float16 \
      --max_num_seqs 128 \
      --max_num_batched_tokens 32768 \
      --block-size 128 \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name Qwen3-VL-30B-A3B-Instruct \
      --compilation-config '{"splitting_ops": ["vllm.unified_attention",
                                                "vllm.unified_attention_with_output",
                                                "vllm.unified_attention_with_output_kunlun",
                                                "vllm.mamba_mixer2",
                                                "vllm.mamba_mixer",
                                                "vllm.short_conv",
                                                "vllm.linear_attention",
                                                "vllm.plamo2_mamba_mixer",
                                                "vllm.gdn_attention",
                                                "vllm.sparse_attn_indexer"]}' \
```

If you see a log as below:

```bash
(APIServer pid=51171) INFO:     Started server process [51171]
(APIServer pid=51171) INFO:     Waiting for application startup.
(APIServer pid=51171) INFO:     Application startup complete.
(Press CTRL+C to quit)
```

Congratulations, you have successfully started the vLLM server!

You can query the model with input prompts:

```bash
curl http://localhost:8356/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen3-VL",
        "messages": [
          {"role": "user", "content": "What is your name?"}
        ],
        "max_tokens": 200,
        "temperature": 0
      }'

```

vLLM is serving as a background process, you can use `kill -2 $VLLM_PID` to stop the background process gracefully, which is similar to `Ctrl-C` for stopping the foreground vLLM process:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
  VLLM_PID=$(pgrep -f "vllm serve")
  kill -2 "$VLLM_PID"
```

The output is as below:

```
INFO:     Shutting down FastAPI HTTP server.
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
```

Finally, you can exit the container by using `ctrl-D`.
::::
:::::
