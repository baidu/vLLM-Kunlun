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
export build_image="quay.io/kunlun/vllm_kunlun:v0.10.1.1rc1"
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
chmod +x /workspace/vLLM-Kunlun/setup_env.sh && source /workspace/vLLM-Kunlun/setup_env.sh
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
                    "text": "What is your name?"
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

::::

::::{tab-item} OpenAI Completions API

vLLM can also be deployed as a server that implements the OpenAI API protocol. Run
the following command to start the vLLM server with the
[Qwen3-8B]model:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8356 \
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
curl http://localhost:8356/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen3-8B",
        "prompt": "What is your name?",
        "max_tokens": 7,
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
