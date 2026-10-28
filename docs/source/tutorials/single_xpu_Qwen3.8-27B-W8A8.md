# Single XPU (Qwen3.8-27B-W8A8)

Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`) is a hybrid-backbone model:
64 decoder layers where every 4th layer is full attention (GQA, 24 query / 4 KV
heads, head_dim 256) and the rest are GatedDeltaNet linear attention (16 KV /
48 value heads, head_dim 128). The checkpoint ships with a vision tower and an
MTP shard (`model-mtp.safetensors`), and the language model is quantized with
compressed-tensors W8A8 (int-quantized, dynamic per-token activations). The
quantization config is auto-detected from `config.json`, so no extra
quantization flag is needed.

Verified on a Kunlunxin P800 (96 GiB HBM) with a single XPU (TP=1),
vLLM-Kunlun 0.25.1: service ready, multimodal warmup passed, CUDA graphs
captured (piecewise + FULL), and a top-1/top-5 next-token differential against
an independent CPU reference agreed on all cases.

## Run vllm-kunlun on Single XPU

Setup environment using container:

Please follow the [installation.md](../installation.md) document to set up the environment first.

Create a container

```bash
#!/bin/bash
# rundocker.sh
XPU_NUM=1
DOCKER_DEVICE_CONFIG=""
if [ $XPU_NUM -gt 0 ]; then
    for idx in $(seq 0 $((XPU_NUM-1))); do
        DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpu${idx}:/dev/xpu${idx}"
    done
    DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpuctrl:/dev/xpuctrl"
fi

export build_image="<your-kunlun-vllm-image>"

docker run -itd ${DOCKER_DEVICE_CONFIG} \
    --net=host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --tmpfs /dev/shm:rw,nosuid,nodev,exec,size=32g \
    -v /home/users/vllm-kunlun:/home/vllm-kunlun \
    -v /usr/local/bin/xpu-smi:/usr/local/bin/xpu-smi \
    --name "$1" \
    -w /workspace \
    "$build_image" /bin/bash
```

### Preparation of Model Weights

- Download the Qwen3.8-27B-W8A8-INT8-Dynamic weights (HuggingFace format,
  15 shards + `model-mtp.safetensors`, about 30 GiB)

The `model-mtp.safetensors` shard holds the MTP draft weights. They are not
parameters of `Qwen3_5ForConditionalGeneration`, so exclude the shard from
weight loading with `--ignore-patterns`; otherwise the loader rejects the
unexpected tensors.

### Start the Server on a Single XPU

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8390 \
    --model /models/Qwen3.8-27B-W8A8-INT8-Dynamic \
    --served-model-name Qwen3.8-27B-Int8 \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --max-model-len 32768 \
    --block-size 16 \
    --gpu-memory-utilization 0.9 \
    --ignore-patterns "model-mtp.safetensors"
```

Parameter notes:

- `--dtype float16`: the checkpoint itself declares `dtype: float16`.
- `--max-model-len 32768`: the checkpoint supports 262144 positions; 32768 is
  what the verification run used. Raise it if your KV budget allows.
- `--ignore-patterns "model-mtp.safetensors"`: skip the MTP draft shard.
- Optional: add `--reasoning-parser qwen3 --tool-call-parser hermes` to split
  reasoning content and parse tool calls (both verified against this model).

### Offline Inference on Single XPU

```{code-block} python
from vllm import LLM, SamplingParams

def main():
    model_path = "/models/Qwen3.8-27B-W8A8-INT8-Dynamic"

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="float16",
        max_model_len=32768,
        ignore_patterns=["model-mtp.safetensors"],
    )

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

    sampling_params = SamplingParams(max_tokens=200, temperature=1.0)

    outputs = llm.chat(messages, sampling_params=sampling_params)
    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    main()
```

### Multi XPU Reference

The 24 full-attention query heads and 48 linear-attention value heads are both
divisible by 8, so an 8-XPU deployment is expected to work with:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3.8-27B-W8A8-INT8-Dynamic \
    --tensor-parallel-size 8 \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --ignore-patterns "model-mtp.safetensors"
```

TP=8 has not been verified yet; the numbers in this document all come from the
single-XPU run above.

### Verified Memory Budget (single XPU, TP=1)

Reconciled against `xpu-smi` on the P800:

| Category | MiB | Source |
| --- | --- | --- |
| Model weights | 29194 | server log |
| KV pool (754392 tokens) | 50801 | server log |
| Graph capture | 123 | server log |
| Driver / runtime / allocator | 8868 | xpu-smi remainder |
| Free | 9318 | xpu-smi |
