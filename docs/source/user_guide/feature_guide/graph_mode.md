# Graph Mode Guide

This guide provides instructions for using Kunlun Graph Mode with vLLM Kunlun. Please note that graph mode is available both on V1 and V0 Engine. All supported models are highly compatible with Kunlun Graph.

## Getting Started

From vLLM-KunLun-0.10.1.1 with V1 Engine, vLLM Kunlun will run models in graph mode by default to keep the same behavior with vLLM. If you hit any issues, please feel free to open an issue on GitHub and fallback to the eager mode temporarily by setting `enforce_eager=True` when initializing the model.

There is a graph mode supported by vLLM Kunlun:

- **KunlunGraph**: This is the default graph mode supported by vLLM Kunlun. In vLLM-KunLun-0.10.1.1, Qwen, GLM and InternVL series models are well tested.

## Using KunlunGraph

KunlunGraph is enabled by default. Take Qwen series models as an example, just set to use V1 Engine(default) is enough.

Offline example:

```python
import os

from vllm import LLM

model = LLM(model="models/Qwen3-8B-Instruct")
outputs = model.generate("Hello, how are you?")
```

Online example:

```shell
vllm serve Qwen3-8B-Instruct
```

## Using KunlunGraph

Enabling Kunlun Graph on the Kunlun platform requires the use of splitting ops.

Online example:

```shell
python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8000 \
      --model /models/Qwen3-8B\
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 1 \
      --dtype float16 \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name Qwen3-8B-Instruct \
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

## Fallback to the Eager Mode

If `KunlunGraph` fail to run, you should fallback to the eager mode.

Online example:

```shell
python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8000 \
      --model /models/Qwen3-8B-Instruct\
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 1 \
      --dtype float16 \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name Qwen3-8B-Instruct \
      --enforce_eager
```
