# LoRA Adapters Guide

## Overview

Like vLLM, vllm_kunlun supports LoRA as well. The usage and more details can be found in [vLLM official document](https://docs.vllm.ai/en/latest/features/lora.html).

You can refer to [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models)to find which models support LoRA in vLLM.

Currently, only vLLM v0 mode (including eager and CUDA Graph modes) supports multi-LoRA inference in vllm_kunlun.

## Example

We provide a simple LoRA example here:

```bash
export ENABLE_KUNLUN_LARGE_OPS=0

USE_ORI_ROPE=0 VLLM_USE_V1=0 vllm serve qwen3-8b \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules lora1=/path/to/lora1 lora2=/path/to/lora2
```

## Custom LoRA Operators

We have implemented LoRA-related custom operators for Kunlun hardware, such as `bgmv_shrink`, `bgmv_expand`, `sgmv_shrink`, and `sgmv_expand`. The implementation can be found in `vllm_kunlun/lora/ops/kunlun_ops/lora_ops.py`.
