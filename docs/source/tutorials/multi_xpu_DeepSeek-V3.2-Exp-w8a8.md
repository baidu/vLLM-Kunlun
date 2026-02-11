# Multi XPU (DeepSeek-V3.2-Exp-w8a8)

## Run vllm-kunlun on Multi XPU

Setup environment using container:

Please follow the [installation.md](../installation.md) document to set up the environment first.

Create a container
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

export build_image="xxx"

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

### Preparation Weight

- Pull DeepSeek-V3.2-Exp-w8a8-int8 weights
  ```
  wget -O DeepSeek-V3.2-Exp-w8a8-int8.tar.gz https://aihc-private-hcd.bj.bcebos.com/v1/LLM/DeepSeek/DeepSeek-V3.2-Exp-w8a8-int8.tar.gz?authorization=bce-auth-v1%2FALTAKvz6x4eqcmSsKjQxq3vZdB%2F2025-12-24T06%3A07%3A10Z%2F-1%2Fhost%2Fa324bf469176934a05f75d3acabc3c1fb891be150f43fb1976e65b7ec68733db
  ```
- Ensure that the field "quantization_config" is included.If not, deployment will result in an OOM (Out of Memory) error.

vim model/DeepSeek-V3.2-Exp-w8a8-int8/config.json
```config.json
"quantization_config": {
    "config_groups": {
      "group_0": {
        "format": "int-quantized",
        "input_activations": {
          "actorder": null,
          "block_structure": null,
          "dynamic": true,
          "group_size": null,
          "num_bits": 8,
          "observer": null,
          "observer_kwargs": {},
          "strategy": "token",
          "symmetric": true,
          "type": "int"
        },
        "output_activations": null,
        "targets": [
          "Linear"
        ],
        "weights": {
          "actorder": null,
          "block_structure": null,
          "dynamic": false,
          "group_size": null,
          "num_bits": 8,
          "observer": "minmax",
          "observer_kwargs": {},
          "strategy": "channel",
          "symmetric": true,
          "type": "int"
        }
      }
    },
    "format": "int-quantized",
    "global_compression_ratio": null,
    "ignore": [
      "lm_head"
    ],
    "kv_cache_scheme": null,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "sparsity_config": {},
    "transform_config": {},
    "version": "0.12.2"
  },
```

### Online Serving on Multi XPU

Start the vLLM server on multi XPU:

```bash
unset XPU_DUMMY_EVENT && \
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
export XMLIR_CUDNN_ENABLED=1 && \
export XPU_USE_DEFAULT_CTX=1 && \
export XMLIR_FORCE_USE_XPU_GRAPH=1 && \
export XMLIR_ENABLE_FAST_FC=1 && \
export XPU_USE_FAST_SWIGLU=1 && \
export CUDA_GRAPH_OPTIMIZE_STREAM=1 && \
export XMLIR_ENABLE_MOCK_TORCH_COMPILE=false && \
export XPU_USE_MOE_SORTED_THRES=1 && \
export USE_ORI_ROPE=1 && \
export VLLM_USE_V1=1 

python -m vllm.entrypoints.openai.api_server  \
        --host 0.0.0.0 \
        --port 8806   \
        --model /data/DeepSeek-V3.2-Exp-w8a8-int8 \
        --gpu-memory-utilization 0.95  \
        --trust-remote-code     \
        --max-model-len 32768 \
        --tensor-parallel-size 8 \
        --dtype float16      \
        --max_num_seqs 32  \
        --max_num_batched_tokens 8192 \
        --block-size 64 \
        --no-enable-chunked-prefill \
        --distributed-executor-backend mp \
        --disable-log-requests \
        --no-enable-prefix-caching  --kv-cache-dtype bfloat16 \
        --compilation-config '{"splitting_ops":["vllm.unified_attention",
            "vllm.unified_attention_with_output",
            "vllm.unified_attention_with_output_kunlun",
            "vllm.mamba_mixer2", 
            "vllm.mamba_mixer", 
            "vllm.short_conv", 
            "vllm.linear_attention",
            "vllm.plamo2_mamba_mixer",
            "vllm.gdn_attention",
            "vllm.sparse_attn_indexer",
            "vllm.sparse_attn_indexer_vllm_kunlun"]}'
```
