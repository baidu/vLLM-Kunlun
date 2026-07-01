# Multi XPU (Step-3.5-Flash)

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

- Pull Step-3.5-Flash-W8A8-Dynamic weights and place them under `/home/step3p5/Step-3.5-Flash-W8A8-Dynamic/Step-3.5-Flash-W8A8-Dynamic-mlp-3d`

- Ensure that the field `"quantization_config"` is included in `config.json`. If not, deployment will result in an OOM (Out of Memory) error.

```bash
vim /home/step3p5/Step-3.5-Flash-W8A8-Dynamic/Step-3.5-Flash-W8A8-Dynamic-mlp-3d/config.json
```

```json
"quantization_config": {
    "config_groups": {
      "W8A8": {
        "format": "int-quantized",
        "input_activations": {
          "actorder": null,
          "block_structure": null,
          "dynamic": true,
          "group_size": null,
          "num_bits": 8,
          "observer": null,
          "observer_kwargs": {},
          "scale_dtype": null,
          "strategy": "token",
          "symmetric": true,
          "type": "int",
          "zp_dtype": null
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
          "observer": "memoryless_minmax",
          "observer_kwargs": {},
          "scale_dtype": null,
          "strategy": "channel",
          "symmetric": true,
          "type": "int",
          "zp_dtype": null
        }
      }
    },
    "format": "int-quantized",
    "global_compression_ratio": null,
    "ignore": [
      "lm_head",
      "re:.*share_expert.*",
      "re:.*self_attn.*",
      "re:.*norm.*",
      "re:.*eh_proj.*",
      "re:.*transformer.*",
      "re:.*embed_tokens.*",
      "re:.*moe\\.gate$",
      "re:.*mlp*"
    ],
    "kv_cache_scheme": null,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "sparsity_config": {},
    "transform_config": {},
    "version": "0.14.0.1"
  },
```

### Online Serving on Multi XPU

Start the vLLM server on multi XPU:

```bash
unset XPU_DUMMY_EVENT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XFT_USE_FAST_SWIGLU=1         # 使用快速 swiglu 实现
export XPU_USE_FAST_SWIGLU=1         # 使用 MoE 算子中快速 swiglu 实现
export XMLIR_CUDNN_ENABLED=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_FORCE_USE_XPU_GRAPH=1
export XPU_USE_MOE_SORTED_THRES=128
export VLLM_HOST_IP=127.0.0.1
export XMLIR_ENABLE_MOCK_TORCH_COMPILE=false
export VLLM_USE_V1=1
export USE_ORI_ROPE=1
export KUNLUN_DISABLE_SMALL_MOE=1    # Step-3.5-Flash temporary fix
export XMLIR_DYNAMO_WORKAROUND=1
export LD_LIBRARY_PATH=/home/step3p5/xmlir_runtime_links:$LD_LIBRARY_PATH

python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8356 \
      --model /home/step3p5/Step-3.5-Flash-W8A8-Dynamic/Step-3.5-Flash-W8A8-Dynamic-mlp-3d \
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 131072 \
      --tensor-parallel-size 8 \
      --dtype float16 \
      --max_num_seqs 128 \
      --max_num_batched_tokens 32768 \
      --block-size 128 \
      --no-enable-prefix-caching \
      --enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name Step-3.5-Flash \
      --reasoning-parser step3p5 \
      --enable-auto-tool-choice \
      --tool-call-parser step3p5
```
