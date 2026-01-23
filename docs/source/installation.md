# Installation

This document describes how to install vllm-kunlun manually.

## Requirements

- **OS**: Ubuntu 22.04 
- **Software**:
  - Python >=3.10
  - PyTorch ≥ 2.5.1
  - vLLM (same version as vllm-kunlun)

## Setup environment using container
We provide a clean, minimal base image for your use`wjie520/vllm_kunlun:base_v0.0.2` and `wjie520/vllm_kunlun:base_mimo_v0.0.2`(Only MIMO_V2 and GPT-OSS).You can pull it using the `docker pull` command.
### Container startup script

:::::{tab-set}
:sync-group: install

::::{tab-item} start_docker.sh
:selected:
:sync: pip
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
export build_image="wjie520/vllm_kunlun:base_v0.0.2"
# or export build_image="iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.32"

docker run -itd ${DOCKER_DEVICE_CONFIG} \
    --net=host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --tmpfs /dev/shm:rw,nosuid,nodev,exec,size=32g \
    --cap-add=SYS_PTRACE \
    -v /home/users/vllm-kunlun:/home/vllm-kunlun \
    --name "$1" \
    -w /workspace \
    "$build_image" /bin/bash
```
::::
:::::
## Install vLLM-kunlun
### Install vLLM 0.11.0
```
uv pip install vllm==0.11.0 --no-build-isolation --no-deps
```
### Build and Install
Navigate to the vllm-kunlun directory and build the package:
```
git clone https://github.com/baidu/vLLM-Kunlun

cd vLLM-Kunlun

uv pip install -r requirements.txt

python setup.py build

python setup.py install

```
## Quick Start

### Set up the environment

```
chmod +x /workspace/vLLM-Kunlun/setup_env.sh && source /workspace/vLLM-Kunlun/setup_env.sh
```

### Run the server
:::::{tab-set}
:sync-group: install

::::{tab-item} start_service.sh
:selected:
:sync: pip
```{code-block} bash
   :substitutions:
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
      --no-enable-prefix-caching \
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
                                                "vllm.sparse_attn_indexer"]}'

```
::::
:::::
