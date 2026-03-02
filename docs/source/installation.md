# Installation

This document describes how to install vllm-kunlun manually.

## Requirements

- **OS**: Ubuntu 22.04 
- **Software**:
  - Python >=3.10
  - PyTorch ≥ 2.5.1
  - vLLM (same version as vllm-kunlun)

## Setup environment using container

We provide a clean, minimal base image for your use`wjie520/vllm_kunlun:uv_base`.You can pull it using the `docker pull wjie520/vllm_kunlun:uv_base` command.

We also provide images with xpytorch and ops installed.You can pull it using the `wjie520/vllm_kunlun:base_v0.0.2 and wjie520/vllm_kunlun:base_mimo_v0.0.2 (Only MIMO_V2 and GPT-OSS)` command

### Container startup script

:::::{tab-set}
:sync-group: install

::::{tab-item} start_docker.sh
:selected:
:sync: uv pip

```{code-block} bash
#!/bin/bash
XPU_NUM=8
DOCKER_DEVICE_CONFIG=""
if [ $XPU_NUM -gt 0 ]; then
    for idx in $(seq 0 $((XPU_NUM-1))); do
        DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpu${idx}:/dev/xpu${idx}"
    done
    DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpuctrl:/dev/xpuctrl"
fi
export build_image="wjie520/vllm_kunlun:uv_base"
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

```bash
uv pip install vllm==0.11.0 --no-build-isolation --no-deps
```

### Build and Install

Navigate to the vllm-kunlun directory and build the package:

```bash
git clone https://github.com/baidu/vLLM-Kunlun

cd vLLM-Kunlun

uv pip install -r requirements.txt

python setup.py build

python setup.py install
```

### Replace eval_frame.py

Copy the eval_frame.py patch:

```bash
cp vllm_kunlun/patches/eval_frame.py /root/miniconda/envs/vllm_kunlun_0.10.1.1/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py
```

## Choose to download customized xpytorch

### Install the KL3-customized build of PyTorch

```bash
wget -O xpytorch-cp310-torch251-ubuntu2004-x64.run https://baidu-kunlun-customer.su.bcebos.com/aiak/qwen3_next/20260226/xpytorch-cp310-torch251-ubuntu2004-x64.run

#for conda
bash xpytorch-cp310-torch251-ubuntu2004-x64.run

#for uv
bash xpytorch-cp310-torch251-ubuntu2004-x64.run --noexec --target xpytorch_unpack && cd xpytorch_unpack/ && \
sed -i 's/pip/uv pip/g; s/CONDA_PREFIX/VIRTUAL_ENV/g' setup.sh && bash setup.sh
```

## Choose to download customized ops

### Install custom ops

```bash
uv pip install "https://baidu-kunlun-customer.su.bcebos.com/aiak/mimo/20260227/kunlun_ops-0.1.58+ee39020a-cp310-cp310-linux_x86_64.whl"
```

### Install the KLX3 custom Triton build

```bash
uv pip install "https://cce-ai-models.bj.bcebos.com/v1/vllm-kunlun-0.11.0/triton-3.0.0%2Bb2cde523-cp310-cp310-linux_x86_64.whl"
```

### Install the AIAK custom ops library

```bash
uv pip install "http://vllm-ai-models.bj.bcebos.com/XSpeedGate-whl/release_merge/20260228_173659/xspeedgate_ops-1.0.0+04b2a8c-cp310-cp310-linux_x86_64.whl" --force-reinstall
```

### Install the Pod custom ops library

```bash
uv pip install "https://vllm-ai-models.bj.bcebos.com/link/20260228_163304/cocopod-1.0.0-cp310-cp310-linux_x86_64.whl"
```

## Latest ops list

You can follow this document for updates to get the latest information.
[vLLM-Kunlun Ops Update List](https://docs.google.com/document/d/1r_Eos0UvBqHBYmoVa4nwIK0oKQH35Tde8oGowvX2znc/edit?usp=sharing)

## Quick Start

### Set up the environment

```bash
chmod +x /workspace/vLLM-Kunlun/setup_env.sh && source /workspace/vLLM-Kunlun/setup_env.sh
```

### Run the server

:::::{tab-set}
:sync-group: install

::::{tab-item} start_service.sh
:selected:
:sync: pip

```{code-block} bash
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