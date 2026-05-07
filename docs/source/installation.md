# Installation

This document describes how to install vllm-kunlun manually.

## Requirements

- **OS**: Ubuntu 20.04
- **Software**:
  - Python >=3.10
  - PyTorch ≥ 2.5.1
  - vLLM (same version as vllm-kunlun)

## Setup environment using container
We provide clean and minimal base images for your use. Choose the image source
based on your network:

- **Public registry**:
  `wjie520/vllm_kunlun:uv_base`
- **Internal registry (Baidu intranet only)**:
  `iregistry.baidu-int.com/hac_test/aiak-inference-llm:vLLM-Kunlun-Base`

Before pulling the image, you can verify that the public tag exists without
downloading the full image:

```bash
docker manifest inspect wjie520/vllm_kunlun:uv_base >/dev/null
```

If the manifest check succeeds but `docker pull` times out, the image tag is
available and the failure is usually caused by Docker Hub connectivity, proxy, or
registry mirror configuration. If the manifest check also times out, configure
your Docker network access first and retry the check. The internal registry is
only reachable from the Baidu intranet.

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
# or export build_image="iregistry.baidu-int.com/hac_test/aiak-inference-llm:vLLM-Kunlun-Base"

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
### Install vLLM

```{code-block} bash
:substitutions:

uv pip install vllm==|pip_vllm_version| --no-build-isolation --no-deps
```

### Build and Install
Navigate to the vllm-kunlun directory and build the package:

```{code-block} bash
:substitutions:

git clone https://github.com/baidu/vLLM-Kunlun

cd vLLM-Kunlun

git checkout |vllm_kunlun_version|

uv pip install -r requirements.txt

python setup.py build

python setup.py install
```

### Replace eval_frame.py
Copy the eval_frame.py patch:

```
cp vllm_kunlun/patches/eval_frame.py "${CONDA_PREFIX:-$VIRTUAL_ENV}"/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py
```

### Replace quantization __init__.py

```
cp vllm_kunlun/quantization/__init__.py "${CONDA_PREFIX:-$VIRTUAL_ENV}"/lib/python3.10/site-packages/vllm/model_executor/layers/quantization/__init__.py
```

## Choose to download customized xpytorch

### Install the KL3-customized build of PyTorch

```
wget -O xpytorch-cp310-torch251-ubuntu2004-x64.run https://baidu-kunlun-customer.su.bcebos.com/aiak/qwen3_next/20260226/xpytorch-cp310-torch251-ubuntu2004-x64.run
bash xpytorch-cp310-torch251-ubuntu2004-x64.run --noexec --target xpytorch_unpack && cd xpytorch_unpack/ && \
sed -i 's/pip/uv pip/g; s/CONDA_PREFIX/VIRTUAL_ENV/g' setup.sh && bash setup.sh
```

## Applying PyTorch patches

```
python vllm_kunlun/patches/patch_torch251.py
```

## Install Kunlun-related packages

```
# Install kunlun_ops
uv pip install "https://baidu-kunlun-customer.su.bcebos.com/aiak/mimo/20260227/kunlun_ops-0.1.58+ee39020a-cp310-cp310-linux_x86_64.whl"

# Install xspeedgate_ops
uv pip install "https://vllm-ai-models.bj.bcebos.com/aiak_share/20260403/xspeedgate_ops-1.1.0+53992ca-cp310-cp310-linux_x86_64.whl"

# Install cocopod
uv pip install "https://vllm-ai-models.bj.bcebos.com/aiak_share/20260403/cocopod-1.1.0-cp310-cp310-linux_x86_64.whl"
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
      --served-model-name Qwen3-VL-30B-A3B-Instruct

```

::::
:::::
