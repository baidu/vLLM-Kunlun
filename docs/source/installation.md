# Installation

This document describes how to install vllm-kunlun manually.

## Requirements

- **OS**: Ubuntu 20.04
- **Software**:
  - Python >=3.10
  - PyTorch ≥ 2.9.0
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
### Install PyTorch

```{code-block} bash
uv pip install torch==2.9.0 torchvision torchaudio
```

### Install vLLM

```{code-block} bash
:substitutions:

uv pip install vllm==|pip_vllm_version| --no-build-isolation --no-deps
```

### Build and Install
Navigate to the vllm-kunlun directory and install the package:

```{code-block} bash
:substitutions:

git clone https://github.com/baidu/vLLM-Kunlun

cd vLLM-Kunlun

git checkout |vllm_kunlun_version|

uv pip install -r requirements.txt
uv pip install --no-build-isolation --no-deps .
```

## Choose to download customized xpytorch

### Install the KL3-customized build of PyTorch

```
wget -O xpytorch-cp310-torch290-ubuntu2004-x64.run https://klx-sdk-release-public.su.bcebos.com/kunlun2jituan/20260806/xpytorch-cp310-torch290-ubuntu2004-x64.run
bash xpytorch-cp310-torch290-ubuntu2004-x64.run --noexec --target xpytorch_unpack && cd xpytorch_unpack/ && \
sed -i 's/pip/uv pip/g; s/CONDA_PREFIX/VIRTUAL_ENV/g' setup.sh && bash setup.sh
```

## Install Kunlun-related packages

```
# Install kunlun_ops
uv pip install "https://klx-sdk-release-public.su.bcebos.com/kunlun2jituan/20260806/kunlun_ops-0.1.227%2B2b100f96-cp310-cp310-linux_x86_64.whl"

# Install xspeedgate_ops (1.5.0 is the minimum: it provides weak_ref_tensor)
uv pip install "https://vllm-ai-models.bj.bcebos.com/aiak_share/20260827/torch29/xspeedgate_ops-1.5.1%2B87067b3.torch29-cp310-cp310-linux_x86_64.whl"

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
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 1 \
      --dtype float16 \
      --served-model-name Qwen3-VL-30B-A3B-Instruct
```

::::
:::::
