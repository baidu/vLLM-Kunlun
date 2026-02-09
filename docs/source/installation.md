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

### Replace eval_frame.py
Copy the eval_frame.py patch:
```
cp vllm_kunlun/patches/eval_frame.py /root/miniconda/envs/vllm_kunlun_0.10.1.1/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py
```

## Choose to download customized xpytorch

### Install the KL3-customized build of PyTorch
```
wget -O xpytorch-cp310-torch251-ubuntu2004-x64.run https://baidu-kunlun-public.su.bcebos.com/v1/baidu-kunlun-share/1130/xpytorch-cp310-torch251-ubuntu2004-x64.run?authorization=bce-auth-v1%2FALTAKypXxBzU7gg4Mk4K4c6OYR%2F2025-12-02T05%3A01%3A27Z%2F-1%2Fhost%2Ff3cf499234f82303891aed2bcb0628918e379a21e841a3fac6bd94afef491ff7
(for the conda)
bash xpytorch-cp310-torch251-ubuntu2004-x64.run
(for the uv)
bash xpytorch-cp310-torch251-ubuntu2004-x64.run --noexec --target xpytorch_unpack && cd xpytorch_unpack/ && \
sed -i 's/pip/uv pip/g; s/CONDA_PREFIX/VIRTUAL_ENV/g' setup.sh && bash setup.sh
```
### Install the KL3-customized build of PyTorch (Only MIMO V2)
```
wget -O xpytorch-cp310-torch251-ubuntu2004-x64.run https://klx-sdk-release-public.su.bcebos.com/kunlun2aiak_output/1231/xpytorch-cp310-torch251-ubuntu2004-x64.run
(for the conda)
bash xpytorch-cp310-torch251-ubuntu2004-x64.run
(for the uv)
bash xpytorch-cp310-torch251-ubuntu2004-x64.run --noexec --target xpytorch_unpack && cd xpytorch_unpack/ && \
sed -i 's/pip/uv pip/g; s/CONDA_PREFIX/VIRTUAL_ENV/g' setup.sh && bash setup.sh

```

### Install the KL3-customized build of PyTorch (Only DeepSeek-V3.2-Exp-w8a8)
```
wget -O xpytorch-cp310-torch251-ubuntu2004-x64.run https://aihc-private-hcd.bj.bcebos.com/v1/vllm-kunlun-ds/xpytorch-cp310-torch251-ubuntu2004-x64.run?authorization=bce-auth-v1%2FALTAKvz6x4eqcmSsKjQxq3vZdB%2F2026-02-03T01%3A59%3A40Z%2F-1%2Fhost%2Ffc4b6f5b83c2fde70d48fdfc23c40c396efc9cb3c36d6f811fdca5f109073321
(for the conda)
bash xpytorch-cp310-torch251-ubuntu2004-x64.run
(for the uv)
bash xpytorch-cp310-torch251-ubuntu2004-x64.run --noexec --target xpytorch_unpack && cd xpytorch_unpack/ && \
mv torch_xray-999.9.9-cp310-cp310-linux_x86_64.whl torch_xray-2.0.3-cp310-cp310-linux_x86_64.whl && \
sed -i 's/pip/uv pip/g; s/CONDA_PREFIX/VIRTUAL_ENV/g; s/torch_xray-999.9.9/torch_xray-2.0.3/' setup.sh && bash setup.sh
```
## Choose to download customized ops

### Install custom ops
```
uv pip install "https://baidu-kunlun-public.su.bcebos.com/v1/baidu-kunlun-share/1130/xtorch_ops-0.1.2209%2B6752ad20-cp310-cp310-linux_x86_64.whl?authorization=bce-auth-v1%2FALTAKypXxBzU7gg4Mk4K4c6OYR%2F2025-12-05T06%3A18%3A00Z%2F-1%2Fhost%2F14936c2b7e7c557c1400e4c467c79f7a9217374a7aa4a046711ac4d948f460cd"
```
### Install custom ops (Only MIMO V2)
```
uv pip install "https://vllm-ai-models.bj.bcebos.com/v1/vLLM-Kunlun/ops/swa/xtorch_ops-0.1.2109%252B523cb26d-cp310-cp310-linux_x86_64.whl"
```
### Install custom ops (Only DeepSeek-V3.2-Exp-w8a8)
```
uv pip install "https://klx-sdk-release-public.su.bcebos.com/kunlun2aiak_output/1215/xtorch_ops-0.1.2263%2Bc030eebd-cp310-cp310-linux_x86_64.whl"
```

## Install the KLX3 custom Triton build
```
uv pip install "https://cce-ai-models.bj.bcebos.com/v1/vllm-kunlun-0.11.0/triton-3.0.0%2Bb2cde523-cp310-cp310-linux_x86_64.whl"
```
## Install the AIAK custom ops library
```
uv pip install "https://cce-ai-models.bj.bcebos.com/XSpeedGate-whl/release_merge/20251219_152418/xspeedgate_ops-0.0.0-cp310-cp310-linux_x86_64.whl"
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
