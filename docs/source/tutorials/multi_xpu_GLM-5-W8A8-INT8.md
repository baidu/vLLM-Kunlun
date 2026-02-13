# Multi XPU (GLM-5-W8A8-INT8)

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

- Pull GLM-5-W8A8-INT8 weights
  ```
  wget -O GLM-5-W8A8-INT8-Dynamic.tar.gz https://aihc-private-hcd.bj.bcebos.com/LLM/AICapX-Quant-Models/GLM-5-W8A8-INT8-Dynamic.tar.gz
  ```

### Online Serving on Multi XPU

Start the vLLM server on multi XPU:

```bash
unset XPU_DUMMY_EVENT && \
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
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
        --model GLM-5-W8A8-INT8-Dynamic \
        --gpu-memory-utilization 0.97  \
        --trust-remote-code     \
        --max-model-len 32768 \
        --tensor-parallel-size 8 \
        --dtype bfloat16 \
        --max_num_seqs 8  \
        --max_num_batched_tokens 8192 \
        --block-size 64 \
        --no-enable-chunked-prefill \
        --distributed-executor-backend mp \
        --disable-log-requests \
        --no-enable-prefix-caching \
        --kv-cache-dtype bfloat16 \
        --compilation-config '{
            "splitting_ops":[
                "vllm.unified_attention",
                "vllm.unified_attention_with_output",
                "vllm.unified_attention_with_output_kunlun",
                "vllm.mamba_mixer2", 
                "vllm.mamba_mixer", 
                "vllm.short_conv", 
                "vllm.linear_attention",
                "vllm.plamo2_mamba_mixer",
                "vllm.gdn_attention",
                "vllm.sparse_attn_indexer",
                "vllm.sparse_attn_indexer_vllm_kunlun"
            ]}'
```
