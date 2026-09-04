# Multi XPU (MiniMax-2.5)

## Run vLLM-Kunlun on Multi XPU

Set up the container and install vLLM-Kunlun by following the [installation guide](../installation.md). The project requirements pin `compressed-tensors==0.17.0` and `openai==2.54.0`, which are required for this MiniMax-2.5 W8A8 deployment.

```bash
uv pip install -r requirements.txt
```

## Online Serving on Multi XPU

Set the environment variables below and launch the OpenAI-compatible API server. Adjust the model path and device list to match the deployment host.

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export XPU_USE_MOE_SORTED_THRES=1
export XFT_USE_FAST_SWIGLU=1
export XPU_USE_FAST_SWIGLU=1
export XMLIR_CUDNN_ENABLED=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_FORCE_USE_XPU_GRAPH=1
export VLLM_HOST_IP=$(hostname -i)
export XMLIR_ENABLE_MOCK_TORCH_COMPILE=false
export XMLIR_DYNAMO_WORKAROUND=1
echo "VLLM_HOST_IP: $VLLM_HOST_IP"

python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model /ssd1/models/MiniMax-M2.5-W8A8-INT8-Dynamic \
    --gpu-memory-utilization 0.92 \
    --max-model-len 196608 \
    --max_num_batched_tokens 32768 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --dtype float16 \
    --max_num_seqs 312 \
    --block-size 128 \
    --served-model-name MiniMax-2.5 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2
```

After the server starts, verify that the model is available:

```bash
curl http://127.0.0.1:8000/v1/models
```

The server exposes the model as `MiniMax-2.5` through the OpenAI-compatible API.
