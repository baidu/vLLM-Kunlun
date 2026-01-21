# Multi XPU (Qwen3-Coder-480B-A35B(W8A8))

## Run vllm-kunlun on Multi XPU

Setup environment using container:

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

export build_image="iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.32"


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

- Pull Qwen3-Coder-480B-A35B-Instruct bf16 weights
- Modify the weights configuration.json file and add the fields quantization_config and compression_config.

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "attention_dropout": 0.0,
  "decoder_sparse_step": 1,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 6144,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 262144,
  "max_window_layers": 62,
  "mlp_only_layers": [],
  "model_type": "qwen3_moe",
  "moe_intermediate_size": 2560,
  "norm_topk_prob": true,
  "num_attention_heads": 96,
  "num_experts": 160,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 62,
  "num_key_value_heads": 8,
  "output_router_logits": false,
  "qkv_bias": false,
  "rms_norm_eps": 1e-6,
  "rope_scaling": null,
  "rope_theta": 10000000,
  "router_aux_loss_coef": 0.0,
  "shared_expert_intermediate_size": 0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_qk_norm": true,
  "use_sliding_window": false,
  "vocab_size": 151936,
  "quantization_config": {
    "quant_method": "compressed-tensors"
  },
  "compression_config": {
    "format": "pack_quantized",
    "config_groups": {
      "linear_w8a8": {
        "targets": ["Linear"],
        "weights": {
          "type": "int",
          "num_bits": 8,
          "strategy": "channel",
          "group_size": null,
          "symmetric": true,
          "dynamic": false
        },
        "input_activations": {
          "type": "int",
          "num_bits": 8,
          "strategy": "token",
          "group_size": null,
          "symmetric": true,
          "dynamic": true
        }
      }
    },
    "ignore": [],
    "sparsity_config": null
  }
}
```

### Online Serving on Multi XPU

Start the vLLM server on multi XPU:

```bash
python3 -m vllm.entrypoints.openai.api_server \
 --host 0.0.0.0 \
 --port 8898 \
 --model /Qwen/Qwen3-Coder-480B-A35B-Instruct \
 --dtype float16 \
 --trust-remote-code \
 --tensor-parallel-size 8 \
 --block-size 128 \
 --max-model-len 40960 \
 --max-num-seqs 512 \
 --max-num-batched-tokens 40960 \
 --max-seq-len-to-capture 40960 \
 --distributed-executor-backend mp \
 --enable-chunked-prefill=False \
 --no-enable-prefix-caching \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --compilation-config '{"splitting_ops": ["vllm.unified_attention",
                                                "vllm.unified_attention_with_output",
                                                "vllm.unified_attention_with_output_kunlun",
                                                "vllm.mamba_mixer2",
                                                "vllm.mamba_mixer",
                                                "vllm.short_conv",
                                                "vllm.linear_attention",
                                                "vllm.plamo2_mamba_mixer",
                                                "vllm.gdn_attention",
                                                "vllm.sparse_attn_indexer"]}' 2>&1 | tee output_p800.log
```
