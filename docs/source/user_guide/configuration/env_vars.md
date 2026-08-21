# Environment Variables

vLLM Kunlun uses both runtime variables from the Kunlun XPU stack and
vLLM-Kunlun-specific variables from `vllm_kunlun.platforms.envs`.

## Runtime Environment

| Environment variable | Recommended value | Description |
| --- | --- | --- |
| `XPU_DUMMY_EVENT` | unset | Uses real XPU events for synchronization and performance measurement. |
| `XPU_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | Selects the visible XPU devices for multi-card inference. |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | Keeps CUDA-compatible device indexing aligned with visible XPUs. |
| `XPU_USE_MOE_SORTED_THRES` | `1` | Enables MoE sorting optimization in the XPU runtime. |
| `XFT_USE_FAST_SWIGLU` | `1` | Enables the fast SwiGLU implementation in XFT. |
| `XPU_USE_FAST_SWIGLU` | `1` | Enables the fast SwiGLU implementation for XPU MoE kernels. |
| `XMLIR_CUDNN_ENABLED` | `1` | Enables the XMLIR cuDNN-compatible optimized path. |
| `XPU_USE_DEFAULT_CTX` | `1` | Uses the default XPU context for runtime consistency. |
| `XMLIR_FORCE_USE_XPU_GRAPH` | `1` | Forces XPU graph mode for graph capture and execution optimization. |
| `VLLM_HOST_IP` | `$(hostname -i)` | Sets the host IP used by distributed vLLM communication. |
| `XMLIR_ENABLE_MOCK_TORCH_COMPILE` | `false` | Disables mock torch compile so the real compile path is used. |

## vLLM-Kunlun Variables

These variables are parsed lazily by `vllm_kunlun.platforms.envs`. Boolean
variables are enabled only when set to `true` or `1`, case-insensitively.

| Environment variable | Default | Description |
| --- | --- | --- |
| `VLLM_MULTI_LOGPATH` | `./logs` | Directory used by the multi-log redirection helper. |
| `ENABLE_VLLM_MULTI_LOG` | `False` | Enables multi-process log redirection for multi-node or multi-card runs. |
| `ENABLE_VLLM_INFER_HOOK` | `False` | Enables XVLLM inference-stage hook logging. |
| `ENABLE_VLLM_OPS_HOOK` | `False` | Enables XVLLM operator hook logging. |
| `ENABLE_VLLM_MODULE_HOOK` | `False` | Enables XVLLM module hook logging. |
| `ENABLE_VLLM_MOE_FC_SORTED` | `False` | Enables the fused sorted MoE FC path when supported by the active model path. |
| `ENABLE_CUSTOM_DPSK_SCALING_ROPE` | `False` | Enables the custom DeepSeek scaling RoPE path. |
| `ENABLE_VLLM_FUSED_QKV_SPLIT_NORM_ROPE` | `False` | Enables fused QKV split, QK norm, and RoPE for supported Qwen3 models. |
| `VLLM_KUNLUN_ENABLE_INT8_BMM` | `False` | Enables the INT8 BMM path for supported MLA attention code paths. |
