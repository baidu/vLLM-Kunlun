# Environment Variables

The following table documents commonly used environment variables. Some are
consumed by the XMLIR or XPU runtime rather than by vLLM-Kunlun itself, so
availability and exact behavior can depend on the installed runtime version.

| Environment variable | Recommended value | Description |
| --- | --- | --- |
| `XPU_DUMMY_EVENT` | unset | Uses real XPU events instead of dummy events for synchronization. |
| `XPU_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | Selects the physical XPU devices visible to the process. Adjust this list to the deployment. |
| `XPU_USE_MOE_SORTED_THRES` | `512` (runtime default) | Legacy `moe_ffn_block` threshold on token rows (`M`): the sorted path is selected at or above this value. The current vLLM-Kunlun fused MoE path does not read this variable, so it should normally remain unset. |
| `XFT_USE_FAST_SWIGLU` | `1` | Enables the fast SwiGLU implementation in XFT operators. |
| `XPU_USE_FAST_SWIGLU` | `1` | Enables the fast SwiGLU implementation in Kunlun MoE operators. |
| `XMLIR_CUDNN_ENABLED` | `1` | Enables the XMLIR runtime's cuDNN-compatible path. The exact implementation depends on the installed runtime. |
| `XPU_USE_DEFAULT_CTX` | `1` | Uses the XPU runtime's default device context. |
| `XMLIR_FORCE_USE_XPU_GRAPH` | `1` | Routes CUDA Graph-compatible APIs to XPU Graph capture and replay. |
| `VLLM_HOST_IP` | `$(hostname -i)` | Advertises the host address to vLLM workers. On multi-NIC hosts, set one explicit address reachable by every worker. |
| `XMLIR_ENABLE_MOCK_TORCH_COMPILE` | `false` | Disables the default eager-only `torch.compile` mock so calls use the real compilation path. |
| `XMLIR_DYNAMO_WORKAROUND` | `1` | Registers `F.linear` as a custom operator to avoid Dynamo tracing issues on Kunlun XPU. |
| `FUSED_QK_ROPE_OP` | `0` | Controls the fused QK-Norm and RoPE implementation. Keep `0` unless a model-specific setup requires it. |
