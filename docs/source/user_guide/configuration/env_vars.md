# Environment Variables

vllm-kunlun uses the following environment variables to configure the system:

| *Environment Variables*                     | ***\*Recommended value\****  | ***\*Function description\****                                           |
| ---------------------------------------- | ----------------- | ------------------------------------------------------------ |
| `unset XPU_DUMMY_EVENT`                  |                   | ***\*Unsets\**** `XPU_DUMMY_EVENT` variable, usually done to ensure real XPU events are used for synchronization and performance measurement. |
| `export XPU_VISIBLE_DEVICES`             | `0,1,2,3,4,5,6,7` | ***\*Specify visible XPU Devices\****. Here, 8 devices (0 to 7) are specified for inference tasks. This is required for multi-card or distributed inference. |
| `export XPU_USE_MOE_SORTED_THRES`        | `1`               | Enables the Moe Model ***\*Sort Optimization\****.Setting to `1` usually enables this performance optimization. |
| `export XFT_USE_FAST_SWIGLU`             | `1`               | Enables the ***\*Fast SwiGLU Ops\****. SwiGLU is a common activation function, and enabling this accelerates model inference. |
| `export XPU_USE_FAST_SWIGLU`             | `1`               | Enables the ***\*Fast SwiGLU Ops\****. Similar to `XFT_USE_FAST_SWIGLU`, this enables the fast SwiGLU calculation in Fused MoE Fusion Ops. |
| `export XMLIR_CUDNN_ENABLED`             | `1`               | Enables XMLIR (an intermediate representation/compiler) to use the ***\*cuDNN compatible/optimized path\**** (which may map to corresponding XPU optimized libraries in the KunlunCore environment). |
| `export XPU_USE_DEFAULT_CTX`             | `1`               | Sets the XPU to use the default context. Typically used to simplify environment configuration and ensure runtime consistency. |
| `export XMLIR_FORCE_USE_XPU_GRAPH`       | `1`               | ***\*Forces the enablement of XPU Graph mode.\****. This can capture and optimize the model execution graph, significantly boosting inference performance. |
| `export VLLM_HOST_IP`                    | `$(hostname -i)`  | ***\*Sets the host IP address for the vLLM service\****. This uses a shell command to dynamically get the current host's internal IP. It's used for inter-node communication in a distributed environment. |
| `export XMLIR_ENABLE_MOCK_TORCH_COMPILE` | `false`           | ***\*Disable Mock Torch Compile Function\****. Set to `false` to ensure the actual compilation and optimization flow is used, rather than mock mode. |
| `FUSED_QK_ROPE_OP`                           | `0`               | ***\*Control whether to use the Fused QK-Norm and RoPE implementation\****. Default is `0` (use original/standard RoPE). Setting to `1` may be used to enable QWEN3. |