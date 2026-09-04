# Baseline environment for an eight-device Kunlun P800 node. Adjust device
# visibility and host addressing for the deployment environment.

# Use real XPU events instead of dummy events for synchronization.
unset XPU_DUMMY_EVENT

# Expose all eight XPU devices on a standard P800 node.
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Enable the fast SwiGLU implementation in XFT operators.
export XFT_USE_FAST_SWIGLU=1

# Enable the fast SwiGLU implementation in Kunlun MoE operators.
export XPU_USE_FAST_SWIGLU=1

# Enable the XMLIR runtime's cuDNN-compatible path (vendor-recommended value).
export XMLIR_CUDNN_ENABLED=1

# Use the XPU runtime's default device context.
export XPU_USE_DEFAULT_CTX=1

# Route CUDA Graph-compatible APIs to XPU Graph capture and replay.
export XMLIR_FORCE_USE_XPU_GRAPH=1

# Advertise this host address to vLLM workers. On multi-NIC hosts, replace
# hostname -i with one explicit address reachable by every worker.
export VLLM_HOST_IP=$(hostname -i)

# Use the real torch.compile path instead of the default eager-only mock.
export XMLIR_ENABLE_MOCK_TORCH_COMPILE=false

# Register F.linear as a custom op to avoid Dynamo tracing issues on XPU.
export XMLIR_DYNAMO_WORKAROUND=1
