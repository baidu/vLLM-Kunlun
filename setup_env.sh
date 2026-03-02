unset XPU_DUMMY_EVENT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XFT_USE_FAST_SWIGLU=1 #使用快速swiglu实现
export XPU_USE_FAST_SWIGLU=1 #使用moe算子中快速swiglu实现
export XMLIR_CUDNN_ENABLED=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_FORCE_USE_XPU_GRAPH=1 # 优化图间sync
export XPU_USE_MOE_SORTED_THRES=128 # Moe sort threshold
export VLLM_HOST_IP=$(hostname -i)
export XMLIR_ENABLE_MOCK_TORCH_COMPILE=false 
VLLM_USE_V1=1
##默认值为1，设置为0启用QWN3融合大算子
USE_ORI_ROPE=1