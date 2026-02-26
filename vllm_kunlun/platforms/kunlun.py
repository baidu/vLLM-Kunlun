"""kunlun"""
import psutil
import torch

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum, _Backend
from typing import Optional, Union
import vllm.envs as envs
from vllm.logger import init_logger


logger = init_logger(__name__)

class KunlunPlatform(Platform):
    """KunlunPlatform"""
    _enum = PlatformEnum.CUDA 
    dist_backend:str = "nccl"
    ray_device_key: str = "GPU"
    device_name: str = "xpu"

    @property
    def device_type(self):
        """
        返回设备类型，固定为'cuda'。
        """
        return "cuda"
    
    def is_kunlun(self) -> bool:
        """is_kunlun"""
        return self._enum == PlatformEnum.CUDA

    def is_cuda(self) -> bool:
        """is_cuda"""
        return False

    def is_rocm(self) -> bool:
        """is_rocm"""
        return self._enum == PlatformEnum.ROCM

    def is_tpu(self) -> bool:
        """is_tpu"""
        return self._enum == PlatformEnum.TPU

    def is_hpu(self) -> bool:
        """is_hpu"""
        return self._enum == PlatformEnum.HPU

    def is_xpu(self) -> bool:
        """is_xpu"""
        return self._enum == PlatformEnum.XPU

    def is_cpu(self) -> bool:
        """is_cpu"""
        return self._enum == PlatformEnum.CPU

    def is_neuron(self) -> bool:
        """is_neuron"""
        return self._enum == PlatformEnum.NEURON

    def is_out_of_tree(self) -> bool:
        """is_out_of_tree"""
        return self._enum == PlatformEnum.OOT

    def is_cuda_alike(self) -> bool:
        """Stateless version of [torch.cuda.is_available][]."""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    def is_sleep_mode_available(self) -> bool:
        """is_sleep_mode_available"""
        return self._enum == PlatformEnum.CUDA

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """
            获取设备名称，默认返回 "kunlun"。
        
        Args:
            device_id (int, optional): 设备ID，默认为0. Ignored in this method. Defaults to 0.
        
        Returns:
            str: 设备名称，固定返回 "kunlun".
        """
        return "kunlun"

    @classmethod
    def get_piecewise_backend_cls(cls) -> str:
        return "vllm.compilation.cuda_piecewise_backend.CUDAPiecewiseBackend"  # noqa

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"  # noqa

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """
            获取设备总内存大小，单位为字节（B）。默认返回第一个设备的总内存大小。
        如果传入参数`device_id`不是整数或者超出了可用设备范围，将会引发ValueError异常。
        
        Args:
            device_id (int, optional): 设备ID，默认为0. Defaults to 0.
        
        Raises:
            ValueError: 当传入的`device_id`不是整数或者超出了可用设备范围时引发此异常。
        
        Returns:
            int: 设备总内存大小，单位为字节（B）。
        """
        return psutil.virtual_memory().total

    @classmethod
    def inference_mode(cls):
        """
            进入推理模式，禁止计算梯度。
        返回：torch.no_grad()，一个上下文管理器，用于禁止计算梯度。
        """
        return torch.no_grad()

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        """get_device_capability"""
        major, minor = torch.cuda.get_device_capability()
        return DeviceCapability(major=major, minor=minor)
    

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """
            根据配置更新各个部分的默认值。
        如果未指定，则根据某些条件自动选择worker类。
        如果缓存配置中没有设置块大小，则将其设置为16。
        如果使用MLA，并且`VLLM_ATTENTION_BACKEND`未设置或设置为"FLASHMLA"，
        则将缓存块大小设置为64。
        如果在DeepEP高吞吐量后端、数据并行大于1和CUDA图形模式下运行，则强制
        强制执行即时模式，因为DP + DeepEP高吞吐量内核不是CUDA图形兼容的，而且
        使用DeepEP低延迟内核可以解决这个问题。
        
        Args:
            vllm_config (VllmConfig): VLLM配置对象。
        
        Raises:
            NotImplementedError: 如果在vLLM V1上使用多步调度，则会引发NotImplementedError。
            请从命令行中删除--num-scheduler-steps参数。
            NotImplementedError: 如果在vLLM V1上使用MLA，则会引发NotImplementedError。
            请确保在使用MLA之前设置了`VLLM_ATTENTION_BACKEND`环境变量。
        
        Returns:
            None: 无返回值。
        """
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            if vllm_config.speculative_config:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                            "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = \
                        "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                    parallel_config.sd_worker_cls = \
                        "vllm.worker.worker.Worker"
            else:
                print(f"envs.VLLM_USE_V1 = {envs.VLLM_USE_V1}")
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                            "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = "vllm.worker.worker.Worker"
        
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        if model_config is not None and model_config.use_mla:
            # if `VLLM_ATTENTION_BACKEND` is not set and we are using MLA, then
            # we default to FlashMLA backend, so we need to force the blocksize
            # here
            use_sparse = hasattr(vllm_config.model_config.hf_config,
                                 "index_topk")
            use_flashmla = (envs.VLLM_ATTENTION_BACKEND is None \
                or envs.VLLM_ATTENTION_BACKEND == "FLASHMLA")
            from vllm.attention.ops.flashmla import is_flashmla_supported
            if use_flashmla and is_flashmla_supported()[0] \
                and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLA backend.")
            if use_sparse and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLASparse "
                    "backend.")

        if (envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput"
                and parallel_config.data_parallel_size > 1
                and vllm_config.compilation_config.use_cudagraph):
            logger.info(
                "Data Parallel: Forcing enforce eager to be True since DP "
                "with DeepEP high-throughput kernels are not CUDA Graph "
                "compatible. The DeepEP low-latency kernels are CUDA Graph "
                "compatible. Set the all_to_all backend to deepep_low_latency "
                "to use those kernels instead.")
            vllm_config.compilation_config.use_cudagraph = False
            vllm_config.model_config.enforce_eager = True
            # TODO (varun): Turning this ON gives incorrect results for the
            # Deepseek-V2-lite model.
            vllm_config.compilation_config.use_inductor = False
        if vllm_config.compilation_config.use_cudagraph and envs.VLLM_USE_V1:
            vllm_config.compilation_config.custom_ops = ["all"]
            vllm_config.compilation_config.pass_config.enable_fusion = False
            vllm_config.compilation_config.use_inductor = False


    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla,use_sink, use_sparse=False):
        """
            Returns the class of attention backend based on the selected backend and other parameters.
        
        Args:
            selected_backend (str): Selected backend name. Currently supported backends are 'kunlun' and 'default'.
            head_size (int): Size of the attention heads.
            dtype (torch.dtype): Data type of the input tensor.
            kv_cache_dtype (torch.dtype): Data type of the key-value cache.
            block_size (int): Block size used in the attention computation.
            use_v1 (bool, optional): Whether to use v1 version of the backend. Defaults to False.
            use_mla (bool, optional): Whether to use MLA version of the backend. Defaults to False.
        
        Returns:
            str: Class name of the attention backend.
        """
        if use_mla:
            if use_sparse:
                logger.info_once("Using Sparse MLA backend on V1 engine.")
                # return ("vllm.v1.attention.backends.mla.flashmla_sparse."
                #         "FlashMLASparseBackend")
                return ("vllm_kunlun.v1.attention.backends.mla.flashmla_sparse."
                        "FlashMLASparseBackend")
            return "vllm_kunlun.v1.attention.backends.mla.flashmla.FlashMLABackend"
        if use_v1:
            return "vllm_kunlun.v1.attention.backends.kunlun_attn.KunlunAttentionBackend"
        elif not use_mla:                     
            return "vllm_kunlun.ops.attention.backends.kunlun_attn.KunlunAttentionBackend"
        else:
            return "vllm_kunlun.attention.backends.kunlun_mla.KunlunMLAAttentionBackend"

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        """
        获取当前设备的内存使用情况，包括已分配和最大分配。
            如果未指定设备，则默认为当前上下文中的设备。
        
            Args:
                device (Optional[torch.types.Device], optional): 可选的设备对象，默认为None。默认为当前上下文中的设备。
        
            Returns:
                float: 返回一个浮点数，表示当前设备的内存使用情况，单位是字节（bytes）。
        
            Raises:
                None.
        """
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """
            判断是否支持异步输出。
        默认情况下，Kunlun 不支持异步输出。
        
        Args:
            enforce_eager (Optional[bool], optional): 是否强制使用 eager execution. Defaults to None.
                None 表示不强制使用 eager execution，而是根据当前环境自动选择。
        
        Returns:
            bool: True 表示支持异步输出，False 表示不支持异步输出。
        """
        # 假设 Kunlun 不支持异步输出
        return False

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        """
            Check if the model config is supported by this class in v1.
        
        Args:
            model_config (ModelConfig): Model configuration to be checked.
        
        Returns:
            bool: Whether the model config is supported in v1. Always returns True for this class.
        """
        return True

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
       '''
       communicator
       '''
       return "vllm_kunlun.distributed.kunlun_communicator.KunlunCommunicator"

    @classmethod
    def get_punica_wrapper(cls):
        ''' 
        kunlun wrapper
        '''
        return "vllm_kunlun.lora.punica_wrapper.punica_kunlun.PunicaWrapperKunlun"
    
    @classmethod
    def check_if_supports_dtype(cls, torch_dtype: torch.dtype):
        '''
        Kunlun3平台支持的数据类型
        '''
        supported_dtypes = {
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
        }
        if torch_dtype not in supported_dtypes:
            raise ValueError(
                f"Kunlun platform does not support dtype {torch_dtype}. "
                "Supported dtypes are: fp32, fp16, bf16, int8."
            )
       
    def opaque_attention_op(cls) -> bool:
        '''
        确保V1 Graph在Kunlun3平台使用vllm.unified_attention_with_output_kunlun作为split ops 
        '''
        return True
    
    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True