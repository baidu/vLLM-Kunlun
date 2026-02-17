"""kunlun"""

from typing import TYPE_CHECKING, Optional

import psutil
import torch
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None


logger = init_logger(__name__)


class KunlunPlatform(Platform):
    """KunlunPlatform"""

    _enum = PlatformEnum.CUDA
    dist_backend: str = "nccl"
    ray_device_key: str = "GPU"
    device_name: str = "xpu"

    @property
    def device_type(self):
        """
        Return the device type.

        The device type is always ``"cuda"``.
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
        Return the device name.

        The device name is always reported as ``"kunlun"``.

        Args:
            device_id (int, optional):
                The device index. This argument is ignored. Defaults to ``0``.

        Returns:
            str:
                Always ``"kunlun"``.
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
        Return the total memory capacity of a device in bytes.

        By default, the memory size of device ``0`` is returned. A ``ValueError``
        is raised if ``device_id`` is not an integer or falls outside the range
        of available devices.

        Args:
            device_id (int, optional):
                The device index. Defaults to ``0``.

        Raises:
            ValueError:
                If ``device_id`` is not an integer or is out of range.

        Returns:
            int:
                Total device memory in bytes.
        """
        return psutil.virtual_memory().total

    @classmethod
    def inference_mode(cls):
        """
        Enter inference mode by disabling gradient computation.

        Returns:
            torch.no_grad: A context manager that disables gradient computation.
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
        TODO Update here for v0.15.1

        Update default values across different config sections.

        If certain fields are not specified, this function will automatically
        choose appropriate defaults based on runtime conditions.

        - If the cache block size is not set, it defaults to 16.
        - If MLA is enabled and `VLLM_ATTENTION_BACKEND` is not set or is set
        to "FLASHMLA", the cache block size will be updated to 64.
        - When running with the DeepEP high-throughput backend, data parallelism
        greater than 1, and CUDA graph mode, eager execution will be enforced.
        This is because DP + DeepEP high-throughput kernels are not compatible
        with CUDA graphs. The DeepEP low-latency kernels should be used instead.

        Args:
            vllm_config (VllmConfig): The vLLM configuration object.

        Raises:
            NotImplementedError:
                If multi-step scheduling is used in vLLM V1.
                Please remove the `--num-scheduler-steps` argument.
            NotImplementedError:
                If MLA is used in vLLM V1 without setting the
                `VLLM_ATTENTION_BACKEND` environment variable.

        Returns:
            None.
        """
        parallel_config = vllm_config.parallel_config  # Not use scheduler_config
        # scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            # v0.15.1 do not support v0.15.1, remove the if condition
            if vllm_config.speculative_config:
                # if envs.VLLM_USE_V1:
                parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
            else:
                parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        if model_config is not None and model_config.use_mla:
            # if `VLLM_ATTENTION_BACKEND` is not set and we are using MLA, then
            # we default to FlashMLA backend, so we need to force the blocksize
            # here
            use_sparse = hasattr(vllm_config.model_config.hf_config, "index_topk")
            use_flashmla = (
                envs.VLLM_ATTENTION_BACKEND is None
                or envs.VLLM_ATTENTION_BACKEND == "FLASHMLA"
            )
            from vllm.attention.ops.flashmla import is_flashmla_supported

            if (
                use_flashmla
                and is_flashmla_supported()[0]
                and cache_config.block_size != 64
            ):
                cache_config.block_size = 64
                logger.info("Forcing kv cache block size to 64 for FlashMLA backend.")
            if use_sparse and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLASparse " "backend."
                )

        if (
            envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput"
            and parallel_config.data_parallel_size > 1
            and vllm_config.compilation_config.use_cudagraph
        ):
            logger.info(
                "Data Parallel: Forcing enforce eager to be True since DP "
                "with DeepEP high-throughput kernels are not CUDA Graph "
                "compatible. The DeepEP low-latency kernels are CUDA Graph "
                "compatible. Set the all_to_all backend to deepep_low_latency "
                "to use those kernels instead."
            )
            vllm_config.compilation_config.use_cudagraph = False
            vllm_config.model_config.enforce_eager = True
            # TODO (varun): Turning this ON gives incorrect results for the
            # Deepseek-V2-lite model.
            vllm_config.compilation_config.use_inductor = False
        # v0.15.1 do not support v0.15.1, remove the if condition
        # if vllm_config.compilation_config.use_cudagraph:
        #     vllm_config.compilation_config.custom_ops = ["all"]
        #     vllm_config.compilation_config.pass_config.enable_fusion = False
        #     vllm_config.compilation_config.use_inductor = False

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
    ) -> str:
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
        if attn_selector_config.use_mla:
            if attn_selector_config.use_sparse:
                logger.info_once("Using Sparse MLA backend on V1 engine.")
                return (
                    "vllm_kunlun.v1.attention.backends.mla.flashmla_sparse."
                    "FlashMLASparseBackend"
                )
            return "vllm_kunlun.v1.attention.backends.mla.flashmla.FlashMLABackend"
        elif not attn_selector_config.use_mla:
            return (
                "vllm_kunlun.v1.attention.backends.kunlun_attn.KunlunAttentionBackend"
            )
        else:
            return (
                "vllm_kunlun.v1.attention.backends.kunlun_mla.KunlunMLAAttentionBackend"
            )

    @classmethod
    def get_current_memory_usage(
        cls, device: Optional[torch.types.Device] = None
    ) -> float:
        """
        Get the memory usage statistics of the target device, including
        the currently allocated memory and the peak allocation.

        If no device is specified, the device in the current context is used.

        Args:
            device (Optional[torch.types.Device], optional):
                The device to query. Defaults to the current active device.

        Returns:
            float:
                The memory usage of the device in bytes.

        Raises:
            None.
        """
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """
        Return whether asynchronous output is supported.

        By default, Kunlun does not support async output.

        Args:
            enforce_eager (Optional[bool], optional):
                Whether to force eager execution. If set to ``None``, the runtime
                will decide automatically based on the current environment.

        Returns:
            bool:
                ``True`` if async output is supported, otherwise ``False``.
        """
        # Assume Kunlun does not support async output.
        return False

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        communicator
        """
        return "vllm_kunlun.distributed.kunlun_communicator.KunlunCommunicator"

    @classmethod
    def get_punica_wrapper(cls):
        """
        kunlun wrapper
        """
        return "vllm_kunlun.lora.punica_wrapper.punica_kunlun.PunicaWrapperKunlun"

    @classmethod
    def check_if_supports_dtype(cls, torch_dtype: torch.dtype):
        """
        Data Types Supported on the Kunlun3 Platform
        """
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
        """
        Ensure that V1 Graph uses `vllm.unified_attention_with_output_kunlun` as the split op on the Kunlun3 platform.
        """
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True
