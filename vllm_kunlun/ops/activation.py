# SPDX-License-Identifier: Apache-2.0
"""Custom activation functions."""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import LazyDict


@CustomOp.register("kunlun_fatrelu_and_mul")
class FatreluAndMul(CustomOp):
    """An activation function for FATReLU.

    The function computes x -> FATReLU(x[:d]) * x[d:] where
    d = x.shape[-1] // 2.
    This is used in openbmb/MiniCPM-S-1B-sft.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self, threshold: float = 0.):
        """
            Initializes the instance.
        
        Args:
            threshold (float, optional): Threshold value for the filter. Defaults to 0..
        
        Returns:
            None: This method does not return anything.
        """
        super().__init__()
        self.threshold = threshold

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输入张量的正向传播，并返回一个新的张量。
            该函数实现了原生的前向传播过程，即对输入张量进行阈值化处理后，将其乘以另一个张量。
        
            Args:
                x (torch.Tensor, shape=[*, d]):
                    输入张量，其中*表示任意维度，d为特征维度。
        
            Returns:
                torch.Tensor, shape=[*, d]:
                    返回一个新的张量，其形状与输入张量相同，除了最后一个维度被设置为d/2。
                    如果输入张量的最后一个维度小于等于d/2，则返回的张量将保持不变；否则，将对输入张量进行阈值化处理。
        """
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        x1 = F.threshold(x1, self.threshold, 0.0)
        return x1 * x2

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """
        在CUDA设备上执行前向传播。
        
        Args:
            x (torch.Tensor): 输入张量，形状为(N, C, H, W)。
        
        Returns:
            torch.Tensor: 输出张量，形状为(N, C, H, W)。
        """
        return self.forward_native(x)


@CustomOp.register("kunlun_silu_and_mul")
class SiluAndMul(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """forward_cuda"""
        import xtorch_ops
        
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(out, x)
        return out

    def forward_kunlun(self, x: torch.Tensor) -> torch.Tensor:
        """forward_kunlun"""
        import xtorch_ops
        
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        xtorch_ops.swiglu(x, out)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply the function on `x` using XPU backend.
        
        Args:
            x (torch.Tensor): Input tensor of any shape. Must be a floating point tensor.
                The number of channels should be even.
        
        Returns:
            torch.Tensor: Output tensor with the same shape as input except the last dimension is reduced by half.
            It has the same dtype as the input and lives on the same device.
        
        Raises:
            None
        """
        from vllm._ipex_ops import ipex_ops as ops

        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.silu_and_mul(out, x)
        return out

    def forward_neuron(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播一个神经元，计算输入的信号。
            参数：
                x (torch.Tensor): 形状为(-1, d)的张量，其中d是输入的维度。
                                  每个元素表示一个输入信号。
            返回值（torch.Tensor）：
                形状为(-1, d)的张量，其中d是输出的维度。
                每个元素表示一个输出信号。
        """
        d = x.shape[-1] // 2
        x_reshaped = x.view(-1, x.shape[-1])
        s = x_reshaped[:, :d] * F.sigmoid(x_reshaped[:, :d])
        result = s * x_reshaped[:, d:]
        return result.view(*x.shape[:-1], d)


@CustomOp.register("kunlun_mul_and_silu")
class MulAndSilu(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> x[:d] * silu(x[d:]) where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self):
        """
            初始化函数，用于实例化类的对象。
        如果当前平台是 CUDA 或 XPU，则使用 torch.ops._C.mul_and_silu 进行操作；
        否则，如果当前平台是 CPU，则使用 forward_native 方法进行操作。
        """
        super().__init__()
        if current_platform.is_cuda_alike():
            self.op = torch.ops._C.mul_and_silu
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops
            self.op = ipex_ops.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return x[..., :d] * F.silu(x[..., d:])

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """
        在CUDA设备上执行前向传播操作。
        
        Args:
            x (torch.Tensor): 输入张量，其形状应为（..., d），其中d是特征维度。
        
        Returns:
            torch.Tensor: 输出张量，其形状与输入张量相同，但最后一个维度被替换为d/2。
        
        Raises:
            无。
        """
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    # TODO implement forward_xpu for MulAndSilu
    # def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:


@CustomOp.register("kunlun_gelu_and_mul")
class GeluAndMul(CustomOp):
    """An activation function for GeGLU.

    The function computes x -> GELU(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    def __init__(self, approximate: str = "none"):
        """
            Initializes the instance.
        
        Args:
            approximate (str, optional): The approximation method to use. Defaults to "none".
                Can be one of "none", "tanh".
        
        Raises:
            ValueError: If the `approximate` parameter is not one of "none", "tanh".
        """
        super().__init__()
        self.approximate = approximate
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unknown approximate mode: {approximate}")

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """
        在CUDA设备上进行前向传播。
        
        Args:
            x (torch.Tensor): 输入张量，形状为（batch_size, ..., dim），其中dim是特征维度。
        
        Returns:
            torch.Tensor: 输出张量，形状为（batch_size, ..., dim//2），其中dim是特征维度，除以2是因为GELU的输出是两个分量。
        
        Raises:
            无。
        """
        # from vllm import _custom_ops as ops
        import xtorch_ops
        # d = x.shape[-1] // 2
        # output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(x, dtype=x.dtype, device=x.device)
        if self.approximate == "none":
            # ops.gelu_and_mul(out, x)
            print(x,x.shape)
            xtorch_ops.gelu(x, out)
        elif self.approximate == "tanh":
            ops.gelu_tanh_and_mul(out, x)
        return out
        
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d, _ = self._check_and_make_out(x)
        # 保守地用 contiguous，避免 view 相关坑
        x = x.contiguous()
        x1 = x[..., :d]
        x2 = x[..., d:]
        return F.gelu(x1, approximate=self.approximate) * x2

    # def forward_native(self, x: torch.Tensor) -> torch.Tensor:
    #     """PyTorch-native implementation equivalent to forward()."""
    #     d = x.shape[-1] // 2
    #     return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply gelu activation function on input tensor using iPEX backend.
        
        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).
                The data type can be float32 or float64.
        
        Returns:
            torch.Tensor: Output tensor with the same shape and data type as input.
            The output will have a range of (-0.5, 0.5) for tanh approximation.
        """
        from vllm._ipex_ops import ipex_ops as ops

        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        if self.approximate == "none":
            ops.gelu_and_mul(out, x)
        elif self.approximate == "tanh":
            ops.gelu_tanh_and_mul(out, x)
        return out

    def extra_repr(self) -> str:
        """
            返回一个字符串，包含有关模型的额外信息。这个函数可以被用于打印出模型的概要信息。
        默认情况下，这个函数会返回一个包含模型是否使用近似值（approximate）的信息。
        
        Returns:
            str (str): 一个字符串，包含有关模型的额外信息。
        """
        return f'approximate={repr(self.approximate)}'


@CustomOp.register("kunlun_gelu_new")
class NewGELU(CustomOp):

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c *
                                           (x + 0.044715 * torch.pow(x, 3.0))))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算CUDA上的GELU函数。
        
        Args:
            x (torch.Tensor): 输入张量，形状为(N, C, H, W)。
        
        Returns:
            torch.Tensor: GELU函数的结果，形状与输入相同。
        
        Raises:
            无。
        """
        from vllm import _custom_ops as ops

        out = torch.empty_like(x)
        ops.gelu_new(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply the GELU activation function element-wise.
        
        Args:
            x (torch.Tensor): Input tensor with any shape. The data type is float32 or float64.
        
        Returns:
            torch.Tensor: Output tensor with the same shape as input. The data type is the same as input.
        
        Raises:
            None
        """
        from vllm._ipex_ops import ipex_ops as ops

        return ops.gelu_new(x)


@CustomOp.register("kunlun_gelu_fast")
class FastGELU(CustomOp):

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 *
                                           (1.0 + 0.044715 * x * x)))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """
            计算输入张量x的CUDA版本GELU（Gaussian Error Linear Unit）。
        该函数调用了vllm模块中的_custom_ops模块中的gelu_fast函数，完成GELU操作。
        
        Args:
            x (torch.Tensor): 输入张量，形状为(N, C, H, W)，类型为float32或float64。
        
        Returns:
            torch.Tensor: GELU后的输出张量，形状与x相同，类型与x相同。
        
        Raises:
            无。
        """
        from vllm import _custom_ops as ops

        out = torch.empty_like(x)
        ops.gelu_fast(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply the GELU function element-wise on input tensor ``x``.
        
        Args:
            x (torch.Tensor): Input tensor with any shape. The data type can be float or half float.
                The range of the input values is expected to be -inf to inf.
        
        Returns:
            torch.Tensor: Output tensor with the same shape and data type as input ``x``.
            The output values are in the range [-0.5, 0.5] for float dtype and [-15, 15] for half float dtype.
        
        Raises:
            TypeError: If the input ``x`` is not a torch.Tensor.
            RuntimeError: If the input ``x`` contains non-finite numbers.
        """
        from vllm._ipex_ops import ipex_ops as ops

        return ops.gelu_fast(x)


@CustomOp.register("kunlun_quick_gelu")
class QuickGELU(CustomOp):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return x * torch.sigmoid(1.702 * x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用CUDA设备进行前向计算。
        
        Args:
            x (torch.Tensor): 输入张量，形状为（N, C, H, W）。
        
        Returns:
            torch.Tensor: 输出张量，形状与输入相同，值为GELU函数的结果。
        
        Raises:
            无。
        """
        from vllm import _custom_ops as ops

        out = torch.empty_like(x)
        ops.gelu_quick(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply the GELU function element-wise on input tensor ``x``.
        
        Args:
            x (torch.Tensor): Input tensor with any shape. The data type is float32 or float64.
        
        Returns:
            torch.Tensor: Output tensor with the same shape and data type as input ``x``.
        
        Raises:
            None
        """
        from vllm._ipex_ops import ipex_ops as ops

        out = torch.empty_like(x)
        ops.gelu_quick(out, x)
        return out

    def forward_kunlun(self, x: torch.Tensor) -> torch.Tensor:
        """forward_kunlun"""
        from vllm._kunlun_ops import KunlunOps as ops
        out = torch.empty_like(x)
        ops.quick_gelu(out, x)
        return out


@CustomOp.register("kunlun_relu2")
class ReLUSquaredActivation(CustomOp):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return torch.square(F.relu(x))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """
        在CUDA设备上执行前向传播。
        
        Args:
            x (torch.Tensor): 输入张量，形状为(N, C, H, W)，数据类型为float32或float64。
        
        Returns:
            torch.Tensor: 输出张量，形状与输入相同，数据类型与输入一致。
        
        Raises:
            无。
        """
        return self.forward_native(x)


class ScaledActivation(nn.Module):
    """An activation function with post-scale parameters.

    This is used for some quantization methods like AWQ.
    """

    def __init__(
        self,
        act_module: nn.Module,
        intermediate_size: int,
        input_is_parallel: bool = True,
        params_dtype: Optional[torch.dtype] = None,
    ):
        """
            Initializes the LayerNorm module.
        
        Args:
            act_module (nn.Module): The activation function to use after layer norm.
                Default: nn.GELU()
            intermediate_size (int): The size of the intermediate representation.
            input_is_parallel (bool, optional): Whether the input is parallelly processed.
                Default: True
            params_dtype (Optional[torch.dtype], optional): The data type of parameters.
                If None, use the default data type. Default: None
        """
        super().__init__()
        self.act = act_module
        self.input_is_parallel = input_is_parallel
        if input_is_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            intermediate_size_per_partition = divide(intermediate_size,
                                                     tp_size)
        else:
            intermediate_size_per_partition = intermediate_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.scales = nn.Parameter(
            torch.empty(intermediate_size_per_partition, dtype=params_dtype))
        set_weight_attrs(self.scales, {"weight_loader": self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，将输入的张量进行缩放和激活操作。
        
        Args:
            x (torch.Tensor): 输入张量，形状为（N, C, H, W）或者（N, C, H, W, D）。
        
        Returns:
            torch.Tensor: 返回处理后的张量，形状与输入相同。
        """
        return self.act(x) / self.scales

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
            加载权重，如果输入是并行的，则需要将其平均分配到每个模型参数中。
        参数：
            param (nn.Parameter): 需要加载权重的模型参数。
            loaded_weight (torch.Tensor): 加载的权重张量。
        返回值：
            无返回值，直接修改了param的数据。
        """
        param_data = param.data
        if self.input_is_parallel:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = param_data.shape[0]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


_ACTIVATION_REGISTRY = LazyDict({
    "gelu":
    lambda: nn.GELU(),
    "gelu_fast":
    lambda: FastGELU(),
    "gelu_new":
    lambda: NewGELU(),
    "gelu_pytorch_tanh":
    lambda: nn.GELU(approximate="tanh"),
    "relu":
    lambda: nn.ReLU(),
    "relu2":
    lambda: ReLUSquaredActivation(),
    "silu":
    lambda: nn.SiLU(),
    "quick_gelu":
    lambda: QuickGELU(),
})


def get_act_fn(
    act_fn_name: str,
    quant_config: Optional[QuantizationConfig] = None,
    intermediate_size: Optional[int] = None,
    input_is_parallel: bool = True,
    params_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()
    # print(f"activation function name: {act_fn_name}")
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Activation function {act_fn_name!r} is not supported.")

    act_fn = _ACTIVATION_REGISTRY[act_fn_name]
    if (quant_config is not None
            and act_fn_name in quant_config.get_scaled_act_names()):
        if intermediate_size is None:
            raise ValueError("intermediate_size must be specified for scaled "
                             "activation functions.")
        return ScaledActivation(act_fn, intermediate_size, input_is_parallel,
                                params_dtype)
    return act_fn

_ACTIVATION_AND_MUL_REGISTRY = LazyDict({
    "gelu": lambda: GeluAndMul(),
    "silu": lambda: SiluAndMul(),
    "geglu": lambda: GeluAndMul(),
})


def get_act_and_mul_fn(act_fn_name: str) -> nn.Module:
    """Get an activation-and-mul (i.e. SiluAndMul) function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_AND_MUL_REGISTRY:
        raise ValueError(
            f"Activation function {act_fn_name!r} is not supported.")

    return _ACTIVATION_AND_MUL_REGISTRY[act_fn_name]
