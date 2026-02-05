"""vllm_utils_wrapper.py"""

import vllm.distributed.parallel_state as parallel_state
import vllm.utils as _orig
from typing import Any, Callable, Optional, Union, get_origin, get_args, List, Tuple
from types import SimpleNamespace
import torch
from torch.library import Library
import inspect
import typing
from torch.library import register_fake
import vllm_kunlun._kunlun
import vllm.envs as envs

def patch_annotations_for_schema(func):
    """patch_annotations_for_schema"""
    sig = inspect.signature(func)
    new_params = []

    for name, param in sig.parameters.items():
        ann = param.annotation

        if get_origin(ann) is typing.Union and type(None) in get_args(ann):
            inner_type = [a for a in get_args(ann) if a is not type(None)][0]
            if get_origin(inner_type) is list:  # Optional[list[int]]
                inner_args = get_args(inner_type)
                new_ann = Optional[List[inner_args[0] if inner_args else typing.Any]]
                param = param.replace(annotation=new_ann)

        elif get_origin(ann) is list:
            args = get_args(ann)
            new_ann = List[args[0] if args else typing.Any]
            param = param.replace(annotation=new_ann)

        new_params.append(param)

    func.__signature__ = sig.replace(parameters=new_params)
    return func


def supports_custom_op() -> bool:
    """supports_custom_op"""
    return hasattr(torch.library, "custom_op")


vllm_lib = Library("vllm", "FRAGMENT")  # noqa


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: Optional[list[str]] = None,
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
    dispatch_key: str = "CUDA",
    tags: tuple[torch.Tag, ...] = (),
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.
    """
    if not supports_custom_op():
        from vllm.platforms import current_platform

        assert not current_platform.is_cuda_alike(), (
            "cuda platform needs torch>=2.4 to support custom op, "
            "chances are you are using an old version of pytorch "
            "or a custom build of pytorch. It is recommended to "
            "use vLLM in a fresh new environment and let it install "
            "the required dependencies."
        )
        return
    if mutates_args is None:
        mutates_args = []
    import torch.library

    if hasattr(torch.library, "infer_schema"):
        patched_func = patch_annotations_for_schema(op_func)
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)


def vllm_kunlun_weak_ref_tensor(tensor: Any) -> Any:
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    """
    # return tensor
    if isinstance(tensor, torch.Tensor):
        return torch.ops._kunlun.weak_ref_tensor(tensor)
    else:
        return tensor


def vllm_kunlun_weak_ref_tensors(
    tensors: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]],
) -> Union[torch.Tensor, list[Any], tuple[Any], Any]:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    """
    if isinstance(tensors, torch.Tensor):
        return vllm_kunlun_weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [vllm_kunlun_weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(vllm_kunlun_weak_ref_tensor(t) for t in tensors)
    raise ValueError("Invalid type for tensors")

vllm_port=envs.VLLM_PORT
def _get_open_port() -> int:
    global vllm_port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", vllm_port))
            vllm_port += 1
            return vllm_port
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

_wrapped = SimpleNamespace(**_orig.__dict__)
_wrapped.direct_register_custom_op = direct_register_custom_op
_wrapped.weak_ref_tensor = vllm_kunlun_weak_ref_tensor
_wrapped.weak_ref_tensors = vllm_kunlun_weak_ref_tensors
_wrapped._get_open_port = _get_open_port

import sys

sys.modules["vllm.utils"] = _wrapped

_original_all_reduce = parallel_state.GroupCoordinator.all_reduce
_original_all_gather = parallel_state.GroupCoordinator.all_gather


def vllm_kunlun_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    """vllm_kunlun_all_reduce"""
    if self.world_size == 1:
        return input_

    torch.distributed.all_reduce(input_, group=self.device_group)
    return input_


def vllm_kunlun_all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """vllm_kunlun_all_reduce"""
    world_size = self.world_size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert (
        -input_.dim() <= dim < input_.dim()
    ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"

    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty(
        (world_size,) + input_size, dtype=input_.dtype, device=input_.device
    )
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=self.device_group
    )
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(
        input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
    )
    return output_tensor


parallel_state.GroupCoordinator.all_reduce = vllm_kunlun_all_reduce
parallel_state.GroupCoordinator.all_gather = vllm_kunlun_all_gather


from torch.library import custom_op, impl
import torch
from vllm import _custom_ops as ops
from typing import Optional, List
import os


@custom_op("_C::rms_norm", mutates_args=())
def rms_norm(
    result: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> None:
    pass


@custom_op("_C::fused_add_rms_norm", mutates_args=())
def fused_add_rms_norm(
    result: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> None:
    pass


@custom_op("_C::static_scaled_fp8_quant", mutates_args=())
def static_scaled_fp8_quant(
    result: torch.Tensor, input: torch.Tensor, scale: torch.Tensor
) -> None:
    pass


@impl("_C::static_scaled_fp8_quant", "CUDA")
def static_scaled_fp8_quant_xpu(
    result: torch.Tensor, input: torch.Tensor, scale: torch.Tensor
) -> None:
    pass


@custom_op("_C::dynamic_scaled_fp8_quant", mutates_args=())
def dynamic_scaled_fp8_quant(
    result: torch.Tensor, input: torch.Tensor, scale: torch.Tensor
) -> None:
    pass


@impl("_C::dynamic_scaled_fp8_quant", "CUDA")
def dynamic_scaled_fp8_quant_xpu(
    result: torch.Tensor, input: torch.Tensor, scale: torch.Tensor
) -> None:
    pass


@custom_op("_C::dynamic_per_token_scaled_fp8_quant", mutates_args=())
def dynamic_per_token_scaled_fp8_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: Optional[torch.Tensor],
) -> None:
    pass


@impl("_C::dynamic_per_token_scaled_fp8_quant", "CUDA")
def dynamic_per_token_scaled_fp8_quant_xpu(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: Optional[torch.Tensor],
) -> None:
    pass


@custom_op("_C::rms_norm_static_fp8_quant", mutates_args=())
def rms_norm_static_fp8_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> None:
    pass


@impl("_C::rms_norm_static_fp8_quant", "CUDA")
def rms_norm_static_fp8_quant_xpu(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> None:
    pass


@custom_op("_C::fused_add_rms_norm_static_fp8_quant", mutates_args=())
def fused_add_rms_norm_static_fp8_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> None:
    pass


@impl("_C::fused_add_rms_norm_static_fp8_quant", "CUDA")
def fused_add_rms_norm_static_fp8_quant_xpu(
    result: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> None:
    pass


@custom_op("_C::rms_norm_dynamic_per_token_quant", mutates_args=())
def rms_norm_dynamic_per_token_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
) -> None:
    pass


@impl("_C::rms_norm_dynamic_per_token_quant", "CUDA")
def rms_norm_dynamic_per_token_quant_xpu(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
) -> None:
    pass


@custom_op("_C::silu_and_mul_quant", mutates_args=())
def silu_and_mul_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> None:
    pass


@impl("_C::silu_and_mul_quant", "CUDA")
def silu_and_mul_quant_xpu(
    result: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> None:
    pass


import torch
import kunlun_ops
from torch.library import custom_op, impl


@custom_op("_C::add_rmsnorm", mutates_args=())
def add_rmsnorm(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-5,
    interweaved: bool = False,
    store_output_before_norm: bool = True,
    bias: torch.Tensor = None,
    smooth: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    output_max: torch.Tensor = None,
) -> None:
    kunlun_ops.add_rmsnorm(
        x,
        y,  # 原来写 residual，这里其实是 y
        residual_output=residual_output,
        weight=weight,
        eps=eps,
        output=output,
    )


@impl("_C::add_rmsnorm", "CUDA")
def add_rmsnorm_cuda(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-5,
    interweaved: bool = False,
    store_output_before_norm: bool = True,
    bias: torch.Tensor = None,
    smooth: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    output_max: torch.Tensor = None,
) -> None:
    kunlun_ops.add_rmsnorm(
        x,
        y,
        residual_output=residual_output,
        weight=weight,
        eps=eps,
        output=output,
    )


@custom_op("_C::rmsnorm", mutates_args=())
def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-5,
    interweave: bool = False,
    store_output_before_norm: bool = True,
    bias: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    output_max: torch.Tensor = None,
) -> None:
    kunlun_ops.rmsnorm(
        x,
        weight,
        output,
        eps,
    )


@impl("_C::rmsnorm", "CUDA")
def rmsnorm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-5,
    interweave: bool = False,
    store_output_before_norm: bool = True,
    bias: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    output_max: torch.Tensor = None,
) -> None:
    kunlun_ops.rmsnorm(
        x,
        weight,
        output,
        eps,
    )


import torch


def _fake_rmsnorm(
    x,
    weight,
    output,
    eps=1e-5,
    interweave=False,
    store_output_before_norm=True,
    bias=None,
    residual_output=None,
    output_max=None,
):
    # 设置 shape/dtype，但不返回值
    output.fake_shape = x.shape
    output.fake_dtype = x.dtype
    return None


rmsnorm.register_fake(_fake_rmsnorm)


def _fake_add_rmsnorm(
    x,
    y,
    weight,
    output,
    eps=1e-5,
    interweaved=False,
    store_output_before_norm=True,
    bias=None,
    smooth=None,
    residual_output=None,
    output_max=None,
):
    output.fake_shape = x.shape
    output.fake_dtype = x.dtype
    return None


add_rmsnorm.register_fake(_fake_add_rmsnorm)


@custom_op("_C::gemma_add_rmsnorm", mutates_args=())
def gemma_add_rmsnorm(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-5,
    enable_pdl: bool = False,
    interweaved: bool = False,
    store_output_before_norm: bool = True,
    bias: torch.Tensor = None,
    smooth: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    force_sdnn: bool = False
) -> None:
    #print("gemma_add_rmsnorm wrapper")
    kunlun_ops.gemma_add_rmsnorm(
        x,
        y,  # 原来写 residual，这里其实是 y
        weight=weight,
        output=output,
        eps=eps,
        enable_pdl=enable_pdl,
        interweaved=interweaved,
        store_output_before_norm=store_output_before_norm,
        bias=bias,
        smooth=smooth,
        residual_output=residual_output,
        force_sdnn=force_sdnn
    )


@impl("_C::gemma_add_rmsnorm", "CUDA")
def gemma_add_rmsnorm_cuda(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-5,
    enable_pdl: bool = False,
    interweaved: bool = False,
    store_output_before_norm: bool = True,
    bias: torch.Tensor = None,
    smooth: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    force_sdnn: bool = False
) -> None:
    #print("gemma_add_rmsnorm_cuda wrapper")
    kunlun_ops.gemma_add_rmsnorm(
        x,
        y,  # 原来写 residual，这里其实是 y
        weight=weight,
        output=output,
        eps=eps,
        enable_pdl=enable_pdl,
        interweaved=interweaved,
        store_output_before_norm=store_output_before_norm,
        bias=bias,
        smooth=smooth,
        residual_output=residual_output,
        force_sdnn=force_sdnn
    )


def _fake_gemma_add_rmsnorm(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-5,
    enable_pdl: bool = False,
    interweaved: bool = False,
    store_output_before_norm: bool = True,
    bias: torch.Tensor = None,
    smooth: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    force_sdnn: bool = False
):
    output.fake_shape = x.shape
    output.fake_dtype = x.dtype
    return None


gemma_add_rmsnorm.register_fake(_fake_gemma_add_rmsnorm)


@custom_op("_C::gemma_rmsnorm", mutates_args=())
def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-5,
    enable_pdl: bool=False,
    interweave: bool = False,
    bias: torch.Tensor = None,
    force_sdnn: bool=False
) -> None:
    #print("gemma_rmsnorm wrapper")
    kunlun_ops.gemma_rmsnorm(
        x,
        weight,
        output,
        eps,
        enable_pdl,
        interweave,
        bias,
        force_sdnn
    )


@impl("_C::gemma_rmsnorm", "CUDA")
def gemma_rmsnorm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-5,
    enable_pdl: bool=False,
    interweave: bool = False,
    bias: torch.Tensor = None,
    force_sdnn: bool=False
) -> None:
    #print("gemma_rmsnorm_cuda wrapper")
    kunlun_ops.gemma_rmsnorm(
        x,
        weight,
        output,
        eps,
        enable_pdl,
        interweave,
        bias,
        force_sdnn
    )


def _fake_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-5,
    enable_pdl: bool=False,
    interweave: bool = False,
    bias: torch.Tensor = None,
    force_sdnn: bool=False
):
    # 设置 shape/dtype，但不返回值
    output.fake_shape = x.shape
    output.fake_dtype = x.dtype
    return None


gemma_rmsnorm.register_fake(_fake_gemma_rmsnorm)


@custom_op("_C::split_norm_rope_neox", mutates_args=())
def split_norm_rope_neox(
    q_emb: torch.Tensor,
    k_emb: torch.Tensor,
    v_out: torch.Tensor,
    qkv: torch.Tensor,
    rotary_pos_embedding: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    num_tokens: int,
    max_seqlen: int,
    head_num: int,
    kv_head_num: int,
    head_dim: int,
    rotary_dim: int,
    emb_batch_size: int = 1,
) -> None:
    kunlun_ops.split_norm_rope_neox(
        q_emb,
        k_emb,
        v_out,
        qkv,
        rotary_pos_embedding,
        q_norm_weight,
        k_norm_weight,
        positions,
        num_tokens,
        max_seqlen,
        head_num,
        kv_head_num,
        head_dim,
        rotary_dim,
    )


@impl("_C::split_norm_rope_neox", "CUDA")
def split_norm_rope_neox_cuda(
    q_emb: torch.Tensor,
    k_emb: torch.Tensor,
    v_out: torch.Tensor,
    qkv: torch.Tensor,
    rotary_pos_embedding: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    num_tokens: int,
    max_seqlen: int,
    head_num: int,
    kv_head_num: int,
    head_dim: int,
    rotary_dim: int,
    emb_batch_size: int = 1,
) -> None:
    kunlun_ops.split_norm_rope_neox(
        q_emb,
        k_emb,
        v_out,
        qkv,
        rotary_pos_embedding,
        q_norm_weight,
        k_norm_weight,
        positions,
        num_tokens,
        max_seqlen,
        head_num,
        kv_head_num,
        head_dim,
        rotary_dim,
    )


def _fake_split_norm_rope_neox(
    q_emb: torch.Tensor,
    k_emb: torch.Tensor,
    v_out: torch.Tensor,
    qkv: torch.Tensor,
    rotary_pos_embedding: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    num_tokens: int,
    max_seqlen: int,
    head_num: int,
    kv_head_num: int,
    head_dim: int,
    rotary_dim: int,
    emb_batch_size: int = 1,
):
    q_emb.fake_shape = q_emb.shape
    q_emb.fake_dtype = q_emb.dtype
    k_emb.fake_shape = k_emb.shape
    k_emb.fake_dtype = k_emb.dtype
    v_out.fake_shape = v_out.shape
    v_out.fake_dtype = v_out.dtype
    return None


split_norm_rope_neox.register_fake(_fake_split_norm_rope_neox)

# register fake op impl here
# for torch.dynamo
from torch.library import register_fake

if hasattr(torch.ops.custom_ops, "fc_fusion"):

    @register_fake("custom_ops::fc_fusion")
    def fc_fusion_fake(
        self: torch.Tensor,
        other: torch.Tensor,
        bias: Optional[torch.Tensor],
        self_trans: bool,
        other_trans: bool,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        act: int = 1,
        multi_stream: bool = False,
        out: torch.Tensor,
    ) -> None:
        pass


@custom_op("_C::silu_and_mul", mutates_args=())
def silu_and_mul(
    out: torch.Tensor, x: torch.Tensor, axis: int = -1, turn: bool = True
) -> None:
    kunlun_ops.swiglu(
        x=x,
        y=out,
    )


@impl("_C::silu_and_mul", "CUDA")
def silu_and_mul_cuda(
    out: torch.Tensor, x: torch.Tensor, axis: int = -1, turn: bool = True
) -> None:
    kunlun_ops.swiglu(
        x=x,
        y=out,
    )


def _fake_silu_and_mul(
    out: torch.Tensor, x: torch.Tensor, axis: int = -1, turn: bool = True
):
    return None


silu_and_mul.register_fake(_fake_silu_and_mul)


@custom_op("_C::swigluoai_and_mul", mutates_args=())
def swigluoai_and_mul(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
    axis: int = -1,
    turn: bool = True,
) -> torch.Tensor:
    """PyTorch-native implementation equivalent to forward()."""
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output


@impl("_C::swigluoai_and_mul", "CUDA")
def swigluoai_and_mul_cuda(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
    axis: int = -1,
    turn: bool = True,
) -> torch.Tensor:
    """PyTorch-native implementation equivalent to forward()."""
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output


def _fake_swigluoai_and_mul(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
    axis: int = -1,
    turn: bool = True,
) -> torch.Tensor:
    """PyTorch-native implementation equivalent to forward()."""
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output


swigluoai_and_mul.register_fake(_fake_swigluoai_and_mul)


@custom_op("_C::moe_softmax_topk", mutates_args=())
def moe_softmax_topk(
    x: torch.Tensor,
    normed_score: torch.Tensor,
    topk_index: torch.Tensor,
    block_statistic: torch.Tensor,
    axis: int = -1,
    turn: bool = True,
) -> None:
    kunlun_ops.moe_softmax_topk(x, normed_score, topk_index, block_statistic)


@impl("_C::moe_softmax_topk", "CUDA")
def moe_softmax_topk_cuda(
    x: torch.Tensor,
    normed_score: torch.Tensor,
    topk_index: torch.Tensor,
    block_statistic: torch.Tensor,
    axis: int = -1,
    turn: bool = True,
) -> None:
    kunlun_ops.moe_softmax_topk(x, normed_score, topk_index, block_statistic)


def _fake_moe_softmax_topk(
    x: torch.Tensor,
    normed_score: torch.Tensor,
    topk_index: torch.Tensor,
    block_statistic: torch.Tensor,
    axis: int = -1,
    turn: bool = True,
) -> None:
    return None


moe_softmax_topk.register_fake(_fake_moe_softmax_topk)


@custom_op("_C::moe_ffn_block", mutates_args=())
def moe_ffn_block(
    out: torch.Tensor,
    x: torch.Tensor,
    expert_num: int,
    moe_top_k: int,
    gate_w: torch.Tensor,
    inter_w: torch.Tensor,
    output_w: torch.Tensor,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    expert_group_num: Optional[int] = 0,
    topk_group: Optional[int] = 0,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> None:
    kunlun_ops.moe_ffn_block(
        x=x,
        gate_w=gate_w,
        inter_w=inter_w,
        output_w=output_w,
        expert_num=expert_num,
        moe_top_k=moe_top_k,
        topk_group=topk_group,
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        expert_group_num=expert_group_num,
        out=out,
    )


@impl("_C::moe_ffn_block", "CUDA")
def moe_ffn_block_cuda(
    out: torch.Tensor,
    x: torch.Tensor,
    expert_num: int,
    moe_top_k: int,
    gate_w: torch.Tensor,
    inter_w: torch.Tensor,
    output_w: torch.Tensor,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    expert_group_num: Optional[int] = 0,
    topk_group: Optional[int] = 0,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> None:
    kunlun_ops.moe_ffn_block(
        x=x,
        gate_w=gate_w,
        inter_w=inter_w,
        output_w=output_w,
        expert_num=expert_num,
        moe_top_k=moe_top_k,
        topk_group=topk_group,
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        expert_group_num=expert_group_num,
        out=out,
    )


def _fake_moe_ffn_block(
    out: torch.Tensor,
    x: torch.Tensor,
    expert_num: int,
    moe_top_k: int,
    gate_w: torch.Tensor,
    inter_w: torch.Tensor,
    output_w: torch.Tensor,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    expert_group_num: Optional[int] = 0,
    topk_group: Optional[int] = 0,
):
    return None


moe_ffn_block.register_fake(_fake_moe_ffn_block)


@custom_op("_C::moe_ffn_per_token_block", mutates_args=())
def moe_ffn_per_token_block(
    x: torch.Tensor,
    inter_weight: torch.Tensor,
    inter_scale: torch.Tensor,
    outer_weight: torch.Tensor,
    outer_scale: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    linear_weights: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
    activation: str = "silu",
    output: Optional[torch.Tensor] = None,
    use_expert_parallel: bool = False,
    ep_size: int = 1,
    ep_rank: int = 0,
) -> None:
    kunlun_ops.moe_ffn_per_token_block(
        x=x,
        inter_weight=inter_weight,
        inter_scale=inter_scale,
        outer_weight=outer_weight,
        outer_scale=outer_scale,
        gate_weight=linear_weights,
        expert_num=global_num_experts,
        moe_top_k=top_k,
        act_type=activation,
        use_expert_parallel=use_expert_parallel,
        ep_size=ep_size,
        ep_rank=ep_rank,
        out=output,
    )


@impl("_C::moe_ffn_per_token_block", "CUDA")
def moe_ffn_per_token_block_cuda(
    x: torch.Tensor,
    inter_weight: torch.Tensor,
    inter_scale: torch.Tensor,
    outer_weight: torch.Tensor,
    outer_scale: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    linear_weights: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
    activation: str = "silu",
    output: Optional[torch.Tensor] = None,
    use_expert_parallel: bool = False,
    ep_size: int = 1,
    ep_rank: int = 0,
) -> None:
    kunlun_ops.moe_ffn_per_token_block(
        x=x,
        inter_weight=inter_weight,
        inter_scale=inter_scale,
        outer_weight=outer_weight,
        outer_scale=outer_scale,
        gate_weight=linear_weights,
        expert_num=global_num_experts,
        moe_top_k=top_k,
        act_type=activation,
        use_expert_parallel=use_expert_parallel,
        ep_size=ep_size,
        ep_rank=ep_rank,
        out=output,
    )


def _fake_moe_ffn_per_token_block(
    x: torch.Tensor,
    inter_weight: torch.Tensor,
    inter_scale: torch.Tensor,
    outer_weight: torch.Tensor,
    outer_scale: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    linear_weights: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
    activation: str = "silu",
    output: Optional[torch.Tensor] = None,
    use_expert_parallel: bool = False,
    ep_size: int = 1,
    ep_rank: int = 0
) -> None:
    # Fake implementation can be a no-op or a simple operation
    if output is not None:
        output.copy_(x)  # Example: simply copy input to output

# Register the fake implementation
moe_ffn_per_token_block.register_fake(_fake_moe_ffn_per_token_block)


@custom_op("_C::rotary_embedding", mutates_args=())
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    kunlun_ops.rotary_embedding(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )


@impl("_C::rotary_embedding", "CUDA")
def rotary_embedding_cuda(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    kunlun_ops.rotary_embedding(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )


def _fake_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    return None


rotary_embedding.register_fake(_fake_rotary_embedding)


@custom_op("_C::gemm_I8_I8_bf16_nt", mutates_args=())
def gemm_I8_I8_bf16_nt(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    kunlun_ops.gemm_I8_I8_bf16_nt(
        lhs=(x_q, x_scale), rhs=(weight, weight_scale), out=out
    )
    

@impl("_C::gemm_I8_I8_bf16_nt", "CUDA")
def gemm_I8_I8_bf16_nt_cuda(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    kunlun_ops.gemm_I8_I8_bf16_nt(
        lhs=(x_q, x_scale), rhs=(weight, weight_scale), out=out
    )


def _fake_gemm_I8_I8_bf16_nt(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    return None


gemm_I8_I8_bf16_nt.register_fake(_fake_gemm_I8_I8_bf16_nt)


@custom_op("_C::moe_softmax_topk_norm", mutates_args=())
def moe_softmax_topk_norm(
    x: torch.Tensor,
    normed_score: torch.Tensor,
    topk_index: torch.Tensor,
    block_statistic: torch.Tensor,
    stable: bool = True,
) -> None:
    kunlun_ops.moe_softmax_topk_norm(
        x, normed_score, topk_index, block_statistic, stable
    )


@impl("_C::moe_softmax_topk_norm", "CUDA")
def moe_softmax_topk_norm_cuda(
    x: torch.Tensor,
    normed_score: torch.Tensor,
    topk_index: torch.Tensor,
    block_statistic: torch.Tensor,
    stable: bool = True,
) -> None:
    kunlun_ops.moe_softmax_topk_norm(
        x, normed_score, topk_index, block_statistic, stable
    )


def _fake_moe_softmax_topk_norm(
    x: torch.Tensor,
    normed_score: torch.Tensor,
    topk_index: torch.Tensor,
    block_statistic: torch.Tensor,
    stable: bool = True,
) -> None:
    return None


moe_softmax_topk_norm.register_fake(_fake_moe_softmax_topk_norm)


@custom_op("_C::gen_block_statistic", mutates_args=())
def gen_block_statistic(topk_ids: torch.Tensor, block_statistic: torch.Tensor) -> None:
    kunlun_ops.gen_block_statistic(topk_ids, block_statistic)


@impl("_C::gen_block_statistic", "CUDA")
def gen_block_statistic_cuda(
    topk_ids: torch.Tensor, block_statistic: torch.Tensor
) -> None:
    kunlun_ops.gen_block_statistic(topk_ids, block_statistic)


def fake_gen_block_statistic(
    topk_ids: torch.Tensor, block_statistic: torch.Tensor
) -> None:
    return None


gen_block_statistic.register_fake(fake_gen_block_statistic)


@custom_op("_C::moe_pre_sorted", mutates_args=())
def moe_pre_sorted(
    x: torch.Tensor,
    topk_index: torch.Tensor,
    block_statistic: torch.Tensor,
    moe_expand: torch.Tensor,
    moe_index: torch.Tensor,
    expert_m: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    index_have_neg: bool = False,
) -> None:
    kunlun_ops.moe_pre_sorted(
        x,
        topk_index,
        block_statistic,
        moe_expand,
        moe_index,
        expert_m,
        sorted_tokens_num_lod,
    )


@impl("_C::moe_pre_sorted", "CUDA")
def moe_pre_sorted_cuda(
    x: torch.Tensor,
    topk_index: torch.Tensor,
    block_statistic: torch.Tensor,
    moe_expand: torch.Tensor,
    moe_index: torch.Tensor,
    expert_m: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    index_have_neg: bool = False,
) -> None:
    kunlun_ops.moe_pre_sorted(
        x,
        topk_index,
        block_statistic,
        moe_expand,
        moe_index,
        expert_m,
        sorted_tokens_num_lod,
    )


def fake_moe_pre_sorted(
    x: torch.Tensor,
    topk_index: torch.Tensor,
    block_statistic: torch.Tensor,
    moe_expand: torch.Tensor,
    moe_index: torch.Tensor,
    expert_m: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    index_have_neg: bool = False,
) -> None:
    return None


moe_pre_sorted.register_fake(fake_moe_pre_sorted)


@custom_op("_C::moe_fc", mutates_args=())
def moe_fc(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    moe_topk: int,
    y: torch.Tensor,
    act: Optional[str] = None,
    x_perchannel_max: Optional[torch.Tensor] = None,
    w_perchannel_max: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_w: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    tgemm_type: Optional[str] = None,
    tweight_type: Optional[str] = None,
    scale_n: Optional[int] = 0,
    scale_k: Optional[int] = 0,
    use_pack_int4: Optional[bool] = False,
    sort_mode: Optional[bool] = True,
) -> None:
    kunlun_ops.moe_fc(
        x=x,
        weight=weight,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=moe_topk,
        y=y,
        act=act,
        x_perchannel_max=x_perchannel_max,
        w_perchannel_max=w_perchannel_max,
        topk_ids=topk_ids,
        topk_w=topk_w,
        bias=bias,
        tgemm_type=tgemm_type,
        tweight_type=tweight_type,
        scale_n=scale_n,
        scale_k=scale_k,
        use_pack_int4=use_pack_int4,
        sort_mode=sort_mode,
    )


@impl("_C::moe_fc", "CUDA")
def moe_fc_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    moe_topk: int,
    y: torch.Tensor,
    act: Optional[str] = None,
    x_perchannel_max: Optional[torch.Tensor] = None,
    w_perchannel_max: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_w: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    tgemm_type: Optional[str] = None,
    tweight_type: Optional[str] = None,
    scale_n: Optional[int] = 0,
    scale_k: Optional[int] = 0,
    use_pack_int4: Optional[bool] = False,
    sort_mode: Optional[bool] = True,
) -> None:
    kunlun_ops.moe_fc(
        x=x,
        weight=weight,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=moe_topk,
        y=y,
        act=act,
        x_perchannel_max=x_perchannel_max,
        w_perchannel_max=w_perchannel_max,
        topk_ids=topk_ids,
        topk_w=topk_w,
        bias=bias,
        tgemm_type=tgemm_type,
        tweight_type=tweight_type,
        scale_n=scale_n,
        scale_k=scale_k,
        use_pack_int4=use_pack_int4,
        sort_mode=sort_mode,
    )


def fake_moe_fc(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    moe_topk: int,
    y: torch.Tensor,
    act: Optional[str] = None,
    x_perchannel_max: Optional[torch.Tensor] = None,
    w_perchannel_max: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_w: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    tgemm_type: Optional[str] = None,
    tweight_type: Optional[str] = None,
    scale_n: Optional[int] = 0,
    scale_k: Optional[int] = 0,
    use_pack_int4: Optional[bool] = False,
    sort_mode: Optional[bool] = True,
) -> None:
    return None


moe_fc.register_fake(fake_moe_fc)


@custom_op("_C::moe_post", mutates_args=())
def moe_post(
    x: torch.Tensor,
    moe_index: torch.Tensor,
    normed_scale: torch.Tensor,
    dequant_scale: torch.Tensor,
    y: torch.Tensor,
) -> None:
    kunlun_ops.moe_post(x, moe_index, normed_scale, dequant_scale, y)


@impl("_C::moe_post", "CUDA")
def moe_post_cuda(
    x: torch.Tensor,
    moe_index: torch.Tensor,
    normed_scale: torch.Tensor,
    dequant_scale: torch.Tensor,
    y: torch.Tensor,
) -> None:
    kunlun_ops.moe_post(x, moe_index, normed_scale, dequant_scale, y)


def fake_moe_post(
    x: torch.Tensor,
    moe_index: torch.Tensor,
    normed_scale: torch.Tensor,
    dequant_scale: torch.Tensor,
    y: torch.Tensor,
) -> None:
    return None


moe_post.register_fake(fake_moe_post)


@custom_op("_C::moe_sigmoid_group_topk_norm", mutates_args=())
def moe_sigmoid_group_topk_norm(
    x: torch.Tensor,
    topk_index: torch.Tensor,
    norm_score: torch.Tensor,
    block_static: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    n_group: int,
    topk_group: int,
) -> None:
    kunlun_ops.moe_sigmoid_group_topk_norm(
        x=x,
        norm_score=norm_score,
        topk_index=topk_index,
        block_static=block_static,
        bias=bias,
        n_group=n_group,
        topk_group=topk_group,
        scale=scale,
    )


@impl("_C::moe_sigmoid_group_topk_norm", "CUDA")
def moe_sigmoid_group_topk_norm_cuda(
    x: torch.Tensor,
    topk_index: torch.Tensor,
    norm_score: torch.Tensor,
    block_static: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    n_group: int,
    topk_group: int,
) -> None:
    kunlun_ops.moe_sigmoid_group_topk_norm(
        x=x,
        norm_score=norm_score,
        topk_index=topk_index,
        block_static=block_static,
        bias=bias,
        n_group=n_group,
        topk_group=topk_group,
        scale=scale,
    )


def _fake_moe_sigmoid_group_topk_norm(
    x: torch.Tensor,
    topk_index: torch.Tensor,
    norm_score: torch.Tensor,
    block_static: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    n_group: int,
    topk_group: int,
) -> None:
    return None


moe_sigmoid_group_topk_norm.register_fake(_fake_moe_sigmoid_group_topk_norm)


##################################################
# --------------- awq_dequantize -----------------
##################################################
@custom_op("_C::awq_dequantize", mutates_args=())
def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    quant_type: int = 0,
    align_type: int = 1,
) -> torch.Tensor:
    weight = torch.empty(
        (qweight.shape[0], qweight.shape[1] * 8),
        dtype=torch.float16,
        device=qweight.device,
    )
    group_m = int(qweight.shape[0] / scales.shape[0])
    kunlun_ops.awq_dequantize(
        qweight=qweight,
        scales=scales,
        zeros=zeros,
        weight=weight,
        group_m=group_m,
        quant_type=quant_type,
        align_type=align_type,
    )
    return weight


@impl("_C::awq_dequantize", "CUDA")
def awq_dequantize_cuda(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    quant_type: int = 0,
    align_type: int = 1,
) -> torch.Tensor:
    weight = torch.empty(
        (qweight.shape[0], qweight.shape[1] * 8),
        dtype=torch.float16,
        device=qweight.device,
    )
    group_m = int(qweight.shape[0] / scales.shape[0])
    out = kunlun_ops.awq_dequantize(
        qweight=qweight,
        scales=scales,
        zeros=zeros,
        weight=weight,
        group_m=group_m,
        quant_type=quant_type,
        align_type=align_type,
    )
    return weight


def _fake_awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    quant_type: int = 0,
    align_type: int = 1,
) -> torch.Tensor:
    weight = torch.empty(
        (qweight.shape[0], qweight.shape[1] * 8),
        dtype=torch.float16,
        device=qweight.device,
    )
    return weight


awq_dequantize.register_fake(_fake_awq_dequantize)


##################################################
# ------------------ awq_gemm -------------------
##################################################
@custom_op("_C::awq_gemm", mutates_args=())
def awq_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zeros: torch.Tensor,
    align_type: int = 1,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], qweight.shape[1] * 8), dtype=torch.float16, device=x.device
    )
    group_size = int(qweight.shape[0] / scale.shape[0])
    kunlun_ops.awq_gemm(
        x=x,
        w=qweight,
        scale=scale,
        zeros=zeros,
        out=out,
        align_type=align_type,
        group_size=group_size,
    )
    return out


@impl("_C::awq_gemm", "CUDA")
def awq_gemm_cuda(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zeros: torch.Tensor,
    align_type: int = 1,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], qweight.shape[1] * 8), dtype=torch.float16, device=x.device
    )
    group_size = int(qweight.shape[0] / scale.shape[0])
    kunlun_ops.awq_gemm(
        x=x,
        w=qweight,
        scale=scale,
        zeros=zeros,
        out=out,
        align_type=align_type,
        group_size=group_size,
    )
    return out


def _fake_awq_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zeros: torch.Tensor,
    align_type: int = 1,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], qweight.shape[1] * 8), dtype=torch.float16, device=x.device
    )
    return out


awq_gemm.register_fake(_fake_awq_gemm)


##################################################
# ---------------- gptq_shuffle ------------------
##################################################
@custom_op("_C::gptq_shuffle", mutates_args=())
def gptq_shuffle(
    q_weight: torch.Tensor,
    q_perm: torch.Tensor,
    bit: int,
) -> None:
    kunlun_ops.gptq_shuffle(weight=q_weight, perm=q_perm, bit=bit)


@impl("_C::gptq_shuffle", "CUDA")
def gptq_shuffle_cuda(
    q_weight: torch.Tensor,
    q_perm: torch.Tensor,
    bit: int,
) -> None:
    kunlun_ops.gptq_shuffle(weight=q_weight, perm=q_perm, bit=bit)


def _fake_gptq_shuffle(
    q_weight: torch.Tensor,
    q_perm: torch.Tensor,
    bit: int,
) -> None:
    return None


gptq_shuffle.register_fake(_fake_gptq_shuffle)


##################################################
# ------------- concat_and_cache_mla -------------
##################################################
@custom_op("_C::concat_and_cache_mla", mutates_args=())
def concat_and_cache_mla(
    kv_c: torch.Tensor,  # [num_tokens, kv_lora_rank]
    k_pe: torch.Tensor,  # [num_tokens, pe_dim]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, (kv_lora_rank + pe_dim)]
    slot_mapping: torch.Tensor,  # [num_tokens] or [num_actual_tokens]
) -> None:
    kunlun_ops.concat_and_cache_mla(
        kv_c=kv_c,
        k_pe=k_pe,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
    )


@impl("_C::concat_and_cache_mla", "CUDA")
def concat_and_cache_mla_cuda(
    kv_c: torch.Tensor,  # [num_tokens, kv_lora_rank]
    k_pe: torch.Tensor,  # [num_tokens, pe_dim]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, (kv_lora_rank + pe_dim)]
    slot_mapping: torch.Tensor,  # [num_tokens] or [num_actual_tokens]
) -> None:
    kunlun_ops.concat_and_cache_mla(
        kv_c=kv_c,
        k_pe=k_pe,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
    )


def _fake_concat_and_cache_mla(
    kv_c: torch.Tensor,  # [num_tokens, kv_lora_rank]
    k_pe: torch.Tensor,  # [num_tokens, pe_dim]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, (kv_lora_rank + pe_dim)]
    slot_mapping: torch.Tensor,  # [num_tokens] or [num_actual_tokens]
) -> None:
    return None


concat_and_cache_mla.register_fake(_fake_concat_and_cache_mla)


######################################################
# -------------- scaled_int8_quant -------------------
######################################################
@custom_op("_C::scaled_int8_quant", mutates_args=())
def scaled_int8_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    static = False
    x_q = torch.empty_like(x, dtype=torch.int8, device=x.device)
    if scale is not None:  # static
        static = True
        torch.ops.xspeedgate_ops.static_scaled_int8_quant(x_q, x, scale, azp)
    else:  # dynamic
        scale = torch.empty(
            (x.numel() // x.shape[-1], 1), device=x.device, dtype=torch.float32
        )
        azp = None if symmetric else torch.empty_like(scale, dtype=torch.int32)
        if symmetric:
            # NOTE: For quant2d ops, scale represents max.
            kunlun_ops.quant2d(x=x.contiguous(), y=x_q, max=scale, force_sdnn=True)
        else:
            torch.ops.xspeedgate_ops.dynamic_scaled_int8_quant(
                x_q, x.contiguous(), scale, azp
            )
    return x_q, scale, azp, static


@impl("_C::scaled_int8_quant", "CUDA")
def scaled_int8_quant_cuda(
    x: torch.Tensor,
    scale: torch.Tensor,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    static = False
    x_q = torch.empty_like(x, dtype=torch.int8, device=x.device)
    if scale is not None:  # static
        static = True
        torch.ops.xspeedgate_ops.static_scaled_int8_quant(x_q, x, scale, azp)
    else:  # dynamic
        scale = torch.empty(
            (x.numel() // x.shape[-1], 1), device=x.device, dtype=torch.float32
        )
        azp = None if symmetric else torch.empty_like(scale, dtype=torch.int32)
        if symmetric:
            # NOTE: For quant2d ops, scale represents max.
            kunlun_ops.quant2d(x=x.contiguous(), y=x_q, max=scale, force_sdnn=True)
        else:
            torch.ops.xspeedgate_ops.dynamic_scaled_int8_quant(
                x_q, x.contiguous(), scale, azp
            )
    return x_q, scale, azp, static


def _fake_scaled_int8_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    x_q = torch.empty_like(x, dtype=torch.int8, device=x.device)
    scale = torch.empty(
        (x.numel() // x.shape[-1], 1), device=x.device, dtype=torch.float32
    )
    azp = None if symmetric else torch.empty_like(scale, dtype=torch.int32)
    return x_q, scale, azp, False


scaled_int8_quant.register_fake(_fake_scaled_int8_quant)


######################################################
# ---------------- cutlass_scaled_mm -----------------
######################################################
@custom_op("_C::cutlass_scaled_mm", mutates_args=())
def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    torch.ops.xspeedgate_ops.cutlass_scaled_mm(
        out, a.contiguous(), b.contiguous(), scale_a, scale_b, bias
    )
    return out


@impl("_C::cutlass_scaled_mm", "CUDA")
def cutlass_scaled_mm_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    torch.ops.xspeedgate_ops.cutlass_scaled_mm(
        out, a.contiguous(), b.contiguous(), scale_a, scale_b, bias
    )
    return out


def fake_cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)


cutlass_scaled_mm.register_fake(fake_cutlass_scaled_mm)


######################################################
# ------------ cutlass_scaled_mm_azp -----------------
######################################################
@custom_op("_C::cutlass_scaled_mm_azp", mutates_args=())
def cutlass_scaled_mm_azp(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    torch.ops.xspeedgate_ops.cutlass_scaled_mm_azp(
        out, a.contiguous(), b.contiguous(), scale_a, scale_b, azp_adj, azp, bias
    )
    return out


@impl("_C::cutlass_scaled_mm_azp", "CUDA")
def cutlass_scaled_mm_azp_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    torch.ops.xspeedgate_ops.cutlass_scaled_mm_azp(
        out, a.contiguous(), b.contiguous(), scale_a, scale_b, azp_adj, azp, bias
    )
    return out


def fake_cutlass_scaled_mm_azp(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)


cutlass_scaled_mm_azp.register_fake(fake_cutlass_scaled_mm_azp)


##################################################
# ------------------ matmul ---------------------
##################################################
@custom_op("_C::matmul", mutates_args=())
def matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    out_dtype: torch.dtype,
    x_trans: bool = False,
    w_trans: bool = True,
    alpha: float = 1.0,
    beta: float = 0.0,
    bias: torch.Tensor = None,
    x_max: torch.Tensor = None,
    w_max: torch.Tensor = None,
    x_pc_max: torch.Tensor = None,
    w_pc_max: torch.Tensor = None,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], w.shape[0] if w_trans else w.shape[1]),
        dtype=out_dtype,
        device=x.device,
    )
    kunlun_ops.matmul(
        x=x.contiguous(),
        w=w.contiguous(),
        out=out,
        x_trans=x_trans,
        w_trans=w_trans,
        alpha=alpha,
        beta=beta,
        bias=bias,
        x_max=x_max,
        w_max=w_max,
        x_pc_max=x_pc_max,
        w_pc_max=w_pc_max,
    )
    return out


@impl("_C::matmul", "CUDA")
def matmul_cuda(
    x: torch.Tensor,
    w: torch.Tensor,
    out_dtype: torch.dtype,
    x_trans: bool = False,
    w_trans: bool = True,
    alpha: float = 1.0,
    beta: float = 0.0,
    bias: torch.Tensor = None,
    x_max: torch.Tensor = None,
    w_max: torch.Tensor = None,
    x_pc_max: torch.Tensor = None,
    w_pc_max: torch.Tensor = None,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], w.shape[0] if w_trans else w.shape[1]),
        dtype=out_dtype,
        device=x.device,
    )
    kunlun_ops.matmul(
        x=x.contiguous(),
        w=w.contiguous(),
        out=out,
        x_trans=x_trans,
        w_trans=w_trans,
        alpha=alpha,
        beta=beta,
        bias=bias,
        x_max=x_max,
        w_max=w_max,
        x_pc_max=x_pc_max,
        w_pc_max=w_pc_max,
    )
    return out


def _fake_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    out_dtype: torch.dtype,
    x_trans: bool = False,
    w_trans: bool = True,
    alpha: float = 1.0,
    beta: float = 0.0,
    bias: torch.Tensor = None,
    x_max: torch.Tensor = None,
    w_max: torch.Tensor = None,
    x_pc_max: torch.Tensor = None,
    w_pc_max: torch.Tensor = None,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], w.shape[0] if w_trans else w.shape[1]),
        dtype=out_dtype,
        device=x.device,
    )


matmul.register_fake(_fake_matmul)


##################################################
# ------------------- quant2d --------------------
##################################################
@custom_op("_C::quant2d", mutates_args=())
def quant2d(
    x: torch.Tensor,
    x_q: torch.Tensor,
    max: torch.Tensor,
    force_sdnn: bool = False,
) -> None:
    kunlun_ops.quant2d(
        x=x,
        y=x_q,
        max=max,
        force_sdnn=force_sdnn,
    )


@impl("_C::quant2d", "CUDA")
def quant2d_cuda(
    x: torch.Tensor,
    x_q: torch.Tensor,
    max: torch.Tensor,
    force_sdnn: bool = False,
) -> None:
    kunlun_ops.quant2d(
        x=x,
        y=x_q,
        max=max,
        force_sdnn=force_sdnn,
    )


def _fake_quant2d(
    x: torch.Tensor,
    x_q: torch.Tensor,
    max: torch.Tensor,
    force_sdnn: bool = False,
) -> None:
    return None


quant2d.register_fake(_fake_quant2d)
