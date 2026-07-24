# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_MULTI_LOGPATH: str = ("./log",)
    ENABLE_VLLM_MULTI_LOG: bool = (False,)
    ENABLE_VLLM_INFER_HOOK: bool = (False,)
    ENABLE_VLLM_OPS_HOOK: bool = (False,)
    ENABLE_VLLM_MODULE_HOOK: bool = False
    VLLM_KUNLUN_DISABLE_MOE_PRE_QUANT: bool = False
    VLLM_KUNLUN_DISABLE_ROUTER_MOE_GATE: bool = False
    VLLM_KUNLUN_DISABLE_XSPEED_SHARED_GATE: bool = False
    VLLM_KUNLUN_DISABLE_FUSED_SWIGLU_QUANT: bool = False
    VLLM_KUNLUN_DISABLE_FUSED_DENSE_SWIGLU_QUANT: bool = False
    VLLM_KUNLUN_DISABLE_FUSED_NORM_QUANT: bool = False
    VLLM_KUNLUN_DISABLE_GRAMMAR_BITMASK: bool = False
    VLLM_KUNLUN_DISABLE_MSA_SHORT_PREFILL_DENSE: bool = False


def maybe_convert_int(value: Optional[str]) -> Optional[int]:
    """
    如果值是None，则返回None；否则将字符串转换为整数并返回。

    Args:
        value (Optional[str], optional): 要转换的可选字符串. Defaults to None.

    Returns:
        Optional[int]: 如果值是None，则返回None；否则将字符串转换为整数并返回.
    """
    if value is None:
        return None
    return int(value)


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

xvllm_environment_variables: dict[str, Callable[[], Any]] = {
    # path to the logs of redirect-output, abstrac of related are ok
    "VLLM_MULTI_LOGPATH": lambda: os.environ.get("VLLM_MULTI_LOGPATH", "./logs"),
    # turn on / off multi-log of multi nodes & multi cards
    "ENABLE_VLLM_MULTI_LOG": lambda: (
        os.environ.get("ENABLE_VLLM_MULTI_LOG", "False").lower() in ("true", "1")
    ),
    # turn on / off XVLLM infer stage log ability
    "ENABLE_VLLM_INFER_HOOK": lambda: (
        os.environ.get("ENABLE_VLLM_INFER_HOOK", "False").lower() in ("true", "1")
    ),
    # turn on / off XVLLM infer_ops log ability
    "ENABLE_VLLM_OPS_HOOK": lambda: (
        os.environ.get("ENABLE_VLLM_OPS_HOOK", "False").lower() in ("true", "1")
    ),
    "ENABLE_VLLM_MODULE_HOOK": lambda: (
        os.environ.get("ENABLE_VLLM_MODULE_HOOK", "False").lower() in ("true", "1")
    ),
    # fuse sorted op with fused_moe kernel
    "ENABLE_VLLM_MOE_FC_SORTED": lambda: (
        os.environ.get("ENABLE_VLLM_MOE_FC_SORTED", "False").lower() in ("true", "1")
    ),
    # enable custom dpsk scaling rope
    "ENABLE_CUSTOM_DPSK_SCALING_ROPE": lambda: (
        os.environ.get("ENABLE_CUSTOM_DPSK_SCALING_ROPE", "False").lower()
        in ("true", "1")
    ),
    # fuse qkv split & qk norm & qk rope
    # only works for qwen3 dense and qwen3 moe models
    "ENABLE_VLLM_FUSED_QKV_SPLIT_NORM_ROPE": lambda: (
        os.environ.get("ENABLE_VLLM_FUSED_QKV_SPLIT_NORM_ROPE", "False").lower()
        in ("true", "1")
    ),
    # use int8 bmm
    "VLLM_KUNLUN_ENABLE_INT8_BMM": lambda: (
        os.environ.get("VLLM_KUNLUN_ENABLE_INT8_BMM", "False").lower() in ("true", "1")
    ),
    # use the legacy BF16-dispatch-then-quantize MoE input path
    "VLLM_KUNLUN_DISABLE_MOE_PRE_QUANT": lambda: (
        os.environ.get("VLLM_KUNLUN_DISABLE_MOE_PRE_QUANT", "False").lower()
        in ("true", "1")
    ),
    # use the original BF16->FP32 cast + ReplicatedLinear router path
    "VLLM_KUNLUN_DISABLE_ROUTER_MOE_GATE": lambda: (
        os.environ.get(
            "VLLM_KUNLUN_DISABLE_ROUTER_MOE_GATE", "False"
        ).lower()
        in ("true", "1")
    ),
    # restore the legacy routed-topk + explicit shared-route assembly
    "VLLM_KUNLUN_DISABLE_XSPEED_SHARED_GATE": lambda: (
        os.environ.get(
            "VLLM_KUNLUN_DISABLE_XSPEED_SHARED_GATE", "False"
        ).lower()
        in ("true", "1")
    ),
    # use the original BF16 SwigluOAI output followed by a separate quant2d
    "VLLM_KUNLUN_DISABLE_FUSED_SWIGLU_QUANT": lambda: (
        os.environ.get(
            "VLLM_KUNLUN_DISABLE_FUSED_SWIGLU_QUANT", "False"
        ).lower()
        in ("true", "1")
    ),
    # use dense BF16 SwigluOAI followed by the linear kernel's quant2d
    "VLLM_KUNLUN_DISABLE_FUSED_DENSE_SWIGLU_QUANT": lambda: (
        os.environ.get(
            "VLLM_KUNLUN_DISABLE_FUSED_DENSE_SWIGLU_QUANT", "False"
        ).lower()
        in ("true", "1")
    ),
    # use Gemma add-RMSNorm followed by a separate dynamic INT8 quantization
    "VLLM_KUNLUN_DISABLE_FUSED_NORM_QUANT": lambda: (
        os.environ.get(
            "VLLM_KUNLUN_DISABLE_FUSED_NORM_QUANT", "False"
        ).lower()
        in ("true", "1")
    ),
    # restore xgrammar's torch-native structured-output bitmask path
    "VLLM_KUNLUN_DISABLE_GRAMMAR_BITMASK": lambda: (
        os.environ.get(
            "VLLM_KUNLUN_DISABLE_GRAMMAR_BITMASK", "False"
        ).lower()
        in ("true", "1")
    ),
    # keep all-block (<= sparse top-k width) prefill on the sparse pipeline
    "VLLM_KUNLUN_DISABLE_MSA_SHORT_PREFILL_DENSE": lambda: (
        os.environ.get(
            "VLLM_KUNLUN_DISABLE_MSA_SHORT_PREFILL_DENSE", "False"
        ).lower()
        in ("true", "1")
    ),
}

# end-env-vars-definition


def __getattr__(name: str):
    """
    当调用不存在的属性时，该函数被调用。如果属性是xvllm_environment_variables中的一个，则返回相应的值。否则引发AttributeError异常。

    Args:
        name (str): 要获取的属性名称。

    Raises:
        AttributeError (Exception): 如果属性不是xvllm_environment_variables中的一个，则会引发此异常。

    Returns:
        Any, optional: 如果属性是xvllm_environment_variables中的一个，则返回相应的值；否则返回None。
    """
    # lazy evaluation of environment variables
    if name in xvllm_environment_variables:
        return xvllm_environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    返回一个包含所有可见的变量名称的列表。

    返回值（list）：一个包含所有可见的变量名称的列表，这些变量是通过`xvllm_environment_variables`字典定义的。

    Returns:
        List[str]: 一个包含所有可见的变量名称的列表。
                   这些变量是通过`xvllm_environment_variables`字典定义的。
    """
    return list(xvllm_environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in xvllm_environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
