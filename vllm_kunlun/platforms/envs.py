# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_MULTI_LOGPATH : str = "./log",
    ENABLE_VLLM_MULTI_LOG : bool = False,
    ENABLE_VLLM_INFER_HOOK : bool = False,
    ENABLE_VLLM_OPS_HOOK : bool = False,
    ENABLE_VLLM_MODULE_HOOK : bool = False

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
    "VLLM_MULTI_LOGPATH":
    lambda: os.environ.get("VLLM_MULTI_LOGPATH", "./logs"),

    # turn on / off multi-log of multi nodes & multi cards 
    "ENABLE_VLLM_MULTI_LOG":
    lambda: (os.environ.get("ENABLE_VLLM_MULTI_LOG", "False").lower() in 
             ("true", "1")),

    # turn on / off XVLLM infer stage log ability 
    "ENABLE_VLLM_INFER_HOOK":
    lambda: (os.environ.get("ENABLE_VLLM_INFER_HOOK", "False").lower() in
            ("true", "1")),

    # turn on / off XVLLM infer_ops log ability 
    "ENABLE_VLLM_OPS_HOOK":
    lambda: (os.environ.get("ENABLE_VLLM_OPS_HOOK", "False").lower() in
            ("true", "1")),

    "ENABLE_VLLM_MODULE_HOOK":
    lambda: (os.environ.get("ENABLE_VLLM_MODULE_HOOK", "False").lower() in
            ("true", "1")),

    # fuse sorted op with fused_moe kernel
    "ENABLE_VLLM_MOE_FC_SORTED":
    lambda: (os.environ.get("ENABLE_VLLM_MOE_FC_SORTED", "False").lower() in 
             ("true", "1")),

    # enable custom dpsk scaling rope
    "ENABLE_CUSTOM_DPSK_SCALING_ROPE":
    lambda: (os.environ.get("ENABLE_CUSTOM_DPSK_SCALING_ROPE", "False").lower() in 
             ("true", "1")),

    # fuse qkv split & qk norm & qk rope
    # only works for qwen3 dense and qwen3 moe models
    "ENABLE_VLLM_FUSED_QKV_SPLIT_NORM_ROPE":
    lambda: (os.environ.get("ENABLE_VLLM_FUSED_QKV_SPLIT_NORM_ROPE", "False").lower() in 
             ("true", "1")),

    # use int8 bmm
    "VLLM_KUNLUN_ENABLE_INT8_BMM":
    lambda: (os.environ.get("VLLM_KUNLUN_ENABLE_INT8_BMM", "False").lower() in 
             ("true", "1")),
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
