import inspect
import logging
import os
import sys
import time
import typing
from typing import Callable, List, Optional, get_args, get_origin

import torch
import vllm
from torch.library import Library
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.utils.torch_utils import weak_ref_tensor

import vllm_kunlun.platforms.envs as xenvs

logger = logging.getLogger("vllm_kunlun")

_VLLM_UTILS_WRAPPER_IMPORTED_AT = time.time()

logger.info(
    f"[KunlunPlugin] vllm_utils imported OK. " f"PID={os.getpid()} file={__file__}"
)


def redirect_output():
    """
    重定向输出到指定目录，并将日志文件命名为pp=0_rank=X或pp=1_rank=X。
    如果是第一个进程组的第一个进程，则使用pp=0；否则使用pp=1。

    Args:
        无参数。

    Returns:
        无返回值，直接修改sys.stdout和sys.stderr的文件描述符。
    """
    from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank

    rank = get_tensor_model_parallel_rank()
    dir_path = xenvs.VLLM_MULTI_LOGPATH
    os.makedirs(dir_path, exist_ok=True)
    if get_pp_group().is_first_rank:
        log_file = os.path.join(dir_path, f"pp=0_rank={rank}.log")
    else:
        log_file = os.path.join(dir_path, f"pp=1_rank={rank}.log")
    fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())
    os.close(fd)


def multi_log_monkey_patch(func):
    """
    多次打印日志的猴子补丁函数，用于测试日志重定向功能。
    该函数会在每次调用被补丁的函数时打印一条日志信息。

    Args:
        func (function): 需要被补丁的原始函数。

    Returns:
        function: 返回一个包装后的新函数，每次调用都会打印一条日志信息。
    """

    def wrapper(*args, **kwargs):
        print("[monkey patch] ensure_model_parallel_initialized")
        func(*args, **kwargs)
        redirect_output()

    return wrapper


# if os.environ.get("VLLM_MULTI_LOG", "0") == "1":
if xenvs.ENABLE_VLLM_MULTI_LOG:
    print("ENABLE_VLLM_MULTI_LOG monkey--------")
    vllm.distributed.ensure_model_parallel_initialized = multi_log_monkey_patch(
        vllm.distributed.ensure_model_parallel_initialized
    )


class StageHookPre(object):
    def __call__(self, *args, **kwargs):
        """
            在调用对象时，会自动执行此方法。
        如果当前的attention metadata不为None，并且已经处理了一个token，则打印"Per Token Start"；否则打印"First Token Start"。

        Args:
            args (tuple, optional): 可变参数，默认为空元组。
            kwargs (dict, optional): 关键字参数，默认为空字典。

        Returns:
            None: 无返回值。
        """
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None:
            if attn_metadata.num_decode_tokens == 0:
                print("First Token Start", flush=True)
            else:
                print("Per Token Start", flush=True)


class StageHookPost(object):
    def __call__(self, *args, **kwargs):
        """
            如果当前上下文中的attention metadata不为None，并且num_decode_tokens等于0，则打印"First Token End"。
        否则，打印"Per Token End"。

        Args:
            args (Tuple[Any]): 可变长度参数列表，无用参数传入。
            kwargs (Dict[str, Any]): 字典类型的关键字参数，无用参数传入。

        Returns:
            None: 该函数没有返回值。
        """
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None:
            if attn_metadata.num_decode_tokens == 0:
                print("First Token End", flush=True)
            else:
                print("Per Token End", flush=True)


class ModuleLoggingHookPre(object):
    def __init__(self):
        """
            初始化函数，用于初始化缩进列表和名称列表。
        缩进列表用于存储每一行的缩进信息，名称列表用于存储每一个变量或函数的名称。
        """
        self.indent_list = list()
        self.indent_list.append("")
        self.name_list = list()

    def __call__(self, *args, **kwargs):
        """
            重写了 __call__ 方法，用于在类实例化时调用。
        将当前缩进增加一个 Tab，并记录当前类名称。
        打印开始信息，flush=True 表示立即输出到控制台。

        Args:
            args (tuple): 传入的参数列表，第一个元素是类实例。
            kwargs (dict): 传入的关键字参数列表，不使用。

        Returns:
            None.
        """
        self.indent_list.append(self.indent_list[-1] + "\t")
        self.name_list.append(args[0].__class__.__module__ + args[0].__class__.__name__)
        print(self.indent_list[-1] + self.name_list[-1] + " Start", flush=True)


class ModuleLoggingHookPost(object):
    def __init__(self, indent_list, name_list):
        """
            初始化函数，设置缩进列表和名称列表。

        Args:
            indent_list (List[str]): 包含每个节点的缩进字符串的列表，索引从0开始。
            name_list (List[str]): 包含每个节点的名称字符串的列表，索引从0开始。
            注意：缩进列表和名称列表应该有相同长度，否则会导致错误。

        Returns:
            None. 无返回值，直接修改了类实例的属性。
        """
        self.indent_list = indent_list
        self.name_list = name_list

    def __call__(self, *args, **kwargs):
        """
            当调用对象时，输出模块结束信息。
        参数：*args、**kwargs - 可变长度的位置参数列表和关键字参数字典，未使用。
        返回值：None，无返回值。
        """
        print(self.indent_list[-1] + self.name_list[-1] + " Module End", flush=True)
        self.indent_list.pop()
        self.name_list.pop()


# if os.environ.get("ENABLE_VLLM_MODULE_HOOK", "0") == "1":
if xenvs.ENABLE_VLLM_MODULE_HOOK:
    from torch.nn.modules.module import (
        register_module_forward_hook,
        register_module_forward_pre_hook,
    )

    module_logging_hook_pre = ModuleLoggingHookPre()
    module_logging_hook_post = ModuleLoggingHookPost(
        module_logging_hook_pre.indent_list, module_logging_hook_pre.name_list
    )
    register_module_forward_pre_hook(module_logging_hook_pre)
    register_module_forward_hook(module_logging_hook_post)
else:
    module_logging_hook_pre = None
    module_logging_hook_post = None


class LoggingDispatchMode(TorchDispatchMode):
    def __init__(self):
        """
            初始化函数，用于初始化类的属性和方法。
        在此处可以进行一些初始化操作，例如设置默认值等。
        """
        super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        Override the default dispatch behavior of torch.nn.Module.
            This function will be called before and after each method call on this module.
            It can be used to log information about the method calls.

            Args:
                func (function): The function that is being called on this module.
                types (Tuple[str]): A tuple of strings representing the type signatures of the arguments.
                    See torch.types for more details.
                args (Tuple[Any], optional): The positional arguments passed to the function. Defaults to ().
                kwargs (Dict[str, Any], optional): The keyword arguments passed to the function. Defaults to {}.

            Returns:
                Any: The result returned by the function.
        """
        global module_logging_hook_pre
        if module_logging_hook_pre is not None:
            indent = module_logging_hook_pre.indent_list[-1]
        else:
            indent = "\t"
        print(indent + "{} calling".format(func), flush=True)
        result = func(*args, **(kwargs or {}))
        print(indent + "{} called".format(func), flush=True)

        return result


class CUDAGraphInnerWatcher(TorchDispatchMode):

    def __init__(self, name_list):
        """
            初始化函数，将传入的名称列表保存到类属性中。
        同时创建一个字典来记录已经追踪过的张量。

        Args:
            name_list (List[str]): 包含需要追踪的张量名称的列表。

        Returns:
            None.
        """
        self.name_list = name_list
        self.traced_tensor = dict()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        Override the default dispatch behavior of PyTorch tensors to track
        the tracing process. If the result of a function call is a tensor on CUDA,
        it will be added to the traced_tensor dictionary with the name of the function.

        Args:
            func (Callable): The function to be called.
            types (Tuple[Type]): The type hints of the function.
            args (Tuple[Any], optional): Positional arguments for the function. Defaults to ().
            kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for the function. Defaults to None.

        Returns:
            Any: The result of the function call.
        """
        result = func(*args, **(kwargs or {}))
        if isinstance(result, torch.Tensor) and result.is_cuda:
            if func._name in self.name_list:
                self.traced_tensor[func._name] = weak_ref_tensor(result)
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
            清空 traced_tensor 和 name_list，并调用父类的 __exit__ 方法。

        Args:
            exc_type (Optional[Type[BaseException]]): 异常类型，默认为 None。
            exc_val (Optional[BaseException]): 异常值，默认为 None。
            exc_tb (Optional[TracebackType]): Traceback 对象，默认为 None。

        Returns:
            None.
        """
        for name, value in self.traced_tensor.items():
            print(name, value)
        self.traced_tensor.clear()
        self.name_list.clear()
        super(CUDAGraphInnerWatcher, self).__exit__(exc_type, exc_val, exc_tb)


# def patch_annotations_for_schema(func):
#     sig = inspect.signature(func)
#     new_params = []
#     for name, param in sig.parameters.items():
#         anno = param.annotation
#         if anno == list[int]:
#             anno = typing.List[int]
#         new_params.append(param.replace(annotation=anno))
#     new_sig = sig.replace(parameters=new_params)
#     func.__signature__ = new_sig
#     return func


def patch_annotations_for_schema(func):
    """
    运行时替换函数签名里的 list[int]、Optional[list[int]] 为 typing.List[int] / Optional[typing.List[int]]
    让 torch.library.infer_schema 能识别
    """
    sig = inspect.signature(func)
    new_params = []

    for name, param in sig.parameters.items():
        ann = param.annotation

        # 如果是 Optional[T]
        if get_origin(ann) is typing.Union and type(None) in get_args(ann):
            inner_type = [a for a in get_args(ann) if a is not type(None)][0]
            if get_origin(inner_type) is list:  # Optional[list[int]]
                inner_args = get_args(inner_type)
                new_ann = Optional[List[inner_args[0] if inner_args else typing.Any]]
                param = param.replace(annotation=new_ann)

        # 如果是直接 list[int]
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
    mutates_args: list[str],
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

    import torch.library

    if hasattr(torch.library, "infer_schema"):
        patch_annotations_for_schema(op_func)
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
