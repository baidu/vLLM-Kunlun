import inspect
import typing
from typing import Callable, List, Optional, get_args, get_origin

import torch
import vllm.utils.torch_utils as torch_utils_orig
from torch.library import Library


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


torch_utils_orig.direct_register_custom_op = direct_register_custom_op
