import inspect
import types
import typing
from typing import Any, Callable, List, Optional, get_args, get_origin

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


def _normalize_ann(ann):
    """Convert PEP604 `T | None` to typing.Union[T, None] for torch infer_schema."""
    if isinstance(ann, types.UnionType):  # Python 3.10: int | None
        args = get_args(ann)
        return typing.Union[args]  # type: ignore[misc]
    return ann


def patch_annotations_for_schema(func):
    """patch_annotations_for_schema"""
    sig = inspect.signature(func)

    # patch return annotation too
    ret = _normalize_ann(sig.return_annotation)
    sig = sig.replace(return_annotation=ret)

    new_params = []
    for name, param in sig.parameters.items():
        ann = _normalize_ann(param.annotation)

        origin = get_origin(ann)
        args = get_args(ann)

        # Optional[T] (typing.Union[T, NoneType])
        if origin is typing.Union and type(None) in args:
            inner = [a for a in args if a is not type(None)][0]
            inner_origin = get_origin(inner)
            inner_args = get_args(inner)

            # Optional[list[T]]
            if inner_origin is list:
                new_ann = Optional[List[inner_args[0] if inner_args else Any]]
                ann = new_ann

        # list[T] -> typing.List[T]
        elif origin is list:
            ann = List[args[0] if args else Any]

        new_params.append(param.replace(annotation=ann))

    func.__signature__ = sig.replace(parameters=new_params)
    return func


torch_utils_orig.direct_register_custom_op = direct_register_custom_op
