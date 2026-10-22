"""vllm_utils_wrapper.py"""

import socket
from types import SimpleNamespace
from typing import Any, Union

import torch
import vllm.distributed.parallel_state as parallel_state
import vllm.envs as envs
import vllm.utils as _orig

try:
    import vllm_kunlun._kunlun  # noqa: F401
except ImportError as e:
    try:
        from . import _kunlun  # noqa: F401, F403
    except ImportError:
        print(f"Warning: Failed to load vllm_kunlun native extension: {e}")


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


vllm_port = envs.VLLM_PORT


def _get_open_port() -> int:
    """Return a port this process just verified as bindable.

    The previous version returned ``vllm_port + 1`` -- a port it had never
    bound -- and crashed outright when VLLM_PORT was unset (``bind(("", None))``
    raises TypeError). Scan upward from VLLM_PORT instead, which is also what
    upstream ``vllm.utils.network_utils._get_open_port`` does, so concurrent
    data-parallel engine cores settle on distinct ports.
    """
    global vllm_port
    if vllm_port is not None:
        port = vllm_port
        for _ in range(1000):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    vllm_port = port + 1
                    return port
            except OSError:
                port += 1
        raise RuntimeError(
            "no free port found after 1000 attempts from %d" % vllm_port
        )
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError("no free ephemeral port available")


_wrapped = SimpleNamespace(**_orig.__dict__)
_wrapped.weak_ref_tensor = vllm_kunlun_weak_ref_tensor
_wrapped.weak_ref_tensors = vllm_kunlun_weak_ref_tensors
_wrapped._get_open_port = _get_open_port

import sys  # noqa: E402

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
