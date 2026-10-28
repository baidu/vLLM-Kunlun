# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""qwen4_exp 需要、但随包 vLLM 里还没有的上游辅助定义。

qwen4_exp 是跟着上游一个尚未合入的 PR 一起下来的模型，它引用了几个只存在于
那个 PR 里的辅助函数与类型别名。为了不修改 vllm 包，这里按上游语义原样回填，
模型侧统一从本模块 import：

* ``is_model_fused_shared_expert_compatible`` —— 上游
  ``vllm.model_executor.layers.fused_moe.utils``
* ``maybe_fuse_shared_experts`` —— 上游 ``vllm.model_executor.models.utils``
* ``short_conv_state_shape`` —— 上游同名 classmethod 增加了 ``num_spec``
  预留长度；随包版本没有这个入参，这里按上游公式重算，卷积状态的
  DS/SD 朝向仍然复用 vllm 的 ``_orient_conv_shape``，保证与随包实现一致。

``MambaStateCopyFunc`` / ``MambaStateCopyFuncs`` / ``MambaStateCopyFuncsByType``
随包 vLLM 0.25.1 已经有了（``vllm/model_executor/layers/mamba/mamba_utils.py``），
这里只做转发、不重新声明——同名但不同一的 TypeAlias 会让人误以为随包版本缺
这几个名字。

一旦随包 vLLM 升级到带这些定义的版本，本模块应删除，模型侧改回直接 import
上游位置。
"""

from collections.abc import Iterable, Iterator

import torch
from vllm.distributed import divide
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncs,
    MambaStateCopyFuncsByType,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

__all__ = [
    "MambaAttentionBackendEnum",
    "MambaStateCopyFunc",
    "MambaStateCopyFuncs",
    "MambaStateCopyFuncsByType",
]


def short_conv_state_shape(
    *,
    tp_world_size: int,
    intermediate_size: int,
    conv_kernel: int,
    num_spec: int = 0,
) -> tuple[tuple[int, int]]:
    """短卷积状态的形状，``num_spec`` 是投机解码额外预留的长度。"""
    conv_dim = divide(intermediate_size, tp_world_size)
    conv_state_shape = MambaStateShapeCalculator._orient_conv_shape(
        conv_dim, conv_kernel - 1 + num_spec
    )
    return (conv_state_shape,)


def is_model_fused_shared_expert_compatible(
    layers: Iterable[torch.nn.Module],
    moe_cls: type,
    moe_name: str,
) -> bool:
    """整模型只允许一种 fused-shared-expert 状态，返回是否全部开启。"""

    def get_moe_layer(layer):
        for name in moe_name.split("."):
            layer = getattr(layer, name, None)
            if layer is None:
                return None
        return layer

    moe_layers = []
    for layer in layers:
        if isinstance(layer, PPMissingLayer):
            continue
        moe_layer = get_moe_layer(layer)
        if moe_layer is not None and isinstance(moe_layer, moe_cls):
            moe_layers.append(moe_layer)

    enabled = [
        getattr(layer, "is_fused_shared_expert_enabled", False) for layer in moe_layers
    ]
    enabled_count = sum(enabled)
    disabled_count = len(enabled) - enabled_count
    if enabled_count > 0 and disabled_count > 0:
        raise NotImplementedError(
            "Fused shared experts must be enabled for all MoE layers; found "
            f"{enabled_count} enabled and {disabled_count} disabled layers."
        )
    return enabled_count > 0 and disabled_count == 0


def maybe_fuse_shared_experts(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    n_routed_experts: int,
    n_shared_experts: int,
    ckpt_prefix: str = "mlp.shared_experts",
    enabled: bool | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """把 shared expert 权重改名塞进 routed expert 槽位（AITER 融合专用）。

    昆仑上这条融合始终关闭，因此退化为直通；函数体保持与上游一致，万一被打开
    行为也一样。
    """
    if enabled is None:
        try:
            from vllm._aiter_ops import rocm_aiter_ops

            enabled = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        except Exception:
            enabled = False
    if not enabled:
        yield from weights
        return

    prefix = f"{ckpt_prefix}."
    for name, loaded_weight in weights:
        if prefix not in name:
            yield name, loaded_weight
            continue
        split_dim = 1 if ("down_proj.weight" in name and loaded_weight.ndim > 1) else 0
        total = loaded_weight.shape[split_dim]
        if total % n_shared_experts != 0:
            raise ValueError(
                f"FSE shared-expert weight {name!r} has size {total} along axis "
                f"{split_dim}, not divisible by n_shared_experts={n_shared_experts}."
            )
        chunk = total // n_shared_experts
        for j in range(n_shared_experts):
            sl = slice(j * chunk, (j + 1) * chunk)
            if loaded_weight.ndim == 1:
                chunk_weight = loaded_weight[sl]
            elif split_dim == 0:
                chunk_weight = loaded_weight[sl, :]
            else:
                chunk_weight = loaded_weight[:, sl]
            yield (
                name.replace(prefix, f"mlp.experts.{n_routed_experts + j}."),
                chunk_weight,
            )
