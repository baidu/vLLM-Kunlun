# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp decode GEMM selection — Q38 XPU PORT。

上游实现依赖 Blackwell(SM103) 专用的 CuTe-DSL skinny GEMM
(`vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm`)，昆仑 P800 的
vLLM 构建里没有这个模块，import 阶段就会 ModuleNotFoundError。

**但 P800 上真正的瓶颈不是"缺一个瘦长 GEMM kernel"，而是 `F.linear` 这条路本身慢。**
按真实模型形状（TP=8，13 种形状 / 521 个线性层）实测一次 forward 的线性部分：

| M | eager F.linear | eager kunlun matmul | 图重放 F.linear | 图重放 matmul |
| --- | --- | --- | --- | --- |
| 1（decode） | 68.17 ms | 9.10 ms | **15.97 ms** | **9.11 ms** |
| 2048（prefill） | 71.01 ms | 54.85 ms | 57.75 ms | 54.85 ms |

`kunlun_ops.matmul` 在每个形状、每个 M 上都不慢于 `F.linear`，而且相对 fp32 参考的
误差与 `F.linear` **完全一致**（2.0e-3 ~ 3.2e-3，M=1 与 M=512 都验过）。
所以这里不写新 kernel，直接把未量化线性层的 apply 换成 `kunlun_ops.matmul`：
decode 图重放每 token 省约 6.9 ms（45.9 ms 的步长里约 15%）。

开关：`Q38_KL_LINEAR=0` 回到 `F.linear`（用于区分"是不是这条改动引入的问题"）。
"""

import os

import torch
from torch import nn
from vllm.utils.torch_utils import direct_register_custom_op

# 上游的 plan 表在 XPU 上没有意义（不按 (N,K) + token 数选 kernel），保留空表只为
# 让 import 它的模块（mtp.py / model.py）不必改。
QWEN4_EXP_GEMM_PLANS: dict = {}

_ENABLED = os.environ.get("Q38_KL_LINEAR", "1") == "1"


def _kunlun_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    import kunlun_ops

    # kunlun_ops.matmul 只吃 2 维 (m, k)，w_trans 默认 True，正好是 F.linear 的权重序。
    flat = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
    if not flat.is_contiguous():
        flat = flat.contiguous()
    out = torch.empty((flat.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    kunlun_ops.matmul(flat, weight, out)
    if x.dim() == 2:
        return out
    return out.view(*x.shape[:-1], weight.shape[0])


def _kunlun_linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen4_exp_kunlun_linear",
    op_func=_kunlun_linear,
    fake_impl=_kunlun_linear_fake,
)


def enable_qwen4_exp_low_latency_gemm(module: nn.Module, dtype: torch.dtype) -> None:
    """把未量化线性层的 apply 换成 kunlun_ops.matmul（`Q38_KL_LINEAR=0` 关闭）。"""

    if not _ENABLED or dtype != torch.bfloat16:
        return

    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )

    class _KunlunApply:
        def apply(self, layer, x, bias=None):
            weight = layer.weight
            if (
                bias is None
                and x.dtype == torch.bfloat16
                and weight.dtype == torch.bfloat16
                and weight.dim() == 2
                and x.shape[-1] == weight.shape[1]
            ):
                return torch.ops.vllm.qwen4_exp_kunlun_linear(x, weight)
            return super().apply(layer, x, bias)

    class _KunlunLinearMethod(_KunlunApply, UnquantizedLinearMethod):
        pass

    class _KunlunEmbeddingMethod(_KunlunApply, UnquantizedEmbeddingMethod):
        pass

    replaced = 0
    for child in module.modules():
        if (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        ):
            child.quant_method = _KunlunLinearMethod()
        elif (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        ):
            child.quant_method = _KunlunEmbeddingMethod()
        else:
            continue
        replaced += 1
