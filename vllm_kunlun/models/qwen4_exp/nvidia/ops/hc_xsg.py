# SPDX-License-Identifier: Apache-2.0
"""Q38 XPU PORT: HyperConnection 算子的 XSpeedGate 融合实现。

`hc_torch.py` 是逐算子的纯 torch 兜底，一层 HC 会展开成约 30 个 elementwise
kernel，decode 阶段完全是 launch 开销（实测 tokens=1 时 4 个算子合计
607 us/层，48 层 29 ms，比整步 46 ms 的一半还多）。这里改调
`torch.ops.xspeedgate_ops.hc_*`，语义与 `hc_torch.py` 逐位对齐：

- hc_grouped_rmsnorm: 每行按 num_groups 分组做 RMSNorm，affine 为 (1 + w)
- hc_silu:            x/HC 后做 SiLU
- hc_gate_mix:        sum_s sigmoid(g_s) * x_s / HC
- hc_combine_norm:    先 combine 并回 dtype，再对每条 HC 流做 RMSNorm；
                      norm_weight=None 时只做 combine（第二个输出为空张量）

算子不可用（包未装 / 版本旧）时自动退回 `hc_torch`，置 Q38_XSG_HC=0 可强制走
torch 兜底。
"""

import os

import torch

from . import hc_torch

_FORCE_TORCH = os.environ.get("Q38_XSG_HC", "1") != "1"


def _have(name: str) -> bool:
    if _FORCE_TORCH:
        return False
    try:
        import xspeedgate_ops  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops.xspeedgate_ops, name)


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    if not _have("hc_grouped_rmsnorm"):
        return hc_torch.grouped_gemma_rmsnorm(x, weight, eps, num_groups)
    return torch.ops.xspeedgate_ops.hc_grouped_rmsnorm(
        x.contiguous(), weight.contiguous(), eps, num_groups
    )


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    if not _have("hc_silu"):
        return hc_torch.hc_silu(x, hc_count)
    return torch.ops.xspeedgate_ops.hc_silu(x.contiguous(), hc_count)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    if not _have("hc_gate_mix"):
        return hc_torch.hc_gate_mix(x, gate, hc_count)
    return torch.ops.xspeedgate_ops.hc_gate_mix(
        x.contiguous(), gate.contiguous(), hc_count
    )


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if not _have("hc_combine_norm"):
        return hc_torch.hc_combine(residual, block_output, injection_logits, hc_count)
    out, _ = torch.ops.xspeedgate_ops.hc_combine_norm(
        residual.contiguous(),
        block_output.contiguous(),
        injection_logits.contiguous(),
        None,
        0.0,
        hc_count,
    )
    return out


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _have("hc_combine_norm"):
        return hc_torch.hc_combine_norm(
            residual, block_output, injection_logits, norm_weight, eps, hc_count
        )
    return torch.ops.xspeedgate_ops.hc_combine_norm(
        residual.contiguous(),
        block_output.contiguous(),
        injection_logits.contiguous(),
        norm_weight.contiguous(),
        eps,
        hc_count,
    )
