# SPDX-License-Identifier: Apache-2.0
"""Q38 XPU PORT: HyperConnection 算子的纯 torch 实现。

上游 `nvidia/ops/hc.py` 里 5 个 HC 融合算子是 triton kernel。昆仑 P800 上
triton 能编译但加载二进制失败（`RuntimeError: Triton Error [CUDA]:
CUDA_ERROR_NOT_SUPPORTED`），因此这里按 kernel 语义逐个改写为 torch 实现，
数值路径保持一致（fp32 归约 + 输出回原 dtype）：

- grouped_gemma_rmsnorm: 每行按 num_groups 分组做 RMSNorm，affine 为 (1 + w)
- hc_silu:               x/HC 后做 SiLU
- hc_gate_mix:           sum_s sigmoid(g_s) * x_s / HC
- hc_combine:            residual + block ⊗ (2*sigmoid(inj/HC))
- hc_combine_norm:       先 combine 并回 dtype，再对每条 HC 流做 RMSNorm
"""

import torch


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    n, dim = x.shape
    assert dim % num_groups == 0
    group_dim = dim // num_groups
    xf = x.float().view(n, num_groups, group_dim)
    rrms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = xf * rrms
    w = weight.float()
    if w.numel() == group_dim:
        w = w.view(1, 1, group_dim)
    else:
        w = w.view(1, num_groups, group_dim)
    y = y + y * w
    return y.view(n, dim).to(x.dtype)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    xf = x.float() / hc_count
    return (xf * torch.sigmoid(xf)).to(x.dtype)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    n, dim = gate.shape
    assert x.shape == gate.shape and dim % hc_count == 0
    hc_dim = dim // hc_count
    xf = x.float().view(n, hc_count, hc_dim)
    gf = gate.float().view(n, hc_count, hc_dim)
    out = (torch.sigmoid(gf) * xf).sum(dim=1) / hc_count
    return out.to(x.dtype)


def _injection_scale(injection_logits: torch.Tensor, hc_count: int) -> torch.Tensor:
    return 2.0 * torch.sigmoid(injection_logits.float() / hc_count)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    n, dim = residual.shape
    hc_dim = dim // hc_count
    assert block_output.shape == (n, hc_dim)
    assert injection_logits.shape == (n, hc_count)
    res = residual.float().view(n, hc_count, hc_dim)
    inj = _injection_scale(injection_logits, hc_count).view(n, hc_count, 1)
    out = res + block_output.float().unsqueeze(1) * inj
    return out.view(n, dim).to(residual.dtype)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, dim = residual.shape
    hc_dim = dim // hc_count
    # kernel 在归一化之前会把 combine 结果按输出 dtype 落盘，这里保持同一舍入边界
    out = hc_combine(residual, block_output, injection_logits, hc_count)
    outf = out.float().view(n, hc_count, hc_dim)
    rrms = torch.rsqrt(outf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = outf * rrms
    w = norm_weight.float()
    if w.numel() == hc_dim:
        w = w.view(1, 1, hc_dim)
    else:
        w = w.view(1, hc_count, hc_dim)
    y = y + y * w
    return out, y.view(n, dim).to(residual.dtype)
