"""Backend FFN and combine stages for Kunlun MoE."""

from typing import Optional

import torch

from .activation import MOE_ACTIVATIONS_ALLOCATING_OUTPUT, apply_gated_activation
from .workspace import MoeWorkspaces

_CACHE_RETIRED: list[torch.Tensor] = []
_DEQUANT_SCALE_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}
_ROW_INDEX_CACHE: dict[torch.device, torch.Tensor] = {}


def _capacity(required: int) -> int:
    return 1 << (max(required, 1) - 1).bit_length()


def _dequant_scale(num_tokens: int, top_k: int, device: torch.device) -> torch.Tensor:
    key = (device, top_k)
    value = _DEQUANT_SCALE_CACHE.get(key)
    if value is None or value.shape[0] < num_tokens:
        if value is not None:
            _CACHE_RETIRED.append(value)
        value = torch.ones(
            (_capacity(num_tokens), top_k), dtype=torch.float32, device=device
        )
        _DEQUANT_SCALE_CACHE[key] = value
    return value[:num_tokens]


def _row_index(num_rows: int, device: torch.device) -> torch.Tensor:
    value = _ROW_INDEX_CACHE.get(device)
    if value is None or value.numel() < num_rows:
        if value is not None:
            _CACHE_RETIRED.append(value)
        value = torch.arange(_capacity(num_rows), device=device)
        _ROW_INDEX_CACHE[device] = value
    return value[:num_rows]


def _sorted_row_expert(
    lod: torch.Tensor, num_rows: int, num_experts: int
) -> torch.Tensor:
    # Clamped rows after lod[-1] are skipped by moe_post and only protect bias indexing.
    return torch.searchsorted(
        lod[1:].contiguous(), _row_index(num_rows, lod.device), right=True
    ).clamp_max_(num_experts - 1)


def run_moe_ffn(
    moe_expand: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    lod: torch.Tensor,
    workspaces: MoeWorkspaces,
    num_tokens: int,
    top_k: int,
    activation: str,
    fuse_swiglu: bool = False,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    num_rows = num_tokens * top_k
    gate_up_size = w13_weight.shape[1]
    inter_size = gate_up_size // 2
    output_size = w2_weight.shape[1]
    sorted_expert = None
    if w13_bias is not None or w2_bias is not None:
        sorted_expert = _sorted_row_expert(lod, num_rows, w13_weight.shape[0])

    if fuse_swiglu:
        activated = MoeWorkspaces.view(
            workspaces.b, num_rows, inter_size, num_tokens, top_k
        )
        torch.ops._C.moe_fc(
            x=moe_expand,
            weight=w13_weight,
            sorted_tokens_num_lod=lod,
            sorted_tokens_idx=sorted_tokens_idx,
            moe_topk=top_k,
            y=activated,
            topk_ids=topk_ids,
            act="SWISH_GLU",
        )
        expert_output = MoeWorkspaces.view(
            workspaces.a, num_rows, output_size, num_tokens, top_k
        )
    else:
        gate_up = MoeWorkspaces.view(
            workspaces.b, num_rows, gate_up_size, num_tokens, top_k
        )
        torch.ops._C.moe_fc(
            x=moe_expand,
            weight=w13_weight,
            sorted_tokens_num_lod=lod,
            sorted_tokens_idx=sorted_tokens_idx,
            moe_topk=top_k,
            y=gate_up,
            topk_ids=topk_ids,
            act=None,
        )
        if w13_bias is not None:
            gate_up.view(-1, gate_up_size).add_(w13_bias[sorted_expert])
        out = (
            None
            if activation in MOE_ACTIVATIONS_ALLOCATING_OUTPUT
            else MoeWorkspaces.view(
                workspaces.a, num_rows, inter_size, num_tokens, top_k
            )
        )
        activated = apply_gated_activation(
            activation, gate_up, out, swiglu_alpha, swiglu_beta, swiglu_limit
        )
        expert_output = MoeWorkspaces.view(
            workspaces.b, num_rows, output_size, num_tokens, top_k
        )

    torch.ops._C.moe_fc(
        x=activated.reshape(num_rows, -1),
        weight=w2_weight,
        sorted_tokens_num_lod=lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=top_k,
        y=expert_output,
        topk_ids=topk_ids,
        act=None,
    )
    if w2_bias is not None:
        expert_output.view(-1, output_size).add_(w2_bias[sorted_expert])
    return expert_output


def run_moe_ffn_int8(
    moe_expand: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    lod: torch.Tensor,
    workspaces: MoeWorkspaces,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Run the INT8 FFN contract, whose activation is always SiLU/Swish."""
    num_rows = num_tokens * top_k
    hidden_size = moe_expand.shape[1]
    gate_up_size = w13_weight.shape[1]
    inter_size = gate_up_size // 2
    output_size = w2_weight.shape[1]
    x_q = MoeWorkspaces.view(workspaces.q, num_rows, hidden_size, num_rows)
    x_scale = workspaces.scales[:num_rows].view(num_rows, 1)
    torch.ops._C.quant2d(moe_expand, x_q, x_scale, force_sdnn=True)
    gate_up = MoeWorkspaces.view(
        workspaces.b, num_rows, gate_up_size, num_tokens, top_k
    )
    torch.ops._C.moe_fc(
        x=x_q,
        x_perchannel_max=x_scale,
        weight=w13_weight,
        w_perchannel_max=w13_weight_scale,
        sorted_tokens_num_lod=lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=top_k,
        y=gate_up,
        topk_ids=topk_ids,
        act=None,
    )
    activated = MoeWorkspaces.view(
        workspaces.a, num_rows, inter_size, num_tokens, top_k
    )
    torch.ops._C.silu_and_mul(activated, gate_up)
    x_q = MoeWorkspaces.view(workspaces.q, num_rows, inter_size, num_rows)
    torch.ops._C.quant2d(
        activated.reshape(num_rows, inter_size), x_q, x_scale, force_sdnn=True
    )
    expert_output = MoeWorkspaces.view(
        workspaces.b, num_rows, output_size, num_tokens, top_k
    )
    torch.ops._C.moe_fc(
        x=x_q,
        x_perchannel_max=x_scale,
        weight=w2_weight,
        w_perchannel_max=w2_weight_scale,
        sorted_tokens_num_lod=lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=top_k,
        y=expert_output,
        topk_ids=topk_ids,
        act=None,
    )
    return expert_output


def combine_moe_output(
    expert_output: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    scores: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    output = MoeWorkspaces.output(num_tokens, expert_output.shape[-1], expert_output)
    torch.ops._C.moe_post(
        x=expert_output,
        moe_index=sorted_tokens_idx.view(num_tokens, top_k),
        normed_scale=scores,
        dequant_scale=_dequant_scale(num_tokens, top_k, expert_output.device),
        y=output,
    )
    return output
