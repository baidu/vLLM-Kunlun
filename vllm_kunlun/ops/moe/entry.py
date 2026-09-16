"""Public fused MoE entry points built from the shared Kunlun pipeline."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .activation import MOE_ACTIVATIONS_ALLOCATING_OUTPUT, SUPPORTED_MOE_ACTIVATIONS
from .ffn import combine_moe_output, run_moe_ffn, run_moe_ffn_int8
from .preprocess import (
    MOE_PREPROCESS_THRESHOLD,
    contiguous_expert_map,
    map_expert_ids,
    prepare_moe_tokens,
)
from .router import MoeRouter, make_moe_router, run_moe_router
from .workspace import MoeWorkspaces

_MOE_FUSED_SWIGLU_MIN_M = 2048


@dataclass(frozen=True)
class _MoePlan:
    num_tokens: int
    hidden_size: int
    num_rows: int
    num_experts: int
    gate_up_size: int
    output_size: int
    top_k: int
    activation: str
    use_sorted: bool
    fuse_swiglu: bool


def _activation_name(activation) -> str:
    activation = getattr(activation, "value", activation)
    if activation == "gelu_pytorch_tanh":
        activation = "gelu_tanh"
    if activation not in SUPPORTED_MOE_ACTIVATIONS:
        raise ValueError(
            f"Unsupported gated MoE activation for Kunlun path: {activation}"
        )
    return activation


def _plan(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    top_k: int,
    activation: str,
) -> _MoePlan:
    activation = _activation_name(activation)
    num_experts, gate_up_size, _ = w13_weight.shape
    num_tokens, hidden_size = hidden_states.shape
    num_rows = num_tokens * top_k
    return _MoePlan(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_rows=num_rows,
        num_experts=num_experts,
        gate_up_size=gate_up_size,
        output_size=w2_weight.shape[1],
        top_k=top_k,
        activation=activation,
        use_sorted=num_rows > MOE_PREPROCESS_THRESHOLD,
        fuse_swiglu=(
            activation in ("silu", "swish")
            and hidden_states.dtype == torch.float16
            and num_tokens >= _MOE_FUSED_SWIGLU_MIN_M
        ),
    )


def _router(
    router: Optional[MoeRouter],
    top_k: int,
    scoring_func: str,
    renormalize: bool,
    use_grouped_topk: bool,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
    correction_bias: Optional[torch.Tensor],
    routed_scaling_factor: float,
    custom_routing_function: Optional[Callable],
) -> MoeRouter:
    return router or make_moe_router(
        top_k=top_k,
        scoring_func=scoring_func,
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        has_e_score_correction_bias=correction_bias is not None,
        routed_scaling_factor=routed_scaling_factor,
        has_custom_routing=custom_routing_function is not None,
    )


@dataclass(frozen=True)
class _Routed:
    scores: torch.Tensor
    ids: torch.Tensor
    block_statistic: Optional[torch.Tensor]
    index_have_neg: bool


def _route(
    plan: _MoePlan,
    hidden_states: torch.Tensor,
    router: MoeRouter,
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    custom_routing_function: Optional[Callable],
    block_statistic: Optional[torch.Tensor] = None,
    local_experts: Optional[int] = None,
    ep_rank: Optional[int] = None,
    expert_map: Optional[torch.Tensor] = None,
) -> _Routed:
    mapped = local_experts is not None
    global_experts = router_logits.shape[1]
    identity = (
        mapped
        and expert_map is None
        and ep_rank == 0
        and local_experts == global_experts
    )
    scores, ids, block_statistic = run_moe_router(
        router,
        router_logits,
        hidden_states=hidden_states,
        e_score_correction_bias=correction_bias,
        custom_routing_function=custom_routing_function,
        # A mapped EP histogram counts global IDs and cannot serve local IDs.
        want_block_statistic=(plan.use_sorted and not mapped) or identity,
        block_statistic=block_statistic if not mapped or identity else None,
    )
    if not mapped or identity:
        return _Routed(scores, ids, block_statistic, False)
    if expert_map is None:
        expert_map = contiguous_expert_map(
            ids.device, global_experts, local_experts, ep_rank
        )
    elif expert_map.numel() != global_experts:
        raise ValueError(
            f"expert_map has {expert_map.numel()} entries, expected {global_experts}"
        )
    local_ids = map_expert_ids(ids, expert_map)
    # moe_post consumes fixed-shape rows and reads their scores; mask routes
    # skipped by the local preprocess index so this rank contributes zero.
    scores = scores * local_ids.ge(0)
    return _Routed(scores, local_ids, None, True)


def _empty_output(plan: _MoePlan, hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (0, plan.output_size), dtype=hidden_states.dtype, device=hidden_states.device
    )


def _run_fp_moe(
    plan: _MoePlan,
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    routed: _Routed,
    w13_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    swiglu_alpha: Optional[float],
    swiglu_beta: Optional[float],
    swiglu_limit: Optional[float],
    workspaces: MoeWorkspaces,
) -> torch.Tensor:
    if plan.num_tokens == 0:
        return _empty_output(plan, hidden_states)
    if routed.index_have_neg:
        # EP preprocess keeps non-local rows in the fixed shape but the FC kernels
        # skip them. Zero the shared W13/output buffer so moe_post cannot observe
        # stale values for those rows; local rows are overwritten by the kernels.
        workspaces.b.zero_()
    sorted_idx, lod, expanded = prepare_moe_tokens(
        hidden_states,
        routed.ids,
        plan.num_experts,
        workspaces.a,
        workspaces.metadata,
        routed.block_statistic,
        index_have_neg=routed.index_have_neg,
    )
    expert_output = run_moe_ffn(
        expanded.reshape(plan.num_rows, plan.hidden_size),
        w13_weight,
        w2_weight,
        routed.ids,
        sorted_idx,
        lod,
        workspaces,
        plan.num_tokens,
        plan.top_k,
        plan.activation,
        plan.fuse_swiglu and w13_bias is None,
        w13_bias,
        w2_bias,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
    )
    return combine_moe_output(
        expert_output, sorted_idx, routed.scores, plan.num_tokens, plan.top_k
    )


def fused_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    router_logits: torch.Tensor,
    moe_top_k: int,
    renormalize: bool,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    custom_routing_function: Optional[Callable] = None,
    routed_scaling_factor: float = 1.0,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    swiglu_limit: Optional[float] = None,
    router: Optional[MoeRouter] = None,
) -> torch.Tensor:
    plan = _plan(hidden_states, w13_weight, w2_weight, moe_top_k, activation)
    if plan.num_tokens == 0:
        return _empty_output(plan, hidden_states)
    router = _router(
        router,
        moe_top_k,
        scoring_func,
        renormalize,
        use_grouped_topk,
        num_expert_group,
        topk_group,
        e_score_correction_bias,
        routed_scaling_factor,
        custom_routing_function,
    )
    workspaces = MoeWorkspaces.fp(
        num_rows=plan.num_rows,
        num_experts=plan.num_experts,
        hidden_size=plan.hidden_size,
        gate_up_size=plan.gate_up_size,
        output_size=plan.output_size,
        dtype=hidden_states.dtype,
        use_sorted=plan.use_sorted,
        fuse_swiglu=plan.fuse_swiglu and w13_bias is None,
        activation_allocates=plan.activation in MOE_ACTIVATIONS_ALLOCATING_OUTPUT,
    )
    routed = _route(
        plan,
        hidden_states,
        router,
        router_logits,
        e_score_correction_bias,
        custom_routing_function,
        workspaces.metadata.block_statistic,
    )
    return _run_fp_moe(
        plan,
        hidden_states,
        w13_weight,
        w2_weight,
        routed,
        w13_bias,
        w2_bias,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        workspaces,
    )


def fused_moe_int8(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    router_logits: torch.Tensor,
    moe_top_k: int,
    router: MoeRouter,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    custom_routing_function: Optional[Callable] = None,
) -> torch.Tensor:
    plan = _plan(hidden_states, w13_weight, w2_weight, moe_top_k, "silu")
    if plan.num_tokens == 0:
        return _empty_output(plan, hidden_states)
    ws = MoeWorkspaces.int8(
        num_rows=plan.num_rows,
        num_experts=plan.num_experts,
        hidden_size=plan.hidden_size,
        gate_up_size=plan.gate_up_size,
        output_size=plan.output_size,
        dtype=hidden_states.dtype,
        use_sorted=plan.use_sorted,
    )
    routed = _route(
        plan,
        hidden_states,
        router,
        router_logits,
        e_score_correction_bias,
        custom_routing_function,
        ws.metadata.block_statistic,
    )
    sorted_idx, lod, expanded = prepare_moe_tokens(
        hidden_states,
        routed.ids,
        plan.num_experts,
        ws.a,
        ws.metadata,
        routed.block_statistic,
    )
    expert_output = run_moe_ffn_int8(
        expanded.reshape(plan.num_rows, plan.hidden_size),
        w13_weight,
        w13_weight_scale,
        w2_weight,
        w2_weight_scale,
        routed.ids,
        sorted_idx,
        lod,
        ws,
        plan.num_tokens,
        plan.top_k,
    )
    return combine_moe_output(
        expert_output, sorted_idx, routed.scores, plan.num_tokens, plan.top_k
    )


def fused_moe_ep(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    router_logits: torch.Tensor,
    ep_rank: int,
    top_k: int,
    renormalize: bool,
    inplace: bool = False,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    custom_routing_function: Optional[Callable] = None,
    routed_scaling_factor: float = 1.0,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    swiglu_limit: Optional[float] = None,
    expert_map: Optional[torch.Tensor] = None,
    router: Optional[MoeRouter] = None,
) -> torch.Tensor:
    del inplace
    plan = _plan(hidden_states, w13_weight, w2_weight, top_k, activation)
    if plan.num_tokens == 0:
        return _empty_output(plan, hidden_states)
    if plan.num_experts == 0:
        return torch.zeros(
            (plan.num_tokens, plan.output_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    router = _router(
        router,
        top_k,
        scoring_func,
        renormalize,
        use_grouped_topk,
        num_expert_group,
        topk_group,
        e_score_correction_bias,
        routed_scaling_factor,
        custom_routing_function,
    )
    workspaces = MoeWorkspaces.fp(
        num_rows=plan.num_rows,
        num_experts=plan.num_experts,
        hidden_size=plan.hidden_size,
        gate_up_size=plan.gate_up_size,
        output_size=plan.output_size,
        dtype=hidden_states.dtype,
        use_sorted=plan.use_sorted,
        fuse_swiglu=plan.fuse_swiglu and w13_bias is None,
        activation_allocates=plan.activation in MOE_ACTIVATIONS_ALLOCATING_OUTPUT,
    )
    routed = _route(
        plan,
        hidden_states,
        router,
        router_logits,
        e_score_correction_bias,
        custom_routing_function,
        workspaces.metadata.block_statistic,
        plan.num_experts,
        ep_rank,
        expert_map,
    )
    return _run_fp_moe(
        plan,
        hidden_states,
        w13_weight,
        w2_weight,
        routed,
        w13_bias,
        w2_bias,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        workspaces,
    )
