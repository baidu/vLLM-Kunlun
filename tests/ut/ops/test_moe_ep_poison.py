"""EP local-map and poisoned-workspace regression checks on Kunlun XPU."""

import pytest
import torch
from vllm.v1.worker import workspace as workspace_module
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_kunlun.ops._kunlun_ops import KunlunOps
from vllm_kunlun.ops.moe.entry import _plan, _route
from vllm_kunlun.ops.moe.router import make_moe_router
from vllm_kunlun.ops.moe.workspace import MOE_BLOCK_STATISTIC_ROWS, MoeWorkspaces


def _inputs(dtype=torch.float16, tokens=17, global_experts=4, hidden=32, inter=16):
    torch.manual_seed(1234 + tokens)
    device = torch.device("cuda")
    x = torch.randn(tokens, hidden, dtype=dtype, device=device) / 8
    w13 = (
        torch.randn(global_experts, 2 * inter, hidden, dtype=dtype, device=device) / 16
    )
    w2 = torch.randn(global_experts, hidden, inter, dtype=dtype, device=device) / 16
    logits = torch.randn(tokens, global_experts, dtype=dtype, device=device)
    return x, w13, w2, logits


def _router(top_k=2):
    return make_moe_router(top_k=top_k, scoring_func="softmax", renormalize=True)


def test_full_local_identity_matches_fused_moe():
    init_workspace_manager(torch.device("cuda"))
    x, w13, w2, logits = _inputs(torch.float16, 17)
    router = _router()
    expected = KunlunOps.fused_moe(x, w13, w2, logits, 2, True, router=router)
    actual = KunlunOps.fused_moe_ep(x, w13, w2, logits, 0, 2, True, router=router)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [17, 385])
def test_mapped_ep_poison_is_finite_and_nonlocal_is_zero(tokens):
    init_workspace_manager(torch.device("cuda"))
    x, w13, w2, logits = _inputs(torch.bfloat16, tokens)
    logits.fill_(-10)
    logits[:, 2:] = 10
    expert_map = torch.full((4,), -1, dtype=torch.int32, device=x.device)
    control = KunlunOps.fused_moe_ep(
        x, w13[:2], w2[:2], logits, 0, 2, True, expert_map=expert_map, router=_router()
    )
    torch.cuda.synchronize()
    manager_class = type(workspace_module.current_workspace_manager())
    original = manager_class.get_simultaneous

    def poisoned(manager, *requests):
        tensors = original(manager, *requests)
        for index, tensor in enumerate(tensors):
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
                if index == 1:
                    tensor.zero_()
        return tensors

    manager_class.get_simultaneous = poisoned
    try:
        output = KunlunOps.fused_moe_ep(
            x,
            w13[:2],
            w2[:2],
            logits,
            0,
            2,
            True,
            expert_map=expert_map,
            router=_router(),
        )
        torch.cuda.synchronize()
    finally:
        manager_class.get_simultaneous = original
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output).item() == 0
    torch.testing.assert_close(output, control, rtol=0, atol=0)


def test_mapped_ep_graph_replay_uses_new_inputs():
    init_workspace_manager(torch.device("cuda"))
    x, w13, w2, logits = _inputs(torch.float16, 385)
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32, device=x.device)
    router = _router()

    def run():
        return KunlunOps.fused_moe_ep(
            x, w13[:2], w2[:2], logits, 0, 2, True, expert_map=expert_map, router=router
        )

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize()
    first = captured.detach().clone()
    x.add_(0.125)
    logits.mul_(-1)
    graph.replay()
    torch.cuda.synchronize()
    second = captured.detach().clone()
    assert torch.isfinite(second).all()
    assert not torch.equal(first, second)


@pytest.mark.parametrize("tokens", [384, 385])
def test_mapped_ep_boundary_matches_sum_of_local_ranks(tokens):
    init_workspace_manager(torch.device("cuda"))
    x, w13, w2, logits = _inputs(torch.float16, tokens)
    router = _router()
    expected = KunlunOps.fused_moe(x, w13, w2, logits, 2, True, router=router)
    rank0 = KunlunOps.fused_moe_ep(
        x,
        w13[[0, 2]],
        w2[[0, 2]],
        logits,
        0,
        2,
        True,
        expert_map=torch.tensor([0, -1, 1, -1], dtype=torch.int32, device=x.device),
        router=router,
    )
    rank1 = KunlunOps.fused_moe_ep(
        x,
        w13[[1, 3]],
        w2[[1, 3]],
        logits,
        1,
        2,
        True,
        expert_map=torch.tensor([-1, 0, -1, 1], dtype=torch.int32, device=x.device),
        router=router,
    )
    torch.cuda.synchronize()
    actual = rank0 + rank1
    assert actual.shape == expected.shape == (tokens, x.shape[1])
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-5)


def test_sorted_all_nonlocal_is_zero():
    init_workspace_manager(torch.device("cuda"))
    x, w13, w2, logits = _inputs(torch.float16, 385)
    logits.fill_(-10)
    logits[:, 2:] = 10
    output = KunlunOps.fused_moe_ep(
        x,
        w13[:2],
        w2[:2],
        logits,
        0,
        2,
        True,
        expert_map=torch.full((4,), -1, dtype=torch.int32, device=x.device),
        router=_router(),
    )
    torch.cuda.synchronize()
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output).item() == 0


def test_gen_block_statistic_ignores_negative_ids():
    ids = torch.tensor(
        ([0, 1, -1, -1] * 193)[:770], dtype=torch.int32, device="cuda"
    ).view(385, 2)
    block_statistic = torch.empty(
        (MOE_BLOCK_STATISTIC_ROWS, 2), dtype=torch.int32, device="cuda"
    )
    torch.ops._C.gen_block_statistic(ids, block_statistic)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        block_statistic.sum(dim=0).cpu(), torch.tensor([193, 193])
    )


def test_sorted_mapped_ep_native_stages():
    init_workspace_manager(torch.device("cuda"))
    x, w13, w2, logits = _inputs(torch.float16, 385)
    plan = _plan(x, w13[:2], w2[:2], 2, "silu")
    workspaces = MoeWorkspaces.fp(
        num_rows=plan.num_rows,
        num_experts=plan.num_experts,
        hidden_size=plan.hidden_size,
        gate_up_size=plan.gate_up_size,
        output_size=plan.output_size,
        dtype=x.dtype,
        use_sorted=True,
        fuse_swiglu=False,
        activation_allocates=False,
    )
    routed = _route(
        plan,
        x,
        _router(),
        logits,
        None,
        None,
        workspaces.metadata.block_statistic,
        plan.num_experts,
        0,
        torch.tensor([0, -1, 1, -1], dtype=torch.int32, device=x.device),
    )
    torch.cuda.synchronize()

    torch.ops._C.gen_block_statistic(routed.ids, workspaces.metadata.block_statistic)
    torch.cuda.synchronize()
    assert (
        workspaces.metadata.block_statistic.sum().item()
        == routed.ids.ge(0).sum().item()
    )

    expanded = MoeWorkspaces.view(
        workspaces.a, plan.num_rows, plan.hidden_size, plan.num_rows
    )
    torch.ops._C.moe_pre_sorted(
        x=x,
        topk_index=routed.ids,
        block_statistic=workspaces.metadata.block_statistic,
        moe_expand=expanded,
        moe_index=workspaces.metadata.sorted_tokens_idx,
        expert_m=workspaces.metadata.expert_m,
        sorted_tokens_num_lod=workspaces.metadata.sorted_tokens_num_lod,
        index_have_neg=True,
    )
    torch.cuda.synchronize()

    workspaces.b.zero_()
    gate_up = MoeWorkspaces.view(
        workspaces.b, plan.num_rows, plan.gate_up_size, plan.num_tokens, plan.top_k
    )
    torch.ops._C.moe_fc(
        x=expanded,
        weight=w13[:2],
        sorted_tokens_num_lod=workspaces.metadata.sorted_tokens_num_lod,
        sorted_tokens_idx=workspaces.metadata.sorted_tokens_idx,
        moe_topk=plan.top_k,
        y=gate_up,
        topk_ids=routed.ids,
        act=None,
    )
    torch.cuda.synchronize()

    activated = MoeWorkspaces.view(
        workspaces.a, plan.num_rows, plan.gate_up_size // 2, plan.num_tokens, plan.top_k
    )
    torch.ops._C.silu_and_mul(activated, gate_up)
    torch.cuda.synchronize()

    expert_output = MoeWorkspaces.view(
        workspaces.b, plan.num_rows, plan.output_size, plan.num_tokens, plan.top_k
    )
    torch.ops._C.moe_fc(
        x=activated.reshape(plan.num_rows, -1),
        weight=w2[:2],
        sorted_tokens_num_lod=workspaces.metadata.sorted_tokens_num_lod,
        sorted_tokens_idx=workspaces.metadata.sorted_tokens_idx,
        moe_topk=plan.top_k,
        y=expert_output,
        topk_ids=routed.ids,
        act=None,
    )
    torch.cuda.synchronize()

    output = torch.empty_like(x)
    torch.ops._C.moe_post(
        x=expert_output,
        moe_index=workspaces.metadata.sorted_tokens_idx.view(
            plan.num_tokens, plan.top_k
        ),
        normed_scale=routed.scores,
        dequant_scale=torch.ones_like(routed.scores, dtype=torch.float32),
        y=output,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
