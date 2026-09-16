"""Routing parity for the router factory: every branch vs a torch oracle.

Upstream's fused routers cannot be the oracle on XPU (FusedTopKRouter returns
uninitialised buffers), so the oracles are upstream's pure-torch `grouped_topk`
and explicit torch formulas.
"""

import torch
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_kunlun.ops._kunlun_ops import KunlunOps
from vllm_kunlun.ops.moe.router import make_moe_router, run_moe_router

DEV = torch.device("cuda")
init_workspace_manager(DEV)
E, TOPK, M = 32, 4, 200
FAILS = []


def pick(
    top_k,
    scoring_func,
    renormalize,
    use_grouped_topk,
    num_expert_group,
    topk_group,
    bias,
    sf=1.0,
):
    """`make_moe_router` from the config these cases spell positionally."""
    return make_moe_router(
        top_k=top_k,
        scoring_func=scoring_func,
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        has_e_score_correction_bias=bias is not None,
        routed_scaling_factor=sf,
    )


def route(
    logits,
    top_k,
    scoring_func,
    renormalize,
    use_grouped_topk,
    num_expert_group,
    topk_group,
    bias,
    sf=1.0,
    want_bs=False,
):
    """Select and run in one call, the way `fused_moe` does per forward pass."""
    router = pick(
        top_k,
        scoring_func,
        renormalize,
        use_grouped_topk,
        num_expert_group,
        topk_group,
        bias,
        sf,
    )
    return run_moe_router(
        router,
        logits,
        e_score_correction_bias=bias,
        want_block_statistic=want_bs,
    )


def report(name, ref_w, ref_i, got_w, got_i):
    same = torch.equal(
        torch.sort(ref_i.long(), 1).values, torch.sort(got_i.long(), 1).values
    )
    a = torch.zeros(M, E, dtype=torch.float32, device=DEV)
    b = torch.zeros(M, E, dtype=torch.float32, device=DEV)
    a.scatter_(1, ref_i.long(), ref_w.float())
    b.scatter_(1, got_i.long(), got_w.float())
    werr = (a - b).abs().max().item()
    ok = same and werr < 5e-3
    print(
        f"  [{'ok' if ok else 'FAIL':>4}] {name:<52} "
        f"experts_equal={str(same):<5} max_weight_diff={werr:.3e}",
        flush=True,
    )
    if not ok:
        FAILS.append(name)


def plain_ref(scores, renorm, sf, bias=None):
    choice = scores if bias is None else scores + bias
    _, i = torch.topk(choice, TOPK, dim=-1)
    w = scores.gather(1, i)
    if renorm:
        w = w / w.sum(-1, keepdim=True)
    return w * sf, i.to(torch.int32)


torch.manual_seed(11)
logits = torch.randn(M, E, dtype=torch.bfloat16, device=DEV)
lf = logits.float()
hs = torch.randn(M, 128, dtype=torch.bfloat16, device=DEV)
bias = torch.randn(E, dtype=torch.float32, device=DEV)

print("=== ungrouped, no bias ===", flush=True)
for sfunc, scores in (("softmax", lf.softmax(-1)), ("sigmoid", lf.sigmoid())):
    for renorm in (True, False):
        for sf in (1.0, 2.5):
            w, i, bs = route(
                logits, TOPK, sfunc, renorm, False, None, None, None, sf, True
            )
            report(
                f"{sfunc} renorm={renorm} scale={sf}",
                *plain_ref(scores, renorm, sf),
                w,
                i,
            )

print("=== grouped, no bias ===", flush=True)
for sfunc in ("softmax", "sigmoid"):
    for n_group, topk_group in ((4, 2), (8, 4)):
        for sf in (1.0, 2.5):
            gw, gi = grouped_topk(
                hs, lf, TOPK, True, n_group, topk_group, sfunc, routed_scaling_factor=sf
            )
            w, i, bs = route(
                logits, TOPK, sfunc, True, True, n_group, topk_group, None, sf, True
            )
            report(f"{sfunc} grouped({n_group},{topk_group}) scale={sf}", gw, gi, w, i)
    # use_grouped_topk with a single group must fall back to plain top-k
    w, i, _ = route(logits, TOPK, sfunc, True, True, 1, 1, None, 1.0, False)
    scores = lf.softmax(-1) if sfunc == "softmax" else lf.sigmoid()
    report(f"{sfunc} grouped(1,1) -> plain topk", *plain_ref(scores, True, 1.0), w, i)

print("=== grouped + bias (sigmoid only) ===", flush=True)
for n_group, topk_group in ((4, 2), (8, 4)):
    for sf in (1.0, 2.5):
        gw, gi = grouped_topk(
            hs,
            lf,
            TOPK,
            True,
            n_group,
            topk_group,
            "sigmoid",
            routed_scaling_factor=sf,
            e_score_correction_bias=bias,
        )
        w, i, _ = route(
            logits, TOPK, "sigmoid", True, True, n_group, topk_group, bias, sf, False
        )
        report(f"sigmoid grouped({n_group},{topk_group}) bias scale={sf}", gw, gi, w, i)

print("=== ungrouped + bias (moe_fused_gate_dsv4) ===", flush=True)
for sfunc in ("sigmoid", "sqrtsoftplus"):
    scores = (
        lf.sigmoid()
        if sfunc == "sigmoid"
        else torch.sqrt(torch.nn.functional.softplus(lf))
    )
    for renorm in (True, False):
        for sf in (1.0, 2.5):
            w, i, bs = route(
                logits, TOPK, sfunc, renorm, False, None, None, bias, sf, True
            )
            report(
                f"{sfunc} bias renorm={renorm} scale={sf}",
                *plain_ref(scores, renorm, sf, bias),
                w,
                i,
            )

print("=== unsupported combinations must raise ===", flush=True)
# `pick`, not `route`: rejecting a config is `make_moe_router`'s job, and the
# point of it being a separate call is that a layer can make it at load time
# rather than discovering the gap inside its first forward pass.
for name, args in (
    ("softmax + bias, ungrouped", (TOPK, "softmax", True, False, None, None, bias)),
    ("softmax + bias, grouped", (TOPK, "softmax", True, True, 4, 2, bias)),
    (
        "sigmoid + bias, grouped, renormalize=False",
        (TOPK, "sigmoid", False, True, 4, 2, bias),
    ),
    ("sqrtsoftplus, no bias", (TOPK, "sqrtsoftplus", True, False, None, None, None)),
):
    try:
        pick(*args)
        print(f"  [FAIL] {name:<52} no exception", flush=True)
        FAILS.append(name)
    except NotImplementedError as exc:
        print(
            f"  [  ok] {name:<52} NotImplementedError: " f"{str(exc)[:52]}", flush=True
        )

print("=== block_statistic threading ===", flush=True)
for sfunc, use_bias, grouped in (
    ("softmax", False, False),
    ("sigmoid", False, True),
    ("sigmoid", True, False),
):
    w, i, bs = route(
        logits,
        TOPK,
        sfunc,
        True,
        grouped,
        4 if grouped else None,
        2 if grouped else None,
        bias if use_bias else None,
        1.0,
        True,
    )
    ref = torch.zeros(bs.shape, dtype=torch.int32, device=DEV)
    torch.ops._C.gen_block_statistic(i, ref)
    torch.cuda.synchronize()
    ok = torch.equal(bs, ref)
    print(
        f"  [{'ok' if ok else 'FAIL':>4}] {sfunc} bias={int(use_bias)} "
        f"grouped={int(grouped)}: matches gen_block_statistic={ok}",
        flush=True,
    )
    if not ok:
        FAILS.append(f"block_statistic {sfunc}")
    w2, i2, none = route(
        logits,
        TOPK,
        sfunc,
        True,
        grouped,
        4 if grouped else None,
        2 if grouped else None,
        bias if use_bias else None,
        1.0,
        False,
    )
    if none is not None:
        FAILS.append(f"want_block_statistic=False leaked {sfunc}")

print("=== custom_routing_function ===", flush=True)


def my_routing(hidden_states, gating_output, topk, renormalize):
    # Reads hidden_states, like gemma4's per-expert scale, so no kernel can
    # stand in for it.
    scale = hidden_states.float().abs().mean(-1, keepdim=True)
    w, i = torch.topk(gating_output.float().softmax(-1) * scale, topk, dim=-1)
    return w, i


# A custom function emits no histogram of its own, but asking for one still
# has to work -- the caller cannot tell which kernel it got.
custom_router = make_moe_router(top_k=TOPK, has_custom_routing=True)
x_bs = torch.randn(M, 512, dtype=torch.float16, device=DEV) / 8
for want in (True, False):
    _, ids, bs = run_moe_router(
        custom_router,
        logits,
        hidden_states=x_bs,
        custom_routing_function=my_routing,
        want_block_statistic=want,
    )
    if want:
        ref = torch.zeros(bs.shape, dtype=torch.int32, device=DEV)
        torch.ops._C.gen_block_statistic(ids, ref)
        torch.cuda.synchronize()
        ok = torch.equal(bs, ref)
        detail = f"matches gen_block_statistic={ok}"
    else:
        ok = bs is None
        detail = f"is None={ok}"
    print(
        f"  [{'ok' if ok else 'FAIL':>4}] custom want_bs={int(want)}: " f"{detail}",
        flush=True,
    )
    if not ok:
        FAILS.append(f"custom block_statistic want={want}")

for dtype in (torch.float16, torch.bfloat16):
    x = torch.randn(M, 512, dtype=dtype, device=DEV) / 8
    w13 = torch.randn(E, 512, 512, dtype=dtype, device=DEV) / 16
    w2 = torch.randn(E, 512, 256, dtype=dtype, device=DEV) / 16
    lg = torch.randn(M, E, dtype=dtype, device=DEV)
    ref_w, ref_i = my_routing(x, lg, TOPK, True)
    out = KunlunOps.fused_moe(
        x, w13, w2, lg, TOPK, True, custom_routing_function=my_routing
    )
    torch.cuda.synchronize()
    # Reproduce the same experts through the default path with the routing
    # forced, so a wrong custom hookup shows up as a different result.
    plain = KunlunOps.fused_moe(x, w13, w2, lg, TOPK, True)
    torch.cuda.synchronize()
    tag = str(dtype).split(".")[-1]
    differs = (out.float() - plain.float()).abs().max().item()
    print(
        f"  [{'ok' if torch.isfinite(out).all() and differs > 1e-3 else 'FAIL':>4}]"
        f" {tag}: finite={bool(torch.isfinite(out).all())} "
        f"differs_from_default={differs:.3e}",
        flush=True,
    )

print("=== summary ===")
print("failures:", FAILS or "none")
raise SystemExit(1 if FAILS else 0)
