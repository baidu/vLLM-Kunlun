"""Correctness sweep for KunlunOps.fused_moe in fp16 and bf16.

Phase 1 pins down the gate/up layout of every gated activation kernel against
torch formulas, so phase 2's fp32 reference cannot be wrong about the layout.
Phase 2 compares fused_moe against that reference, and against the same
reference evaluated in the test dtype, so the verdict is "is the kernel path
worse than plain torch at this dtype" rather than an absolute tolerance.
"""

import torch
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_kunlun.ops._kunlun_ops import KunlunOps
from vllm_kunlun.ops.moe.activation import apply_gated_activation
from vllm_kunlun.ops.moe.router import make_moe_router, run_moe_router

DEV = torch.device("cuda")
init_workspace_manager(DEV)

ACTS = [
    "silu",
    "gelu_tanh",
    "swigluoai",
    "swigluoai_uninterleave",
    "swiglustep",
]
LIMIT, ALPHA = 7.0, 1.702


def torch_act(name, g, u):
    if name in ("silu", "swish"):
        return torch.nn.functional.silu(g) * u
    if name == "gelu_tanh":
        return torch.nn.functional.gelu(g, approximate="tanh") * u
    if name == "swiglustep":
        return torch.nn.functional.silu(g).clamp(max=LIMIT) * u.clamp(-LIMIT, LIMIT)
    if name in ("swigluoai", "swigluoai_uninterleave"):
        # Same formula either way; the two names only differ in the gate/up
        # layout they expect, which `probe_layouts` discovers separately.
        gc, uc = g.clamp(max=LIMIT), u.clamp(-LIMIT, LIMIT)
        return gc * torch.sigmoid(gc * ALPHA) * (uc + 1)
    raise ValueError(name)


def split(h, layout):
    n = h.shape[-1] // 2
    if layout == "half":
        return h[..., :n], h[..., n:]
    return h[..., 0::2], h[..., 1::2]


def probe_layouts(dtype):
    """Return {act: layout} decided by matching the kernel output."""
    torch.manual_seed(0)
    n, rows = 64, 16
    x = torch.randn(rows, 2 * n, dtype=dtype, device=DEV) * 3
    out = {}
    for act in ACTS:
        got = apply_gated_activation(
            act, x.clone(), torch.empty(rows, n, dtype=dtype, device=DEV)
        ).float()
        errs = {}
        for layout in ("half", "inter"):
            g, u = split(x.float(), layout)
            errs[layout] = (got - torch_act(act, g, u)).abs().max().item()
        best = min(errs, key=errs.get)
        scale = got.abs().max().item()
        print(
            f"  {act:<11} layout={best:<5} "
            f"half={errs['half']:.3e} inter={errs['inter']:.3e} scale={scale:.2f}"
        )
        assert errs[best] < 1e-2 * max(scale, 1.0), f"{act}: no layout matches"
        out[act] = best
    return out


def ref_moe(x, w13, w2, topk_ids, score, act, b13, b2, layout, acc_dtype):
    """Dense per-expert reference in `acc_dtype`, using the kernel's routing."""
    num_experts = w13.shape[0]
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=x.device)
    xc = x.to(acc_dtype)
    for e in range(num_experts):
        rows, ks = (topk_ids == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        h = xc[rows] @ w13[e].to(acc_dtype).T
        if b13 is not None:
            h = h + b13[e].to(acc_dtype)
        g, u = split(h, layout)
        y = torch_act(act, g, u) @ w2[e].to(acc_dtype).T
        if b2 is not None:
            y = y + b2[e].to(acc_dtype)
        out.index_add_(0, rows, (y.float() * score[rows, ks].unsqueeze(1)))
    return out


HIDDEN, INTER, E, TOPK = 512, 256, 32, 4
FAILS = []


def run_case(dtype, num_tokens, act, layouts, use_bias, renormalize):
    torch.manual_seed(num_tokens * 7 + len(act))
    x = torch.randn(num_tokens, HIDDEN, dtype=dtype, device=DEV) / 8
    w13 = torch.randn(E, 2 * INTER, HIDDEN, dtype=dtype, device=DEV) / 16
    w2 = torch.randn(E, HIDDEN, INTER, dtype=dtype, device=DEV) / 16
    logits = torch.randn(num_tokens, E, dtype=dtype, device=DEV)
    b13 = b2 = None
    if use_bias:
        b13 = torch.randn(E, 2 * INTER, dtype=dtype, device=DEV) / 4
        b2 = torch.randn(E, HIDDEN, dtype=dtype, device=DEV) / 4

    router = make_moe_router(
        top_k=TOPK, scoring_func="softmax", renormalize=renormalize
    )
    score, topk_ids = run_moe_router(router, logits)[:2]
    got = KunlunOps.fused_moe(
        x,
        w13,
        w2,
        logits,
        TOPK,
        renormalize,
        w13_bias=b13,
        w2_bias=b2,
        activation=act,
    )
    torch.cuda.synchronize()

    layout = layouts[act]
    ref = ref_moe(x, w13, w2, topk_ids, score, act, b13, b2, layout, torch.float32)
    base = ref_moe(x, w13, w2, topk_ids, score, act, b13, b2, layout, dtype)
    scale = ref.abs().max().item()
    err_k = (got.float() - ref).abs().max().item() / scale
    err_t = (base - ref).abs().max().item() / scale
    ratio = err_k / max(err_t, 1e-12)
    bad = (not torch.isfinite(got).all()) or ratio > 4.0
    tag = "FAIL" if bad else "ok"
    fuse = (
        dtype == torch.float16
        and act in ("silu", "swish")
        and not use_bias
        and (num_tokens <= 2 or num_tokens >= 2048)
    )
    print(
        f"  [{tag:>4}] M={num_tokens:<5} {act:<11} bias={int(use_bias)} "
        f"renorm={int(renormalize)} pre={'small' if num_tokens * TOPK <= 768 else 'sorted'}"
        f" swiglu_fused={int(fuse)} rel_kernel={err_k:.2e} "
        f"rel_torch={err_t:.2e} ratio={ratio:.2f}",
        flush=True,
    )
    if bad:
        FAILS.append((str(dtype), num_tokens, act, use_bias, renormalize, ratio))


for dtype in (torch.float16, torch.bfloat16):
    name = str(dtype).split(".")[-1]
    print(f"=== {name}: activation layout probe ===", flush=True)
    layouts = probe_layouts(dtype)
    print(f"=== {name}: fused_moe sweep ===", flush=True)
    for M in (1, 2, 3, 17, 96, 200, 2048):
        for act in ACTS:
            run_case(dtype, M, act, layouts, False, True)
    for M in (1, 17, 200, 2048):
        for act in ("silu", "swigluoai"):
            run_case(dtype, M, act, layouts, True, True)
    for M in (17, 200):
        run_case(dtype, M, "silu", layouts, False, False)

print("=== summary ===")
if FAILS:
    for f in FAILS:
        print("  FAIL", f)
    raise SystemExit(1)
print("all cases OK")
