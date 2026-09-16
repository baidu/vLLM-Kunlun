"""End-to-end fused_moe with the new routing branches, above and below the
moe_pre_small/moe_pre_sorted threshold, so the threaded block_statistic is
exercised through moe_pre_sorted."""

import torch
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_kunlun.ops._kunlun_ops import KunlunOps
from vllm_kunlun.ops.moe.router import make_moe_router, run_moe_router

DEV = torch.device("cuda")
init_workspace_manager(DEV)
HIDDEN, INTER, E, TOPK = 512, 256, 32, 4
FAILS = []


def ref_moe(x, w13, w2, topk_ids, score, acc):
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=x.device)
    xc = x.to(acc)
    for e in range(w13.shape[0]):
        rows, ks = (topk_ids == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        h = xc[rows] @ w13[e].to(acc).T
        n = h.shape[-1] // 2
        y = (torch.nn.functional.silu(h[..., :n]) * h[..., n:]) @ w2[e].to(acc).T
        out.index_add_(0, rows, y.float() * score[rows, ks].unsqueeze(1))
    return out


def case(dtype, M, sfunc, renorm, grouped, use_bias, sf):
    torch.manual_seed(M + len(sfunc))
    x = torch.randn(M, HIDDEN, dtype=dtype, device=DEV) / 8
    w13 = torch.randn(E, 2 * INTER, HIDDEN, dtype=dtype, device=DEV) / 16
    w2 = torch.randn(E, HIDDEN, INTER, dtype=dtype, device=DEV) / 16
    lg = torch.randn(M, E, dtype=dtype, device=DEV)
    bias = torch.randn(E, dtype=torch.float32, device=DEV) if use_bias else None
    ng, tg = (4, 2) if grouped else (None, None)

    router = make_moe_router(
        top_k=TOPK,
        scoring_func=sfunc,
        renormalize=renorm,
        use_grouped_topk=grouped,
        num_expert_group=ng,
        topk_group=tg,
        has_e_score_correction_bias=bias is not None,
        routed_scaling_factor=sf,
    )
    score, ids = run_moe_router(router, lg, e_score_correction_bias=bias)[:2]
    got = KunlunOps.fused_moe(
        x,
        w13,
        w2,
        lg,
        TOPK,
        renorm,
        use_grouped_topk=grouped,
        num_expert_group=ng,
        topk_group=tg,
        scoring_func=sfunc,
        e_score_correction_bias=bias,
        routed_scaling_factor=sf,
    )
    torch.cuda.synchronize()
    ref = ref_moe(x, w13, w2, ids, score, torch.float32)
    base = ref_moe(x, w13, w2, ids, score, dtype)
    scale = ref.abs().max().item()
    ek = (got.float() - ref).abs().max().item() / scale
    et = (base - ref).abs().max().item() / scale
    ratio = ek / max(et, 1e-12)
    bad = (not torch.isfinite(got).all()) or ratio > 4.0
    name = (
        f"M={M:<5} {sfunc:<12} renorm={int(renorm)} grouped={int(grouped)} "
        f"bias={int(use_bias)} scale={sf}"
    )
    print(
        f"  [{'FAIL' if bad else 'ok':>4}] {name} "
        f"pre={'small' if M * TOPK <= 768 else 'sorted'} "
        f"rel_kernel={ek:.2e} rel_torch={et:.2e} ratio={ratio:.2f}",
        flush=True,
    )
    if bad:
        FAILS.append(name)


for dtype in (torch.float16, torch.bfloat16):
    print(f"=== {str(dtype).split('.')[-1]} ===", flush=True)
    for M in (17, 200, 2048):
        case(dtype, M, "softmax", True, True, False, 1.0)
        case(dtype, M, "softmax", True, True, False, 2.5)
        case(dtype, M, "sigmoid", True, True, True, 1.0)
        case(dtype, M, "sigmoid", True, True, True, 2.5)
        case(dtype, M, "sigmoid", True, False, True, 2.5)
        case(dtype, M, "sigmoid", False, False, False, 1.0)
        case(dtype, M, "sqrtsoftplus", True, False, True, 1.0)

print("=== summary ===")
print("failures:", FAILS or "none")
raise SystemExit(1 if FAILS else 0)
