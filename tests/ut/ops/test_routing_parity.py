"""Routing parity for fused_moe: _route_moe vs pure-torch oracles.

Upstream's FusedTopKRouter cannot be used as the oracle here -- on XPU it
returns uninitialized buffers (expert ids in the 1e8 range), which is exactly
why monolithic mode takes over routing. So the oracles are torch softmax+topk
and upstream's pure-torch ``grouped_topk``.
"""

import torch
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_kunlun.ops._kunlun_ops import _MOE_BLOCK_STATISTIC_ROWS, KunlunOps, _route_moe

DEV = torch.device("cuda")
init_workspace_manager(DEV)
E, TOPK, M = 32, 4, 64
FAILS = []


def report(name, ref_w, ref_i, got_w, got_i, expect_match=True):
    same = torch.equal(
        torch.sort(ref_i.long(), 1).values, torch.sort(got_i.long(), 1).values
    )
    a = torch.zeros(M, E, dtype=torch.float32, device=DEV)
    b = torch.zeros(M, E, dtype=torch.float32, device=DEV)
    a.scatter_(1, ref_i.long(), ref_w.float())
    b.scatter_(1, got_i.long(), got_w.float())
    werr = (a - b).abs().max().item()
    ok = same and werr < 5e-3
    verdict = "MATCH" if ok else "DIVERGES"
    flag = "" if ok == expect_match else "   <-- UNEXPECTED"
    print(
        f"  {name:<46} {verdict:<9} experts_equal={str(same):<5} "
        f"max_weight_diff={werr:.3e}{flag}",
        flush=True,
    )
    if ok != expect_match:
        FAILS.append(name)


def softmax_topk_ref(logits, renormalize):
    w, i = torch.topk(logits.float().softmax(-1), TOPK, dim=-1)
    if renormalize:
        w = w / w.sum(-1, keepdim=True)
    return w, i.to(torch.int32)


for dtype in (torch.float16, torch.bfloat16):
    tag = str(dtype).split(".")[-1]
    print(f"=== routing ({tag}) ===", flush=True)
    torch.manual_seed(1)
    logits = torch.randn(M, E, dtype=dtype, device=DEV)
    hs = torch.randn(M, 128, dtype=dtype, device=DEV)
    bias = torch.randn(E, dtype=torch.float32, device=DEV)

    for renorm in (True, False):
        w, i = _route_moe(logits, TOPK, "softmax", renorm, None, None, None)
        report(
            f"softmax topk, renormalize={renorm}",
            *softmax_topk_ref(logits, renorm),
            w,
            i,
        )

    # What the layer can hand fused_moe but _route_moe drops on the floor.
    gw, gi = grouped_topk(hs, logits.float(), TOPK, True, 4, 2, "softmax")
    w, i = _route_moe(logits, TOPK, "softmax", True, 4, 2, None)
    report("softmax + use_grouped_topk", gw, gi, w, i, expect_match=False)

    # Ignored-argument checks: identical output with and without the argument
    # proves _route_moe never looks at it.
    w0, i0 = _route_moe(logits, TOPK, "softmax", True, None, None, None)
    w1, i1 = _route_moe(logits, TOPK, "softmax", True, 4, 2, None)
    report("group args change _route_moe output?", w0, i0, w1, i1)
    w2, i2 = _route_moe(logits, TOPK, "softmax", True, None, None, bias)
    report("e_score_correction_bias changes output?", w0, i0, w2, i2)

    # The sigmoid branch as written raises; call the kernel with the kwarg it
    # actually takes to see whether the routing itself would be right.
    try:
        _route_moe(logits, TOPK, "sigmoid", True, 4, 2, bias)
        print("  sigmoid branch: no exception (already fixed?)")
    except TypeError as exc:
        print(f"  sigmoid branch raises TypeError: {exc}", flush=True)
    score = torch.empty(M, TOPK, dtype=torch.float32, device=DEV)
    ids = torch.empty(M, TOPK, dtype=torch.int32, device=DEV)
    kunlun_import = __import__("kunlun_ops")
    kunlun_import.moe_sigmoid_group_topk_norm(
        x=logits.float(),
        topk_index=ids,
        norm_score=score,
        block_statistic=torch.zeros(
            (_MOE_BLOCK_STATISTIC_ROWS, E), dtype=torch.int32, device=DEV
        ),
        bias=bias,
        scale=1.0,
        n_group=4,
        topk_group=2,
    )
    sw, si = grouped_topk(
        hs, logits.float(), TOPK, True, 4, 2, "sigmoid", e_score_correction_bias=bias
    )
    report("sigmoid grouped + bias (kwarg fixed by hand)", sw, si, score, ids)
    sw25, _ = grouped_topk(
        hs,
        logits.float(),
        TOPK,
        True,
        4,
        2,
        "sigmoid",
        routed_scaling_factor=2.5,
        e_score_correction_bias=bias,
    )
    print(
        f"  routed_scaling_factor=2.5 -> upstream weights are "
        f"{(sw25.abs().max() / sw.abs().max()).item():.2f}x the scale=1.0 "
        f"kernel output",
        flush=True,
    )

print("=== edge cases ===", flush=True)
for dtype in (torch.float16, torch.bfloat16):
    tag = str(dtype).split(".")[-1]
    w13 = torch.randn(E, 128, 128, dtype=dtype, device=DEV) / 16
    w2 = torch.randn(E, 128, 64, dtype=dtype, device=DEV) / 16
    for nt in (0, 1):
        x = torch.randn(nt, 128, dtype=dtype, device=DEV)
        lg = torch.randn(nt, E, dtype=dtype, device=DEV)
        try:
            out = KunlunOps.fused_moe(x, w13, w2, lg, TOPK, True)
            torch.cuda.synchronize()
            print(f"  {tag} num_tokens={nt} -> {tuple(out.shape)} ok")
        except Exception as exc:
            print(
                f"  {tag} num_tokens={nt} -> {type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:100]}"
            )

print("=== summary ===")
print("unexpected results:", FAILS or "none")
