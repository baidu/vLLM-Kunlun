"""Does dropping the fp32 cast on router_logits change any routing kernel?

`run_moe_router` used to cast router_logits to fp32 on the common path. Only
moe_fused_gate_dsv4 needs fp32, so the cast moved into that branch. This drives
every kernel twice -- once with the model's own half dtype, once with an fp32
copy, i.e. exactly the pre-change behaviour -- and reports whether the results
still agree.
"""

import time

import torch

import vllm_kunlun  # noqa: F401  registers the ops
from vllm_kunlun.ops.moe.router import make_moe_router, run_moe_router

DEV = torch.device("cuda:0")
E = 128
CASES = (
    ("softmax_topk renorm", dict(scoring_func="softmax", renormalize=True)),
    ("softmax_topk raw", dict(scoring_func="softmax", renormalize=False)),
    (
        "group_topk softmax",
        dict(
            scoring_func="softmax",
            renormalize=True,
            use_grouped_topk=True,
            num_expert_group=8,
            topk_group=4,
        ),
    ),
    (
        "group_topk sigmoid",
        dict(scoring_func="sigmoid", renormalize=True),
    ),
    (
        "sigmoid_group +bias",
        dict(
            scoring_func="sigmoid",
            renormalize=True,
            use_grouped_topk=True,
            num_expert_group=8,
            topk_group=4,
            has_e_score_correction_bias=True,
            routed_scaling_factor=2.5,
        ),
    ),
    (
        "fused_gate +bias",
        dict(
            scoring_func="sigmoid",
            renormalize=True,
            has_e_score_correction_bias=True,
            routed_scaling_factor=2.5,
        ),
    ),
)


def run(router, logits, bias):
    return run_moe_router(
        router,
        logits,
        e_score_correction_bias=bias,
        want_block_statistic=True,
    )


def bench(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def main():
    torch.manual_seed(0)
    fails = []
    for dtype in (torch.float16, torch.bfloat16):
        print(f"=== {dtype} ===")
        for name, kw in CASES:
            router = make_moe_router(top_k=8, **kw)
            bias = (
                torch.randn(E, dtype=torch.float32, device=DEV) * 0.1
                if kw.get("has_e_score_correction_bias")
                else None
            )
            for m in (1, 32, 1024):
                half = (torch.randn(m, E, device=DEV) * 3).to(dtype)
                wide = half.to(torch.float32)
                w_h, i_h, b_h = run(router, half, bias)
                w_w, i_w, b_w = run(router, wide, bias)
                same_ids = torch.equal(i_h, i_w)
                same_bs = torch.equal(b_h, b_w)
                same_w = torch.equal(w_h, w_w)
                err = (w_h - w_w).abs().max().item()
                ok = same_ids and same_bs and same_w
                if not ok:
                    fails.append(
                        f"{dtype} {router.kernel} {name} M={m} "
                        f"ids={same_ids} bs={same_bs} w={same_w} err={err:.3g}"
                    )
                print(
                    f"  {router.kernel:14s} {name:20s} M={m:<5d} "
                    f"ids={'=' if same_ids else 'X'} "
                    f"bs={'=' if same_bs else 'X'} "
                    f"w={'=' if same_w else 'X'} max|dw|={err:.3g}"
                )

    # What the removed cast used to cost, at the deployed shape.
    logits = torch.randn(32, E, device=DEV, dtype=torch.float16)
    us = bench(lambda: logits.to(torch.float32))
    print(f"\ncast alone: {us:.2f} us/call (eager, M=32 E={E})")

    if fails:
        print("\nFAIL")
        for f in fails:
            print(" ", f)
    else:
        print("\nall kernels agree with the pre-change fp32 path")


if __name__ == "__main__":
    main()
