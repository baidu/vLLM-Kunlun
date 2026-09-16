import argparse
import statistics
import time

import torch
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_kunlun.ops._kunlun_ops import KunlunOps


def old_ep(x, w13, w2, logits, rank, topk, renorm):
    batch, hidden = x.shape
    local_experts, gate_up, _ = w13.shape
    weights = torch.empty(batch, topk, dtype=logits.dtype, device=logits.device)
    ids = torch.empty(batch, topk, dtype=torch.int32, device=logits.device)
    block = torch.empty(0, dtype=torch.int32, device=logits.device)
    torch.ops._C.moe_softmax_topk(logits, weights, ids, block)
    if renorm:
        weights = weights / weights.sum(1, keepdim=True)
    weights = weights.to(x.dtype)
    out = torch.zeros(batch * topk, hidden, dtype=x.dtype, device=x.device)
    repeated = x.repeat_interleave(topk, dim=0)
    flat_ids = ids.flatten()
    for i in range(local_experts):
        selected = flat_ids == rank * local_experts + i
        if selected.sum():
            tokens = repeated[selected]
            gate = torch.empty(
                selected.sum(), gate_up // 2, dtype=x.dtype, device=x.device
            )
            torch.ops._C.silu_and_mul(gate, tokens @ w13[i].T)
            out[selected] = gate @ w2[i].T
    return (out.view(batch, topk, hidden) * weights.unsqueeze(2)).sum(1).to(x.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=("old", "new"), required=True)
    p.add_argument("--m", type=int, required=True)
    p.add_argument(
        "--routing", choices=("balanced", "skewed", "none"), default="balanced"
    )
    p.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    p.add_argument("--execution", choices=("graph", "eager"), default="graph")
    p.add_argument("--replays", type=int, default=100)
    p.add_argument("--rounds", type=int, default=7)
    args = p.parse_args()

    device = torch.device("cuda")
    init_workspace_manager(device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(910 + args.m)
    hidden, inter, global_e, local_e, topk, rank = 2816, 704, 128, 32, 8, 0
    x = torch.randn(args.m, hidden, dtype=dtype, device=device) / 8
    w13 = torch.randn(local_e, 2 * inter, hidden, dtype=dtype, device=device) / 32
    w2 = torch.randn(local_e, hidden, inter, dtype=dtype, device=device) / 32
    if args.routing == "balanced":
        logits = torch.randn(args.m, global_e, dtype=dtype, device=device)
    elif args.routing == "skewed":
        logits = torch.full((args.m, global_e), -10, dtype=dtype, device=device)
        logits[:, :topk] = 10
    else:
        logits = torch.full((args.m, global_e), -10, dtype=dtype, device=device)
        logits[:, local_e : local_e + topk] = 10

    if args.impl == "old":

        def fn():
            return old_ep(x, w13, w2, logits, rank, topk, True)

    else:

        def fn():
            return KunlunOps.fused_moe_ep(
                x, w13, w2, logits, rank, topk, True, activation="silu"
            )

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if args.execution == "graph":
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                out = fn()
            graph.replay()
            torch.cuda.synchronize()
        except Exception as exc:
            print(
                f"CAPTURE_FAILED impl={args.impl} M={args.m} routing={args.routing} "
                f"dtype={args.dtype} error={type(exc).__name__}:{exc}",
                flush=True,
            )
            return
        replay = graph.replay
    else:
        out = fn()
        torch.cuda.synchronize()
        replay = fn

    samples = []
    for _ in range(args.rounds):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.replays):
            replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / args.replays)
    print(
        f"RESULT impl={args.impl} execution={args.execution} M={args.m} routing={args.routing} dtype={args.dtype} "
        f"median_us={statistics.median(samples):.3f} min_us={min(samples):.3f} "
        f"max_us={max(samples):.3f} mean_abs={out.float().abs().mean().item():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
