"""Correctness and workspace checks for KunlunOps.fused_moe_int8.

There is no int8 MoE checkpoint on this box, so the weights are synthesised:
`probe_int8_moe.py` established that quant2d writes the per-row max and that
int8 moe_fc computes `(x_q @ w_q.T) * x_scale * w_scale / 127**2`, which is what
lets a reference exist at all. The reference calls the same quant2d and
silu_and_mul the chain calls, so what is left to compare is the routing, the
expert grouping, the scale bookkeeping and the workspace slicing -- not
quantization noise.
"""

import torch
from vllm.v1.worker import workspace as WS
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_kunlun.ops._kunlun_ops import MOE_PREPROCESS_THRESHOLD, KunlunOps
from vllm_kunlun.ops.moe.router import make_moe_router, run_moe_router

DEV = torch.device("cuda")
init_workspace_manager(DEV)

HIDDEN, INTER, E, TOPK = 512, 256, 32, 4
FAILS = []

_mgr_cls = type(WS.current_workspace_manager())
_orig_get = _mgr_cls.get_simultaneous
_requests = []


def _spy(self, *shapes_and_dtypes):
    _requests.append([(s, d) for s, d in shapes_and_dtypes])
    return _orig_get(self, *shapes_and_dtypes)


_mgr_cls.get_simultaneous = _spy


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL':>4}] {name} {detail}", flush=True)
    if not cond:
        FAILS.append(name)


def quantize_weight(w):
    """Per-output-channel int8 weight plus the max scale moe_fc expects."""
    w_max = w.abs().amax(dim=2, keepdim=True)
    w_q = (w / w_max * 127).round().clamp(-127, 127).to(torch.int8)
    return w_q, w_max.squeeze(-1).contiguous()


def quant2d(src):
    rows, cols = src.shape
    q = torch.empty(rows, cols, dtype=torch.int8, device=src.device)
    scale = torch.empty(rows, 1, dtype=torch.float32, device=src.device)
    torch.ops._C.quant2d(src.contiguous(), q, scale, force_sdnn=True)
    return q, scale


def int8_gemm(x, w_q, w_scale, dtype):
    """One expert's w8a8 GEMM, quantizing x the way the chain does."""
    x_q, x_scale = quant2d(x)
    acc = x_q.float() @ w_q.float().T
    return (acc * x_scale * w_scale / 127**2).to(dtype)


def ref_moe_int8(x, w13_q, w13_s, w2_q, w2_s, topk_ids, score):
    out = torch.zeros(x.shape[0], w2_q.shape[1], dtype=torch.float32, device=x.device)
    for e in range(w13_q.shape[0]):
        rows, ks = (topk_ids == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        h = int8_gemm(x[rows], w13_q[e], w13_s[e], x.dtype)
        act = torch.empty(rows.numel(), INTER, dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(act, h)
        y = int8_gemm(act, w2_q[e], w2_s[e], x.dtype)
        out.index_add_(0, rows, y.float() * score[rows, ks].unsqueeze(1))
    return out


def inputs(dtype, M, seed=0, hidden=HIDDEN, inter=INTER):
    torch.manual_seed(seed)
    w13 = torch.randn(E, 2 * inter, hidden, dtype=torch.float32, device=DEV) / 16
    w2 = torch.randn(E, hidden, inter, dtype=torch.float32, device=DEV) / 16
    x = (torch.randn(M, hidden, dtype=dtype, device=DEV) / 8).contiguous()
    lg = torch.randn(M, E, dtype=dtype, device=DEV)
    return (*quantize_weight(w13), *quantize_weight(w2), x, lg)


def run(w13_q, w13_s, w2_q, w2_s, x, lg, router):
    _requests.clear()
    out = KunlunOps.fused_moe_int8(x, w13_q, w13_s, w2_q, w2_s, lg, TOPK, router)
    torch.cuda.synchronize()
    return out


print("=== fused_moe_int8 == per-expert w8a8 reference ===", flush=True)
for dtype in (torch.float16, torch.bfloat16):
    for renormalize in (True, False):
        for M in (1, 17, 96, 200, 2048):
            w13_q, w13_s, w2_q, w2_s, x, lg = inputs(dtype, M, seed=M)
            router = make_moe_router(
                top_k=TOPK, scoring_func="softmax", renormalize=renormalize
            )
            score, topk_ids = run_moe_router(router, lg)[:2]
            got = run(w13_q, w13_s, w2_q, w2_s, x, lg, router).float()
            ref = ref_moe_int8(x, w13_q, w13_s, w2_q, w2_s, topk_ids, score)
            scale = ref.abs().max().item()
            err = (got - ref).abs().max().item() / scale
            check(
                f"{str(dtype).split('.')[-1]:<9} renorm={int(renormalize)} "
                f"M={M:<5} pre={'small' if M * TOPK <= 768 else 'sorted'}",
                torch.isfinite(got).all() and err < 3e-2,
                f"rel_err={err:.3e} scale={scale:.3f}",
            )

print("=== M=0 returns an empty output instead of dying ===", flush=True)
for dtype in (torch.float16, torch.bfloat16):
    w13_q, w13_s, w2_q, w2_s, x, lg = inputs(dtype, 0)
    router = make_moe_router(top_k=TOPK, scoring_func="softmax", renormalize=True)
    out = run(w13_q, w13_s, w2_q, w2_s, x, lg, router)
    check(
        f"M=0 {str(dtype).split('.')[-1]}",
        tuple(out.shape) == (0, HIDDEN) and out.dtype == dtype,
        f"shape={tuple(out.shape)} dtype={out.dtype}",
    )
    check(
        f"M=0 {str(dtype).split('.')[-1]} asks for no workspace",
        _requests == [],
        f"requests={_requests}",
    )

print("=== workspace budget matches the documented live ranges ===", flush=True)
# The three shapes make a different term win each max(): inter < hidden with
# gate_up == hidden, inter > hidden, and inter so small that the expert output
# outgrows the W13 output and the expanded tokens outgrow the activation.
for hidden, inter in ((512, 256), (512, 704), (512, 128)):
    for M in (17, 200):
        w13_q, w13_s, w2_q, w2_s, x, lg = inputs(
            torch.bfloat16, M, seed=M, hidden=hidden, inter=inter
        )
        router = make_moe_router(top_k=TOPK, scoring_func="softmax", renormalize=True)
        run(w13_q, w13_s, w2_q, w2_s, x, lg, router)
        rows = M * TOPK
        expanded = rows * hidden if rows > MOE_PREPROCESS_THRESHOLD else 0
        metadata = 12 * E + 2 * E + 1 + rows
        want = [
            ((max(expanded, rows * inter, M * hidden),), torch.bfloat16),
            ((max(rows * 2 * inter, rows * hidden),), torch.bfloat16),
            ((rows * max(hidden, inter),), torch.int8),
            ((rows,), torch.float32),
            ((metadata,), torch.int32),
        ]
        check(
            f"hidden={hidden} inter={inter:<4} M={M:<4} rows={rows:<5}",
            _requests[-1] == want,
            f"got={_requests[-1]}" if _requests[-1] != want else "",
        )

print("=== summary ===")
print("failures:", FAILS or "none")
raise SystemExit(1 if FAILS else 0)
