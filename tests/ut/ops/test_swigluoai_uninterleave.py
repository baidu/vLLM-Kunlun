"""The interleaved and permuted swigluoai paths must agree, bit-for-bit closely.

`uninterleave_moe_w13` reorders W13's rows so the packed-layout Kunlun kernel can
run gpt-oss's activation. That is only sound if fused_moe over (permuted W13,
"swigluoai_uninterleave") computes what fused_moe over (original W13,
"swigluoai") computes. This drives both through the real entry point and
compares, then checks the workspace request and the alpha/beta/limit plumbing.
"""

import time

import torch
from vllm.v1.worker import workspace as WS
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_kunlun.ops._kunlun_ops import KunlunOps
from vllm_kunlun.ops.moe.activation import uninterleave_moe_w13
from vllm_kunlun.ops.moe.preprocess import MOE_PREPROCESS_THRESHOLD

DEV = torch.device("cuda")
init_workspace_manager(DEV)

HIDDEN, INTER, E, TOPK = 512, 704, 32, 8
FAILS = []

_mgr_cls = type(WS.current_workspace_manager())
_orig_get = _mgr_cls.get_simultaneous
_requests = []


def _spy(self, *shapes_and_dtypes):
    _requests.append([s[0][0] for s in shapes_and_dtypes])
    return _orig_get(self, *shapes_and_dtypes)


_mgr_cls.get_simultaneous = _spy


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL':>4}] {name} {detail}", flush=True)
    if not cond:
        FAILS.append(name)


class Layer:
    """Just enough of RoutedExperts for uninterleave_moe_w13 to chew on."""

    def __init__(self, w13, w13_bias=None):
        self.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        if w13_bias is not None:
            self.w13_bias = torch.nn.Parameter(w13_bias, requires_grad=False)


def inputs(dtype, M, seed=0, with_bias=False):
    torch.manual_seed(seed)
    w13 = torch.randn(E, 2 * INTER, HIDDEN, dtype=dtype, device=DEV) / 16
    w2 = torch.randn(E, HIDDEN, INTER, dtype=dtype, device=DEV) / 16
    x = torch.randn(M, HIDDEN, dtype=dtype, device=DEV) / 8
    lg = torch.randn(M, E, dtype=dtype, device=DEV)
    w13_bias = w2_bias = None
    if with_bias:
        w13_bias = torch.randn(E, 2 * INTER, dtype=dtype, device=DEV) / 8
        w2_bias = torch.randn(E, HIDDEN, dtype=dtype, device=DEV) / 8
    return w13, w2, x, lg, w13_bias, w2_bias


def run(w13, w2, x, lg, act, w13_bias=None, w2_bias=None, **kw):
    _requests.clear()
    out = KunlunOps.fused_moe(
        x,
        w13,
        w2,
        lg,
        TOPK,
        True,
        activation=act,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        **kw,
    )
    torch.cuda.synchronize()
    return out


print("=== permuted path == interleaved path ===", flush=True)
for dtype in (torch.float16, torch.bfloat16):
    for with_bias in (False, True):
        for M in (1, 16, 96, 512):
            w13, w2, x, lg, w13_bias, w2_bias = inputs(
                dtype, M, seed=M, with_bias=with_bias
            )
            ref = run(w13, w2, x, lg, "swigluoai", w13_bias, w2_bias).clone()
            layer = Layer(w13.clone(), None if w13_bias is None else w13_bias.clone())
            name = uninterleave_moe_w13(layer, "swigluoai")
            got = run(
                layer.w13_weight,
                w2,
                x,
                lg,
                name,
                getattr(layer, "w13_bias", None),
                w2_bias,
            )
            scale = ref.float().abs().max().item()
            err = (got.float() - ref.float()).abs().max().item() / max(scale, 1e-9)
            check(
                f"{str(dtype).split('.')[-1]:<9} bias={int(with_bias)} M={M:<4}",
                name == "swigluoai_uninterleave" and err < 2e-2,
                f"act={name} rel_err={err:.3e} scale={scale:.3f}",
            )

print("=== pass-through for the other activations ===", flush=True)
for act in ("silu", "gelu_tanh", "swiglustep", "swigluoai_uninterleave"):
    w13, _, _, _, w13_bias, _ = inputs(torch.bfloat16, 8, with_bias=True)
    layer = Layer(w13.clone(), w13_bias.clone())
    name = uninterleave_moe_w13(layer, act)
    untouched = torch.equal(layer.w13_weight.data, w13) and torch.equal(
        layer.w13_bias.data, w13_bias
    )
    check(f"{act:<24} unchanged", name == act and untouched, f"-> {name}")

print("=== W13 permutation is a pure row reorder ===", flush=True)
w13, _, _, _, w13_bias, _ = inputs(torch.bfloat16, 8, with_bias=True)
layer = Layer(w13.clone(), w13_bias.clone())
uninterleave_moe_w13(layer, "swigluoai")
p = layer.w13_weight.data
check(
    "gate rows == even rows",
    torch.equal(p[:, :INTER, :], w13[:, 0::2, :]),
)
check(
    "up rows == odd rows",
    torch.equal(p[:, INTER:, :], w13[:, 1::2, :]),
)
check("permuted W13 is contiguous", p.is_contiguous())
check(
    "bias permuted the same way",
    torch.equal(layer.w13_bias.data[:, :INTER], w13_bias[:, 0::2])
    and torch.equal(layer.w13_bias.data[:, INTER:], w13_bias[:, 1::2]),
)

print("=== workspace: the packed path fills an out param again ===", flush=True)
for M in (16, 96, 512):
    w13, w2, x, lg, _, _ = inputs(torch.bfloat16, M, seed=M)
    rows = M * TOPK
    run(w13, w2, x, lg, "swigluoai")
    a_inter = _requests[-1][0]
    run(w13, w2, x, lg, "swigluoai_uninterleave")
    a_packed = _requests[-1][0]
    output_numel = M * HIDDEN
    expanded = rows * HIDDEN if rows > MOE_PREPROCESS_THRESHOLD else 0
    want_inter = expanded
    want_packed = max(rows * INTER, expanded, output_numel)
    check(
        f"M={M:<4} rows={rows:<5}",
        (a_inter, a_packed) == (want_inter, want_packed),
        f"interleaved a={a_inter} (want {want_inter})  "
        f"packed a={a_packed} (want {want_packed})",
    )

print("=== alpha/beta/limit are read, not ignored ===", flush=True)
w13, w2, x, lg, _, _ = inputs(torch.bfloat16, 64, seed=7)
base = run(w13, w2, x, lg, "swigluoai_uninterleave").clone()
for kw in (
    {"swiglu_alpha": 1.0},
    {"swiglu_beta": 0.0},
    {"swiglu_limit": 0.5},
):
    got = run(w13, w2, x, lg, "swigluoai_uninterleave", **kw)
    delta = (got.float() - base.float()).abs().max().item()
    check(f"packed honours {list(kw)[0]}", delta > 1e-3, f"delta={delta:.3e}")

# The interleaved op hardcodes `up + 1`, so a non-default beta must be refused
# rather than silently dropped.
for kw, want_change in (
    ({"swiglu_alpha": 1.0}, True),
    ({"swiglu_limit": 0.5}, True),
):
    got = run(w13, w2, x, lg, "swigluoai", **kw).clone()
    ref = run(w13, w2, x, lg, "swigluoai")
    delta = (got.float() - ref.float()).abs().max().item()
    check(
        f"interleaved honours {list(kw)[0]}",
        (delta > 1e-3) == want_change,
        f"delta={delta:.3e}",
    )
try:
    run(w13, w2, x, lg, "swigluoai", swiglu_beta=0.0)
    check("interleaved rejects beta != 1", False, "no exception")
except NotImplementedError as exc:
    check("interleaved rejects beta != 1", True, f"{exc}")

print("=== M=0 on the packed path ===", flush=True)
for dtype in (torch.float16, torch.bfloat16):
    w13, w2, x, lg, _, _ = inputs(dtype, 0)
    out = run(w13, w2, x, lg, "swigluoai_uninterleave")
    check(
        f"M=0 {str(dtype).split('.')[-1]}",
        tuple(out.shape) == (0, HIDDEN) and out.dtype == dtype,
        f"shape={tuple(out.shape)}",
    )


def bench(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


print("=== end-to-end fused_moe timing (bf16) ===", flush=True)
for M in (1, 8, 64, 512):
    w13, w2, x, lg, _, _ = inputs(torch.bfloat16, M, seed=M)
    t_i = bench(
        lambda: KunlunOps.fused_moe(x, w13, w2, lg, TOPK, True, activation="swigluoai")
    )
    t_p = bench(
        lambda: KunlunOps.fused_moe(
            x, w13, w2, lg, TOPK, True, activation="swigluoai_uninterleave"
        )
    )
    print(
        f"  M={M:<5} interleaved={t_i * 1e6:8.1f} us  "
        f"packed={t_p * 1e6:8.1f} us  speedup={t_i / t_p:5.2f}x",
        flush=True,
    )

print("=== summary ===")
print("failures:", FAILS or "none")
raise SystemExit(1 if FAILS else 0)
