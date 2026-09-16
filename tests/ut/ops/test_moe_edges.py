"""Edge cases around fused_moe's batch size and workspace budget.

Two things the value-comparison sweeps cannot see: an empty batch (every kernel
in the chain rejects it, so it has to be short-circuited) and a workspace that is
*larger* than the live ranges need (under-sizing shows up as wrong numbers, but
over-sizing is invisible except as memory).
"""

import torch
from vllm.v1.worker import workspace as WS
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_kunlun.ops._kunlun_ops import KunlunOps
from vllm_kunlun.ops.moe.preprocess import MOE_PREPROCESS_THRESHOLD

DEV = torch.device("cuda")
init_workspace_manager(DEV)

HIDDEN, INTER, E, TOPK = 2816, 704, 32, 8
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


def weights(dtype):
    torch.manual_seed(0)
    return (
        torch.randn(E, 2 * INTER, HIDDEN, dtype=dtype, device=DEV) / 16,
        torch.randn(E, HIDDEN, INTER, dtype=dtype, device=DEV) / 16,
    )


def call(M, act, dtype, **kw):
    w13, w2 = weights(dtype)
    x = torch.randn(M, HIDDEN, dtype=dtype, device=DEV) / 8
    lg = torch.randn(M, E, dtype=dtype, device=DEV)
    _requests.clear()
    out = KunlunOps.fused_moe(x, w13, w2, lg, TOPK, True, activation=act, **kw)
    torch.cuda.synchronize()
    return out


print("=== empty batch ===", flush=True)
for dtype in (torch.float16, torch.bfloat16):
    for act in ("silu", "gelu_tanh", "swigluoai", "swiglustep"):
        try:
            out = call(0, act, dtype)
            ok = (
                tuple(out.shape) == (0, HIDDEN)
                and out.dtype == dtype
                and out.device.type == DEV.type
            )
            check(
                f"M=0 {str(dtype).split('.')[-1]} {act}",
                ok,
                f"shape={tuple(out.shape)} dtype={out.dtype}",
            )
        except Exception as exc:  # noqa: BLE001
            check(
                f"M=0 {str(dtype).split('.')[-1]} {act}",
                False,
                f"{type(exc).__name__}: {exc}",
            )

# A grouped/biased config takes a different routing kernel, which is where the
# empty batch used to blow up first.
bias = torch.randn(E, dtype=torch.float32, device=DEV)
try:
    out = call(
        0,
        "silu",
        torch.bfloat16,
        scoring_func="sigmoid",
        use_grouped_topk=True,
        num_expert_group=4,
        topk_group=2,
        e_score_correction_bias=bias,
    )
    check("M=0 grouped sigmoid+bias", tuple(out.shape) == (0, HIDDEN))
except Exception as exc:  # noqa: BLE001
    check("M=0 grouped sigmoid+bias", False, f"{type(exc).__name__}: {exc}")

print("=== workspace budget (bf16, unfused path) ===", flush=True)
# Live ranges, from fused_moe's comment:
#   workspace_a  expert-sorted tokens (only above the preprocess threshold)
#                -> activation output
#   workspace_b  W13 output -> expert output
# swigluoai has no out param, so it allocates its own activation output and
# workspace_a must not reserve a slice for it.
for act in ("silu", "gelu_tanh", "swigluoai"):
    for M in (16, 96, 512):
        call(M, act, torch.bfloat16)
        rows = M * TOPK
        got_a, got_b, got_metadata = _requests[-1]
        want_a = 0 if act == "swigluoai" else M * HIDDEN
        if act != "swigluoai":
            want_a = max(want_a, rows * INTER)
        if rows > MOE_PREPROCESS_THRESHOLD:
            want_a = max(want_a, rows * HIDDEN)
        want_b = max(rows * 2 * INTER, rows * HIDDEN)
        want_metadata = 12 * E + 2 * E + 1 + rows
        check(
            f"{act:<10} M={M:<4} rows={rows:<5}",
            (got_a, got_b, got_metadata) == (want_a, want_b, want_metadata),
            f"a={got_a} (want {want_a})  b={got_b} (want {want_b})  "
            f"metadata={got_metadata} (want {want_metadata})",
        )

print("=== summary ===")
print("failures:", FAILS or "none")
raise SystemExit(1 if FAILS else 0)
