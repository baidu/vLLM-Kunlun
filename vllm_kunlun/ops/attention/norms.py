"""DeepSeek-V4 RMSNorm shortcuts with a guarded native/fallback path.

The V4 model uses three normalization hot paths:

1. ``rms_norm(x, weight, eps)`` -- standard input-only RMSNorm.
2. ``fused_add_rms_norm(x, residual, weight, eps)`` -- fused addition +
   in-place normalization.
3. ``fused_q_kv_rmsnorm(q, kv, q_weight, kv_weight, eps)`` -- Q and KV
   normalizations used by the MLA attention helper.

Upstream stores pure-PyTorch implementations as free functions referenced from
:class:`vllm.model_executor.layers.layernorm.RMSNorm.forward_native`.  This
adapter leaves that upstream module untouched; instead it registers post-import
hooks for the V4 modules that replace those function references with thin
capability-probed wrappers around :obj:`torch.ops._C.rmsnorm` /
:obj:`torch.ops._C.add_rmsnorm` while preserving an unchanged fallback to the
original formulas if native kernels are unavailable or fail at runtime.
"""
import logging
from functools import wraps
from typing import Callable, List, Tuple

import torch

from ...ops.layernorm import KunlunRMSNorm
from vllm_kunlun.runtime_utils import WarningOnce, find_op

LOGGER = logging.getLogger("vllm_kunlun.ops.attention.norms")

# ---------------------------------------------------------------------------
# Capability probe cache (lazy, once per process)
# ---------------------------------------------------------------------------
_NATIVE_OP_CACHE: dict[str, object] = {}
_FAILED_KEYS: set[str] = set()
_MISSING_KEYS_LOGGED: set[str] = set()
_APPLIED_MODEL_KEY = "_dsv4_norms_model_applied"
_ATTN_FN_LABEL_ATTR = "_dsv4_adapter_label"
_FALSE = object()


def _native_op(name: str):
    """Return cached callable handle or None when absent/unimplemented."""
    value = _NATIVE_OP_CACHE.get(name, _FALSE)
    if value is not _FALSE:
        return value if value is not None else None
    try:
        op = find_op(torch.ops._C, name)
        # Some ops appear in dispatcher tables but raise RuntimeError on call;
        # we keep the real object here and warn only when actually executed.
        _NATIVE_OP_CACHE[name] = op
    except Exception:  # noqa: BLE001
        op = None
        _NATIVE_OP_CACHE[name] = None
    return op


def _warn_once_missing(kind: str) -> None:
    if kind in _MISSING_KEYS_LOGGED:
        return
    _MISSING_KEYS_LOGGED.add(kind)
    WarningOnce.emit(
        f"rmsnorm-missing-native:{kind}",
        "%s native kernel is missing/disabled; using reference Python RMSNorm",
        kind,
    )


def _warn_failed(kind: str, exc: BaseException) -> None:
    key = f"{kind}-exec-{type(exc).__name__}"
    if key in _FAILED_KEYS:
        return
    _FAILED_KEYS.add(key)
    LOGGER.warning(
        "%s native kernel failed once (%s); falling back to reference Python RMSNorm",
        kind,
        exc,
    )


# ---------------------------------------------------------------------------
# Reference-level wrappers (installed into V4 modules' namespaces)
# ---------------------------------------------------------------------------
def _make_rms_norm_wrapper(original_rms_norm: Callable) -> Callable:
    def rms_norm_kunlun(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        native = _native_op("rmsnorm")
        if native is not None:
            out = torch.empty_like(x)
            try:
                native(x.contiguous(), weight, out, eps)
                return out
            except Exception as exc:  # noqa: BLE001
                _warn_failed("torch.ops._C.rmsnorm", exc)
        else:
            _warn_once_missing("torch.ops._C.rmsnorm")
        return original_rms_norm(x, weight, eps)

    setattr(rms_norm_kunlun, "_dsv4_adapter_label", True)
    return rms_norm_kunlun


def _make_fused_add_rms_norm_wrapper(original_fn: Callable) -> Callable:
    def fused_add_rms_norm_kunlun(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        native = _native_op("add_rmsnorm")
        if native is not None:
            try:
                # Matches KunlunRMSNorm.forward_oot contract:
                # x receives normalized output, residual carries added input.
                native(x, residual, weight=weight, eps=eps)
                return x, residual
            except Exception as exc:  # noqa: BLE001
                _warn_failed("torch.ops._C.add_rmsnorm", exc)
        else:
            _warn_once_missing("torch.ops._C.add_rmsnorm")
        return original_fn(x, residual, weight, eps)

    setattr(fused_add_rms_norm_kunlun, "_dsv4_adapter_label", True)
    return fused_add_rms_norm_kunlun


def _patch_forward_native_globals(rms_symbol: type, *, source_tag: str) -> List[str]:
    forward_native = getattr(rms_symbol.forward_native, "func", rms_symbol.forward_native)
    glb = getattr(forward_native, "__globals__", None)
    if glb is None:
        return []

    labels = []
    orig_rms = glb.get("rms_norm")
    if orig_rms is not None and not getattr(orig_rms, "_dsv4_adapter_label", False):
        glb["rms_norm"] = _make_rms_norm_wrapper(orig_rms)
        labels.append(f"{source_tag}.global:rms_norm")

    orig_fused = glb.get("fused_add_rms_norm")
    if orig_fused is not None and not getattr(orig_fused, "_dsv4_adapter_label", False):
        glb["fused_add_rms_norm"] = _make_fused_add_rms_norm_wrapper(orig_fused)
        labels.append(f"{source_tag}.global:fused_add_rms_norm")

    return labels


def _wire_instance_method_overrides(v4_model_module: object) -> List[str]:
    patched_classes = []

    def patch_instances(root_object):
        for child in getattr(root_object, "modules", lambda: ())():
            if child.__class__.__name__ == "RMSNorm":
                child._forward_method = KunlunRMSNorm.forward_oot.__get__(child, child.__class__)  # noqa: B010

    for class_name in (
        "DeepseekV4DecoderLayer",
        "DeepseekV4ForCausalLM",
    ):
        model_cls = getattr(v4_model_module, class_name, None)
        if model_cls is None:
            continue
        init_func = model_cls.__init__
        if getattr(init_func, "_dsv4_rms_init_patched", False):
            continue

        old_init = init_func

        def make_new(oi=old_init):
            @wraps(oi)
            def new_init(self, *args, **kwargs):
                oi(self, *args, **kwargs)
                patch_instances(self)

            new_init._dsv4_rms_init_patched = True
            new_init.__doc__ = getattr(oi, "__doc__", None)
            return new_init

        model_cls.__init__ = make_new()
        patched_classes.append(class_name)

    return [
        f"deepseek_v4.{cls.lower()}_init.instance_override"
        for cls in patched_classes
    ]


# ---------------------------------------------------------------------------
# Attention-specific fused_qkv helper wrapper
# ---------------------------------------------------------------------------
def _make_attention_qkv_wrapper(original_attn_fn: Callable) -> Callable:
    def fused_q_kv_rmsnorm_kunlan(
        q: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        native = _native_op("rmsnorm")
        if native is None:
            _warn_once_missing("torch.ops._C.rmsnorm")
            return original_attn_fn(q, kv, q_weight, kv_weight, eps)

        outputs = [None, None]
        tensors_and_weights = ((q, q_weight), (kv, kv_weight))
        success = True
        for i, (tensor, wgt) in enumerate(tensors_and_weights):
            tmp_out = torch.empty_like(tensor)
            try:
                native(tensor.contiguous(), wgt, tmp_out, eps)
                outputs[i] = tmp_out
            except Exception as exc:  # noqa: BLE001
                _warn_failed("torch.ops._C.rmsnorm", exc)
                success = False
                break
        if success and all(o is not None for o in outputs):
            return tuple(outputs)
        return original_attn_fn(q, kv, q_weight, kv_weight, eps)

    setattr(fused_q_kv_rmsnorm_kunlan, _ATTN_FN_LABEL_ATTR, "rmsnorm-shortcut")
    return fused_q_kv_rmsnorm_kunlan


def _model_apply(mod: object) -> List[str]:
    if getattr(mod, _APPLIED_MODEL_KEY, False):
        return []

    rms_symbol = getattr(mod, "RMSNorm", None)
    if rms_symbol is None:
        return []

    labels: List[str] = []
    labels.extend(_patch_forward_native_globals(rms_symbol, source_tag="RMSNorm.v4_model_globals"))
    labels.extend(_wire_instance_method_overrides(mod))

    if labels:
        LOGGER.info(
            "[DSV4 adapter] wired RMSNorm shortcuts into %s (%d targets)",
            mod.__name__,
            len(labels),
        )

    setattr(mod, _APPLIED_MODEL_KEY, True)
    return labels


def _model_predicate(mod: object) -> bool:
    return bool(getattr(mod, _APPLIED_MODEL_KEY, False)) and getattr(mod.RMSNorm.forward_native.__globals__.get("rms_norm"), "_dsv4_adapter_label", False)


def _attn_apply(mod: object) -> List[str]:
    symbol_name = "fused_q_kv_rmsnorm"
    fn_old = getattr(mod, symbol_name, None)
    if fn_old is not None and getattr(fn_old, _ATTN_FN_LABEL_ATTR, None) == "rmsnorm-shortcut":
        return []

    # If there was no pre-existing helper we install our own minimal one so
    # that downstream branches can still fall back without crashing under CPU/
    # no-extension environments.
    if fn_old is None:
        def torch_only_fallback(
            q: torch.Tensor,
            kv: torch.Tensor,
            qw: torch.Tensor,
            kw: torch.Tensor,
            eps: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            variance_q = q.float().pow(2).mean(dim=-1, keepdim=True)
            variance_kv = kv.float().pow(2).mean(dim=-1, keepdim=True)
            rq = q.float() * torch.rsqrt(variance_q + eps)
            rkv = kv.float() * torch.rsqrt(variance_kv + eps)
            return (
                (rq * qw.unsqueeze(-2)).to(dtype=q.dtype),
                (rkv * kw.unsqueeze(-2)).to(dtype=kv.dtype),
            )

        replacement = _make_attention_qkv_wrapper(torch_only_fallback)
    else:
        replacement = _make_attention_qkv_wrapper(fn_old)

    setattr(mod, symbol_name, replacement)
    return [f"{mod.__name__}.{symbol_name}"]


def _attn_predicate(mod: object) -> bool:
    fn = getattr(mod, "fused_q_kv_rmsnorm", None)
    return fn is not None and getattr(fn, _ATTN_FN_LABEL_ATTR, None) == "rmsnorm-shortcut"


# ---------------------------------------------------------------------------
# Public entry point used by registry/installer
# ---------------------------------------------------------------------------
