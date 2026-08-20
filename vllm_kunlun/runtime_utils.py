"""Small, side-effect-free runtime utilities shared across vllm-kunlun.

Keeps fallback/probe/env-gating logic uniform so individual adapter modules
don't invent their own warning latches.
"""
import logging
import os
import typing
from typing import Optional, Sequence, Tuple

import torch

LOGGER = logging.getLogger("vllm_kunlun.runtime_utils")

_T = typing.TypeVar("_T")


class WarningOnce:
    """Process-wide latch that emits exactly one warning per unique *key*."""

    _seen_keys: "set[str]" = set()

    @classmethod
    def clear(cls):
        """Test-only hook allowing resetting warning state."""
        cls._seen_keys.clear()

    @classmethod
    def emit(cls, key: str, fmt: str, *args) -> None:
        if key is None:
            raise ValueError("warning key must not be None")
        if key in cls._seen_keys:
            return
        cls._seen_keys.add(key)
        LOGGER.warning(fmt, *args)


def _strip_lower(v: Optional[str]) -> str:
    return (v.strip().lower()) if isinstance(v, str) else ""


_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_FALSE_VALUES = {"0", "false", "no", "off", "n"}
_INVALID_WARN_KEY_FMT = "env.invalid:{canonical_name}:{source}"
_DEPRECATED_ALIAS_KEY_FMT = "env.alias-deprecated:{primary_name}:{alias_used}"


def _find_raw(
    primary_name: str,
    aliases: Optional[Tuple[str, ...]],
) -> Tuple[Optional[str], str]:
    """Return (raw_value, source_name); prefer primary over oldest listed alias."""
    val = os.environ.get(primary_name)
    if val is not None:
        return val, primary_name
    if aliases is None:
        return None, primary_name
    for alias in aliases:
        val = os.environ.get(alias)
        if val is not None:
            WarningOnce.emit(
                f"env-alias-deprecated:{alias}->{primary_name}",
                "Environment variable %r is deprecated; please switch to %r",
                alias,
                primary_name,
            )
            return val, alias
    return None, primary_name


def env_bool(
    primary_name: str,
    *,
    default: bool = False,
    aliases: Optional[Tuple[str, ...]] = None,
) -> bool:
    """Parse ``KUNLUN_DSV4_XXX`` style boolean flag.

    Canonical name always wins, otherwise the earliest defined legacy alias
    carries the day. Invalid values emit one process-wide warning and fall back
    to *default*.
    """
    raw, source = _find_raw(primary_name, aliases)
    if raw is None:
        return default
    token = _strip_lower(raw)
    if token in _TRUE_VALUES:
        return True
    if token in _FALSE_VALUES:
        return False
    WarningOnce.emit(
        f"env-invalid-bool:{primary_name}",
        "Ignoring non-boolean value for %r=%r (set by %r); defaulting to %s",
        primary_name,
        raw,
        source,
        default,
    )
    return default


def env_str(
    primary_name: str,
    *,
    default: str,
    aliases: Optional[Tuple[str, ...]] = None,
) -> str:
    """String-valued env variable supporting old-name compatibility.

    Unlike *env_bool*, this never warns about values; only alias-use triggers
    a deprecation note.
    """
    raw, _source = _find_raw(primary_name, aliases)
    return raw.strip() if raw is not None else default


def env_int(
    primary_name: str,
    *,
    default: int,
    aliases: Optional[Tuple[str, ...]] = None,
) -> int:
    """Integer variable with canonical name precedence plus deprecated alias support."""
    raw, source = _find_raw(primary_name, aliases)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:  # noqa: BLE001
        WarningOnce.emit(
            f"env-invalid-int:{primary_name}",
            "Ignoring non-integer value for %r=%r (set by %r); defaulting to %d",
            primary_name,
            raw,
            source,
            default,
        )
        return default


__CPU_STAGING_CACHE: "dict[tuple[int|None, ...], torch.Tensor]" = {}


def make_static_cpu_tensor(
    tag: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    fill_value: float = -1.0,
) -> torch.Tensor:
    """Create/reuse a pinned CPU staging tensor with a fixed shape.

    CudaGraph-captured regions require identical allocations during each replay;
    keeping a pinned-paged pool avoids accidental dynamic-allocation variance
    introduced by ad-hoc `.cpu()` copies inside Pythonic fallbacks.
    """
    key = (tag,) + tuple(int(d) for d in shape) + (str(dtype), fill_value)
    existing = __CPU_STAGING_CACHE.get(key)
    if (
        existing is not None
        and tuple(existing.shape) == tuple(shape)
        and existing.dtype == dtype
        and existing.device.type == "cpu"
    ):
        return existing
    buf = torch.full(tuple(shape), fill_value, dtype=dtype, pin_memory=True)
    __CPU_STAGING_CACHE[key] = buf
    return buf


def find_op(*candidates: object) -> Optional["Callable[..., object]"]:
    """Probe a chain of nested attributes starting from arbitrary containers.

    Example::

        op = find_op(torch.ops.xspeedgate_ops, "act_sqrt_softplus")
               or find_op(kunlum_ops, "save_partial_states")

    Returns the first callable found, otherwise ``None``. Non-existent
    intermediate attributes silently abort probing.
    """
    obj: object
    parts: Tuple[str]
    if len(candidates) >= 2 and all(isinstance(x, str) for x in candidates[1:]):
        obj = candidates[0]
        parts = typing.cast(Tuple[str], candidates[1:])
    elif len(candidates) == 1 and isinstance(candidates[0], str):
        root_parts = typing.cast(str, candidates[0]).split(".")
        head = root_parts.pop(0)
        obj = getattr(__builtins__, head, globals().get(head))
        if obj is None:
            obj = getattr(typing.Any, head, None)  # placeholder path unused normally
        parts = tuple(root_parts)
    else:
        raise TypeError(f"Unsupported probe signature {candidates!r}")
    cur = obj
    try:
        for p in parts[:-1]:
            cur = getattr(cur, p)
        leaf = getattr(cur, parts[-1])
        return typing.cast("Optional[Callable[...,object]]", leaf if callable(leaf) else None)
    except AttributeError:
        return None


# Wiring / op inventory diagnostics.
_WIRED_INVENTORY: "dict[str, str]" = {}


def record_wired(op_name: str, source: str) -> None:
    """Record that *op_name* was bound via *source*; logged immediately."""
    _WIRED_INVENTORY[op_name] = source
    LOGGER.info("[KunlunPlugin] wired: %-40s via %s", op_name, source)


def log_wired_inventory() -> None:
    """Dump a per-source summary of the wiring inventory."""
    if not _WIRED_INVENTORY:
        return
    by_src: "dict[str, list[str]]" = {}
    for op, src in _WIRED_INVENTORY.items():
        by_src.setdefault(src, []).append(op)
    summary = ", ".join(
        f"{src}({len(ops)})" for src, ops in sorted(by_src.items())
    )
    LOGGER.info("[KunlunPlugin] wired inventory: %s", summary)
    for src in sorted(by_src):
        for op in sorted(by_src[src]):
            LOGGER.info("[KunlunPlugin]   %-40s <- %s", op, src)


def log_op_inventory(tag: str = "early") -> None:
    """Log op counts per torch dispatcher namespace plus mapped native libs.

    Called twice: once at plugin registration ("early") and once after the
    model runner loads ("final"). About twenty xspeedgate ops register lazily
    with the quant modules, so the early snapshot under-reports; trust the
    final line.
    """
    try:
        import torch

        def names(ns):
            return {n.split("::", 1)[1]
                    for n in torch._C._dispatch_get_all_op_names()
                    if n.startswith(ns + "::")}

        xsg, kl = names("xspeedgate_ops"), names("_C")
        try:
            import xspeedgate_ops
            where = xspeedgate_ops.__file__
        except Exception:  # noqa: BLE001
            where = "(not importable)"
        LOGGER.info("[KunlunPlugin] op inventory (%s): xspeedgate_ops=%d _C=%d from %s",
                    tag, len(xsg), len(kl), where)
        watch = ("sparse_attn_fwd", "act_sqrt_softplus", "dequantize_fp8_blocks",
                 "moe_pre_small", "compressed_attention", "mqa_logits_paged",
                 "moe_hash_topk_fused", "topk_per_row")
        LOGGER.info("[KunlunPlugin] key ops (%s): %s", tag,
                    " ".join("%s=%s" % (w, "Y" if w in xsg else "n") for w in watch))
        with open("/proc/self/maps") as f:
            libs = sorted({ln.split()[-1] for ln in f
                           if "xspeedgate" in ln or "kunlun_ops" in ln})
        for lib in libs:
            LOGGER.info("[KunlunPlugin] mapped %s", lib)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("[KunlunPlugin] op inventory probe failed: %r", e)


def _op_inventory_applied(mod: object) -> bool:
    return getattr(mod, "_kunlun_op_inventory_logged", False)


def _op_inventory_apply(mod: object) -> None:
    mod._kunlun_op_inventory_logged = True
    log_op_inventory(tag="final")
    log_wired_inventory()
