"""Import dispatcher that adapts upstream vLLM modules for Kunlun XPU.

This module owns the mechanism only; the actual compatibility content lives
in two sibling modules:

* ``module_redirects``: upstream modules replaced wholesale by Kunlun
  implementations while imports are being resolved.
* ``compat_patches``: targeted patches applied after an upstream module has
  finished importing.

Each post-import patch is registered as a ``(target, is_applied, apply_patch)``
entry.  The ``is_applied`` predicate makes dispatch idempotent, while
``apply_patch`` performs the minimal compatibility change when it is needed.

``install_import_hook()`` wraps ``builtins.__import__`` so both mechanisms run
automatically as vLLM imports its modules.  Patches are fail-soft: a broken
hook is logged and skipped rather than aborting startup, because a partially
adapted vLLM is still more useful than one that cannot start at all.
"""

import builtins
import logging
import sys
import weakref
from collections.abc import Callable
from types import ModuleType
from typing import NamedTuple

from .compat_patches import DEFAULT_HOOKS
from .module_redirects import preload_import_mappings

HookApplied = Callable[[ModuleType], bool]
HookApply = Callable[[ModuleType], None]


class HookRegistration(NamedTuple):
    """Describe one post-import patch without changing tuple semantics."""

    target: str
    is_applied: HookApplied
    apply_patch: HookApply


# The genuine __import__ must be captured before install_import_hook()
# replaces it; _custom_import delegates the real importing to this.
_OLD_IMPORT = builtins.__import__

# Re-entrancy guard: applying a patch can itself trigger imports, which would
# recursively enter dispatch_hooks() through the custom __import__.
_DISPATCHING = False

# Registered hooks, applied in registration order on every dispatch.
_HOOKS: list[HookRegistration] = []

# Hooks verified applied, memoized by module object (weakref, index-aligned
# with _HOOKS). torch emits lazy imports during steady-state decode, and
# re-running every predicate per import statement is pure overhead. A module
# re-imported after a reload is a fresh object, so it is re-checked
# automatically; a module still executing its body is never memoized.
_APPLIED_REFS: list = []


def _chained_predicates(first: HookApplied, second: HookApplied) -> HookApplied:
    def applied(module: ModuleType) -> bool:
        return first(module) and second(module)

    return applied


def _chained_appliers(first: HookApply, second: HookApply) -> HookApply:
    def apply_patch(module: ModuleType) -> None:
        first(module)
        second(module)

    return apply_patch


def register_hook(target: str, applied: HookApplied, apply: HookApply) -> None:
    """Register one idempotent patch for an upstream module.

    A later registration for an already-hooked target chains onto the existing
    entry instead of being rejected: platform-level patches and per-model
    adapters legitimately stack on the same upstream module, and rejecting one
    would silently disable it.
    """
    for i, existing in enumerate(_HOOKS):
        if existing.target != target:
            continue
        _HOOKS[i] = HookRegistration(
            target,
            _chained_predicates(existing.is_applied, applied),
            _chained_appliers(existing.apply_patch, apply),
        )
        # The old entry may already be memoized as applied for a loaded
        # module; drop the memo so the chained predicates are re-evaluated.
        if len(_APPLIED_REFS) > i:
            _APPLIED_REFS[i] = None
        return
    _HOOKS.append(HookRegistration(target, applied, apply))


def dispatch_hooks() -> None:
    """Apply registered patches whose target modules have finished loading.

    Safe to call any number of times: hooks whose target has not been
    imported yet are simply left for a later dispatch, and hooks already in
    effect are skipped via their ``is_applied`` predicate.  A failing patch
    is logged and isolated from the remaining registrations; this preserves
    vLLM's normal import flow and lets later hooks run.
    """
    global _DISPATCHING
    if _DISPATCHING:
        return
    _DISPATCHING = True
    try:
        logger = logging.getLogger("vllm_kunlun")
        for i, hook in enumerate(_HOOKS):
            module = sys.modules.get(hook.target)
            if module is None:
                continue
            while len(_APPLIED_REFS) <= i:
                _APPLIED_REFS.append(None)
            ref = _APPLIED_REFS[i]
            if ref is not None and ref() is module:
                continue
            try:
                if hook.is_applied(module):
                    spec = getattr(module, "__spec__", None)
                    if spec is None or not getattr(spec, "_initializing", False):
                        _APPLIED_REFS[i] = weakref.ref(module)
                    continue
                hook.apply_patch(module)
            except Exception:
                logger.exception(
                    "[KunlunPlugin] post-import hook failed for target=%s",
                    hook.target,
                )
    finally:
        _DISPATCHING = False


# Register the built-in patches at module import time.  The dispatcher itself
# is installed later by ``vllm_kunlun.register`` after plugin bootstrap.
for _target, _is_applied, _apply_patch in DEFAULT_HOOKS:
    register_hook(_target, _is_applied, _apply_patch)


def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    """Intercept absolute imports, then run any newly eligible patches."""
    # Relative imports (level > 0) can never name an upstream vLLM module
    # directly, so only absolute imports are checked against the redirects.
    if level == 0:
        try:
            preload_import_mappings(module_name, fromlist)
        except Exception:
            # A failed preload must not mask the original import error or
            # change vLLM's normal fallback behavior; the upstream module is
            # used as-is in that case.
            pass
    result = _OLD_IMPORT(module_name, globals, locals, fromlist, level)
    dispatch_hooks()
    return result


def install_import_hook() -> None:
    """Install the Kunlun dispatcher as Python's process-wide import hook."""
    builtins.__import__ = _custom_import
