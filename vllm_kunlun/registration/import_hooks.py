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

# Set when a re-entrant dispatch was skipped by the guard above.  Those nested
# imports can make further hooks eligible, so the outer dispatch runs the table
# again instead of dropping them.
_REDISPATCH = False

# Upper bound on those extra passes.  Patches are idempotent, so one or two
# passes always suffice in practice; the cap only stops a hook whose is_applied
# never turns True from spinning.  Anything left over is picked up by the
# dispatch that follows the next import.
_MAX_DISPATCH_PASSES = 4

# Registered hooks, applied in registration order on every dispatch.
_HOOKS: list[HookRegistration] = []
_HOOK_TARGETS: set[str] = set()


def register_hook(target: str, applied: HookApplied, apply: HookApply) -> None:
    """Register one idempotent patch for an upstream module.

    A target may have only one registration.  Rejecting duplicates here keeps
    patch order deterministic and catches accidental double registration at
    startup rather than during a later import.
    """
    if target in _HOOK_TARGETS:
        raise ValueError(f"Duplicate import hook target: {target}")
    _HOOK_TARGETS.add(target)
    _HOOKS.append(HookRegistration(target, applied, apply))


def _is_executing(module: ModuleType) -> bool:
    """Return whether ``module`` is published but still running its body.

    Python inserts a module into ``sys.modules`` *before* executing it, so a
    dispatch triggered by one of that module's own top-level imports would see
    a half-built module.  Patching it there is worse than useless: the
    statements still to run rebind the very names the patch just installed, and
    a patch that records "done" would never be retried.  Waiting costs nothing
    -- the import currently loading the module ends in another dispatch, by
    which point the body has finished.
    """
    spec = getattr(module, "__spec__", None)
    return spec is not None and getattr(spec, "_initializing", False)


def _dispatch_once() -> None:
    """Run every registered hook whose target is loaded and not yet patched."""
    logger = logging.getLogger("vllm_kunlun")
    for hook in _HOOKS:
        module = sys.modules.get(hook.target)
        if module is None or _is_executing(module):
            continue
        try:
            if not hook.is_applied(module):
                hook.apply_patch(module)
        except Exception:
            logger.exception(
                "[KunlunPlugin] post-import hook failed for target=%s",
                hook.target,
            )


def dispatch_hooks() -> None:
    """Apply registered patches whose target modules have finished loading.

    Safe to call any number of times: hooks whose target has not been
    imported yet are simply left for a later dispatch, and hooks already in
    effect are skipped via their ``is_applied`` predicate.  A failing patch
    is logged and isolated from the remaining registrations; this preserves
    vLLM's normal import flow and lets later hooks run.
    """
    global _DISPATCHING, _REDISPATCH
    if _DISPATCHING:
        _REDISPATCH = True
        return
    _DISPATCHING = True
    try:
        for _ in range(_MAX_DISPATCH_PASSES):
            _REDISPATCH = False
            _dispatch_once()
            if not _REDISPATCH:
                break
    finally:
        _DISPATCHING = False
        _REDISPATCH = False


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
