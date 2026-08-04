"""DeepSeek-V4-Flash Kunlun adapter pack.

Importing this package does **not** perform monkey-patching; call
:func:`apply_all` explicitly from a controlled loader such as the root plugin init.
All adapters are gated by environment flags defined in ``gates.py``.
"""
from typing import List, Optional
import logging
import sys
import typing
from .gates import FeatureFlags

__all__ = ["FeatureFlags", "get_applied_labels", "clear_application_state", "apply_all"]

_APPLIED_LABELS: "list[str]" = []
_APPLICATION_CLEARED: bool = False  # test hook to allow repeatable imports/tests


def get_applied_labels() -> List[str]:
    """Return labels of adapters installed by :func:`apply_all`."""
    return list(_APPLIED_LABELS)


def clear_application_state() -> None:
    """Test-only helper allowing repeated application during unit tests. Do not call at runtime."""
    global _APPLICATION_CLEARED, _APPLIED_LABELS
    _APPLIED_LABELS.clear()
    _APPLICATION_CLEARED = True


def apply_all(
    register_post_import_hook: Optional["Callable[..., None]"] = None,
) -> List[str]:
    """Wire all DSV4 adapters into already-imported or soon-to-be-imported modules.

    Args:
        register_post_import_hook: hook registration callback supplied by root.
            If ``None`` we look up the dispatcher from inside vllm_kunlun.__init__;
            if still absent (e.g., unit tests), only eager adapters are applied.

    Returns:
        Labels newly applied on this invocation.
    """
    global _APPLIED_LABELS, _APPLICATION_CLEARED
    if not _APPLICATION_CLEARED and _APPLIED_LABELS:
        return []

    flags = FeatureFlags()
    if not flags.enabled(master_default=True):
        return []

    labels_installed: "List[str]" = []
    try:
        from .registry import populate_hooks as _populate_registry_hooks

        # If caller passed no callback, attempt to retrieve it automatically so that this function can be called safely in isolation too.
        if register_post_import_hook is None:
            register_post_import_hook = _root_register_callback()

        labels_installed.extend(_populate_registry_hooks(register_post_import_hook))
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger("vllm_kunlun.adapters.dsv4").warning(
            "DSV4 adapter install step failed (%s); continuing without V4-specific patches",
            exc,
        )
        return labels_installed

    _APPLICATION_CLEARED = False
    _APPLIED_LABELS = list(dict.fromkeys(_APPLIED_LABELS + labels_installed))
    return labels_installed


def _root_register_callback() -> Optional["Callable[..., None]"]:
    """Internal helper locating vllm_kunlun's post-import dispatcher when running normally."""
    import sys

    mod = sys.modules.get("vllm_kunlun")
    if mod is None:
        return None
    fn = getattr(mod, "_register_post_import_hook", None)
    if callable(fn):
        return typing.cast("Callable[...,None]", fn)
    return None


def _noop_register(*_args, **_kwargs):
    """Placeholder used when no real dispatcher callback is available (e.g., unit tests)."""
    logging.getLogger("vllm_kunlun.adapters.dsv4").debug(
        "Skipping lazy post-import hook because dispatcher not available",
    )
