"""DeepSeek-V4 Kunlun adapter pack entry point.

Calling :func:`apply_all` registers every DSV4 adapter (see
``vllm_kunlun.patches.registry``) with the post-import dispatcher; adapters
are gated by the ``KUNLUN_DSV4_*`` feature flags.
"""
from typing import List, Optional
import typing
from vllm_kunlun.config.deepseek_v4 import FeatureFlags

__all__ = ["FeatureFlags", "get_applied_labels", "clear_application_state", "apply_all"]

_APPLIED_LABELS: "list[str]" = []
_APPLICATION_CLEARED: bool = False  # test hook to allow repeatable imports/tests


def get_applied_labels() -> List[str]:
    """Return labels of adapters installed by :func:`apply_all`."""
    return list(_APPLIED_LABELS)


def clear_application_state() -> None:
    """Test-only helper allowing repeated application during unit tests."""
    global _APPLICATION_CLEARED, _APPLIED_LABELS
    _APPLIED_LABELS.clear()
    _APPLICATION_CLEARED = True


def apply_all(
    register_post_import_hook: Optional["Callable[..., None]"] = None,
) -> List[str]:
    """Wire all DSV4 adapters into already-imported or soon-to-be-imported modules.

    Args:
        register_post_import_hook: hook registration callback; when ``None``
            the root package's ``_register_post_import_hook`` is looked up
            automatically.

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
        from vllm_kunlun.patches.registry import populate_hooks as _populate_registry_hooks

        if register_post_import_hook is None:
            register_post_import_hook = _root_register_callback()

        labels_installed.extend(_populate_registry_hooks(register_post_import_hook))
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger("vllm_kunlun.patches.dsv4").warning(
            "DSV4 adapter install step failed (%s); continuing without V4-specific patches",
            exc,
        )
        return labels_installed

    _APPLICATION_CLEARED = False
    _APPLIED_LABELS = list(dict.fromkeys(_APPLIED_LABELS + labels_installed))
    return labels_installed


def _root_register_callback() -> Optional["Callable[..., None]"]:
    """Return the root package's post-import hook registrar, or None."""
    import sys

    mod = sys.modules.get("vllm_kunlun")
    fn = getattr(mod, "_register_post_import_hook", None)
    return typing.cast("Callable[...,None]", fn) if callable(fn) else None
