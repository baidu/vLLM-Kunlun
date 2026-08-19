"""Single entry point for installing all DeepSeek-V4 Kunlum OOT adapters.

This thin wrapper lets :file:`vllm_kunlun/__init__.py` stay free of per-feature
gating logic. It simply checks the master switch ``KUNLUN_DSV4_PLUGINS_ENABLED``,
then hands off to ``patches.registry.populate_hooks()``.
"""
from typing import Callable, List

from vllm_kunlun.config.deepseek_v4 import FeatureFlags
from vllm_kunlun.patches.registry import populate_hooks
from vllm_kunlun.adapters.runtime_utils import WarningOnce

LOGGER_NAME = "vllm_kunlun.patches.deepseek_v4.installer"
_WARN_KEY_DISABLED = "dsv4-installer-master-switch-off"
_SUMMARY_LOGGED = False


def install_all(
    register_post_import_hook: Callable[..., None],
    *,
    log_summary: bool = True,
) -> List[str]:
    """Install every registered DSV4 adapter/hook under one master switch.

    Args:
        register_post_import_hook: callback provided by the caller
            (usually ``vllm_kunlun.__init__._register_post_import_hook``) used to
            schedule patches against upstream community modules.
        log_summary: when true, emit a single info listing applied labels.

    Returns:
        Labels of successfully registered/eagerly-applied adapters.
    """
    global _SUMMARY_LOGGED

    master_enabled = FeatureFlags.enabled()
    if not master_enabled:
        WarningOnce.emit(
            _WARN_KEY_DISABLED,
            "DeepSeek-V4 Kunlun adapters are disabled because %s=%r",
            "KUNLUN_DSV4_PLUGINS_ENABLED",
            __import__("os").environ.get("KUNLUN_DSV4_PLUGINS_ENABLED", ""),
        )
        return []

    labels_installed = populate_hooks(register_post_import_hook)

    if log_summary and not _SUMMARY_LOGGED:
        _SUMMARY_LOGGED = True
        _logger = __import__("logging").getLogger(LOGGER_NAME)
        # Keep message level low; registry.apply functions already announce failures.
        _logger.debug("Registered DeepSeek-V4 adapters: %d label(s)", len(labels_installed))

    return labels_installed