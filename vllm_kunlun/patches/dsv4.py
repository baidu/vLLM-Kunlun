"""DeepSeek-V4 Kunlun adapter-pack entry point.

Calling :func:`apply_all` registers every DSV4 adapter (see
``vllm_kunlun.patches.registry``) with the post-import dispatcher; adapters
are gated by the ``KUNLUN_DSV4_*`` feature flags.
"""
import logging
from typing import Callable, Optional

LOGGER = logging.getLogger("vllm_kunlun.patches.dsv4")

_DONE = False


def apply_all(register_post_import_hook: Optional[Callable[..., None]] = None) -> None:
    """Wire all DSV4 adapters into already- or soon-to-be-imported modules.

    Args:
        register_post_import_hook: hook registration callback; when ``None``
            the registration package's ``register_hook`` is used.
    """
    global _DONE
    if _DONE:
        return

    from vllm_kunlun.config.deepseek_v4 import FeatureFlags

    if not FeatureFlags.enabled(master_default=True):
        return
    if register_post_import_hook is None:
        from vllm_kunlun.registration.import_hooks import register_hook

        register_post_import_hook = register_hook

    try:
        from vllm_kunlun.patches.registry import populate_hooks

        populate_hooks(register_post_import_hook)
        _DONE = True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "DSV4 adapter install step failed (%s); continuing without "
            "V4-specific patches",
            exc,
        )
