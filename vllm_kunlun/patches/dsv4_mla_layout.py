"""DSV4 MLA page-layout toggle.

Upstream sets ``use_fp8_ds_mla_layout: ClassVar[bool] = True`` on DeepSeekV4Attention,
asserting FP8 packed KV-cache pages. Kunlun ships BF16 plain-row pages, so every DSv4
attention subclass defined inside ``vllm.models.deepseek_v4.attention`` must be flipped
to False before backend dispatch selects our bf16 path. This adapter performs the
flip post-import-time behind the master plugin switch.
"""
import inspect
import logging
from typing import List

LOGGER = logging.getLogger("vllm_kunlun.patches.dsv4_mla_layout")

_APPLIED_SENTINEL = "_kunlun_dsv4_layout_patched"


def _layout_predicate(mod) -> bool:
    return bool(getattr(mod, _APPLIED_SENTINEL, False))


def _layout_applier(mod) -> None:
    """Flip ``use_fp8_ds_mla_layout`` False on every DSv4 Attention subclass in *mod*."""
    for _name, _cls in list(mod.__dict__.items()):
        if not inspect.isclass(_cls):
            continue
        if _cls.__module__ != mod.__name__:
            continue
        attr_present = (
            "use_fp8_ds_mla_layout" in vars(_cls)
            or hasattr(_cls, "use_fp8_ds_mla_layout")
        )
        if not attr_present:
            continue
        try:
            _cls.use_fp8_ds_mla_layout = False
        except Exception:  # noqa: BLE001
            pass
    setattr(mod, _APPLIED_SENTINEL, True)
    LOGGER.info(
        "Flipped use_fp8_ds_mla_layout=False on %s subclasses", mod.__name__
    )


def apply(master_enabled_check: bool = True) -> List[str]:
    """Legacy eager-install shim kept for backward-compat callers during transition.

    Real registration happens declaratively through
    :data:`vllm_kunlun.patches.registry._STATIC_PATCHES`; this stub returns empty labels
    and never triggers side effects until eventual removal scheduled under G2 cleanup.
    """
    del master_enabled_check
    return []
