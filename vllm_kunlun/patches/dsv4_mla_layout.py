"""Flip ``use_fp8_ds_mla_layout`` to False on DSv4 attention classes.

Upstream asserts FP8 packed KV-cache pages; Kunlun ships BF16 plain-row pages,
so every DSv4 attention subclass must be flipped before backend dispatch
selects the bf16 path.
"""
import inspect
import logging

LOGGER = logging.getLogger("vllm_kunlun.patches.dsv4_mla_layout")

_APPLIED_SENTINEL = "_kunlun_dsv4_layout_patched"


def _layout_predicate(mod) -> bool:
    return bool(getattr(mod, _APPLIED_SENTINEL, False))


def _layout_applier(mod) -> None:
    """Flip ``use_fp8_ds_mla_layout`` on every DSv4 attention subclass in *mod*."""
    for _name, _cls in list(mod.__dict__.items()):
        if not inspect.isclass(_cls):
            continue
        if _cls.__module__ != mod.__name__:
            continue
        if not hasattr(_cls, "use_fp8_ds_mla_layout"):
            continue
        try:
            _cls.use_fp8_ds_mla_layout = False
        except Exception:  # noqa: BLE001
            pass
    setattr(mod, _APPLIED_SENTINEL, True)
    LOGGER.info(
        "Flipped use_fp8_ds_mla_layout=False on %s subclasses", mod.__name__
    )
