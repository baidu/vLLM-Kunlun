"""Runtime adjustment of Kunlun-wide hyperparameters for DeepSeek-V4.

Upstream ``KunlunPlatform.check_and_update_config`` handles generic OOT setup
(flashing MLA block size defaults, cudagraph settings, worker class selection).
For DeepSeek-V4-Flash we observed that larger page granularity performs better
with packed FP8/BF16 KV caches and dense monolithic paths, but the right choice
depends heavily on deployment mode. We therefore expose both explicit overrides
and attribute-driven hints OUTSIDE the platform implementation.

No code below mutates ``vllm_kunlun/platforms/kunlun.py`` directly; wrapping is
performed once at process startup by :func:`apply()`.
"""
import functools
import logging
import os
from typing import List, Optional

from vllm_kunlun.adapters.runtime_utils import WarningOnce, env_int

LOGGER = logging.getLogger("vllm_kunlun.adapters")
_APPLIED_KEY = "_dsv4_platform_adapter_installed"
_POLICY_LOG_KEY = "dsv4.platform.block_size_adjusted_once"

VALID_BLOCK_SIZES = {64, 128, 256}


def _looks_like_dsv4(model_hf_config: object) -> bool:
    """Identify DSV4 configurations structurally without hardcoding class names.

    Positive signals include:
      * hf_config.model_type == "deepseek_v4"
      * presence of DSV4-only compression / hyper-connection fields such as
        compress_ratios, hc_sinkhorn_iters and o_lora_rank together
    A positive signal gives permission to tune defaults; environment variables
    can always opt-out or override through KUNLUN_DSV4_FORCE_MLA_BLOCK_SIZE.
    """
    if getattr(model_hf_config, "model_type", None) == "deepseek_v4":
        return True
    composite_signals = (
        getattr(model_hf_config, "compress_ratios", None),
        getattr(model_hf_config, "hc_sinkhorn_iters", None),
        getattr(model_hf_config, "o_lora_rank", None),
    )
    return sum(1 for s in composite_signals if s is not None) >= 3


def _forced_block_size() -> Optional[int]:
    """Read explicit block-size override requested via CLI/env."""
    val: int | None = env_int(
        "KUNLUN_DSV4_FORCE_MLA_BLOCK_SIZE",
        default=-1,
    )
    if val < 0:
        return None
    if val in VALID_BLOCK_SIZES:
        return val
    LOGGER.warning(
        "Ignoring invalid %s=%r (expected one of %r)",
        "KUNLUN_DSV4_FORCE_MLA_BLOCK_SIZE",
        str(val),
        sorted(VALID_BLOCK_SIZES),
    )
    return None


def _resolve_dsv4_block_size(use_sparse: bool, vllm_config: object) -> Optional[int]:
    """Return desired KV-block size for DSV4 layouts; None keeps baseline choice."""
    forced = _forced_block_size()
    if forced is not None:
        return forced
    if not use_sparse:
        return None
    hf_cfg = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    if hf_cfg is None:
        return None
    if _looks_like_dsv4(hf_cfg):
        # Native sparse MLA kernels have shown fewer shape issues with coarser pages;
        # operator can override this default through FORCE_MLA_BLOCK_SIZE.
        return 256
    return None


def _policy_post_hook(_platform_cls, vllm_config):
    """Executed after baseline check_and_update_config finishes unchanged.

    Only touches ``cache_config.block_size``, preserving every other platform-level
    decision (worker class, data-parallel eager handling, cudagraph flags).
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    if cache_config is None or model_config is None:
        return
    if not getattr(model_config, "use_mla", False):
        return
    use_sparse = hasattr(getattr(model_config, "hf_config", None), "index_topk")
    target_bs = _resolve_dsv4_block_size(use_sparse, vllm_config)
    if target_bs is None:
        return
    if getattr(cache_config, "block_size", None) != target_bs:
        WarningOnce.emit(
            _POLICY_LOG_KEY,
            "DSV4 adapter policy applied kv-cache block_size=%d",
            target_bs,
        )
        cache_config.block_size = target_bs


def apply() -> List[str]:
    """Wrap ``KunlunPlatform.check_and_update_config`` transparently.

    Idempotent per-process; returns empty list on subsequent calls.
    """
    from vllm_kunlun.platforms.kunlun import KunlunPlatform

    if getattr(KunlunPlatform, _APPLIED_KEY, False):
        return []

    original_cm = KunlunPlatform.__dict__["check_and_update_config"]
    assert isinstance(original_cm, classmethod), (
        f"expected classmethod, got {type(original_cm)}"
    )
    underlying_fn = original_cm.__func__

    @functools.wraps(underlying_fn)
    def wrapped_classmethod(target_cls, vllm_config):
        result = underlying_fn(target_cls, vllm_config)
        try:
            _policy_post_hook(target_cls, vllm_config)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "DSV4 platform-policy override failed (%s); using baseline values",
                exc,
            )
        return result

    KunlunPlatform.check_and_update_config = classmethod(wrapped_classmethod)
    setattr(KunlunPlatform, _APPLIED_KEY, True)
    return ["kunlun.KunlunPlatform.check_and_update_config.dsv4_wrap"]
