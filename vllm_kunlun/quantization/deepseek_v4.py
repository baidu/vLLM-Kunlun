"""DeepSeek-V4 W8A8-INT8 MoE route-method bridge.

Upstream ``compressed-tensors`` exposes a general-purpose W8A8-INT8 Fused-MoE
kernel suitable for ``softmax``/``sigmoid`` routed models. DeepSeek-V4 uses
the sqrtsoftplus scorer (+ optional hash routes), whose logits/topk are produced
by the community router above the expert execution stage.

A thin ``# [dsv4-bridge]`` delegator inside
`vllm_kunlun/quantization/compressed_tensors/compressed_tensors_moe.py`
calls :func:`create_modular_v4_method`, while environment/kill-switch handling
lives here so the quantization source stays free of ad-hoc env reads.
"""
import logging
from typing import Any, List

from vllm_kunlun.config.deepseek_v4 import FeatureFlags
from vllm_kunlun.adapter_utils import WarningOnce

LOGGER = logging.getLogger("vllm_kunlun.quantization.deepseek_v4")
_METHOD_CLS_CACHE: "dict[str, Any]" = {}
_FLAGS_CACHE = None
_INSTALL_WARNED = False


def _flags() -> FeatureFlags:
    global _FLAGS_CACHE  # noqa: PLW0603
    if _FLAGS_CACHE is None:
        _FLAGS_CACHE = FeatureFlags()
    return _FLAGS_CACHE  # type: ignore[return-value]


def int8_w8a8_route_native_enabled(default_when_unset: bool = True) -> bool:
    """Return effective state of the INT8-V4 native pipeline switch.

    Canonical name::
      KUNLUN_DSV4_INT8_W8A8_ROUTE_METHOD=1|0
    Legacy alias read-through (with deprecation warning)::
      KUNLUN_INT8_MOE_NATIVE=1|0
    """
    master_ok = FeatureFlags.enabled()
    return (
        default_when_unset
        and master_ok
        and _flags().int8_w8a8_route_method
    )


def create_modular_v4_method(weight_quant: Any, input_quant: Any, layer: Any):
    """Return the modular DeepSeek-V4 INT8 expert runner instance.

    Args:
        weight_quant/input_quant: compressed-tensors quantization args chosen
            for this MoE layer.
        layer: RoutedExperts/FusedMoE module carrying ``layer.moe_config``.

    The heavy subclass remains physically next to its parent Kunlum MoE classes;
    this bridge factory avoids duplicating environment gating in that file while
    still keeping construction centralized under DSV4-specific logic.
    """
    cache_key = "KunlunCompressedTensorsW8A8Int8MoEMethodV4"
    ctor = _METHOD_CLS_CACHE.get(cache_key)
    if ctor is None:
        from vllm_kunlun.quantization.compressed_tensors.compressed_tensors_moe import (
            KunlunCompressedTensorsW8A8Int8MoEMethodV4,
        )
        _METHOD_CLS_CACHE[cache_key] = KunlunCompressedTensorsW8A8Int8MoEMethodV4
        ctor = KunlunCompressedTensorsW8A8Int8MoEMethodV4
    return ctor(weight_quant, input_quant, getattr(layer, "moe_config", None))


def apply(master_enabled_check: bool = True) -> List[str]:  # noqa: ARG001
    """Notify the adapter subsystem that the INT8-V4 route was considered.

    This adapter deliberately delegates through an import-time shim rather than
    runtime monkey-patches; repeated calls stay harmless via a single-process
    log guard.
    """
    global _INSTALL_WARNED  # noqa: PLW0603
    if not _INSTALL_WARNED:
        LOGGER.debug("DSV4 INT8-W8A8-MoE bridge active")
        _INSTALL_WARNED = True
    return []
