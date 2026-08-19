"""DeepSeek-V4 mHC (multi-HyperConnection) alias wiring.

V4's hyper-connection layer is implemented via tilelang helpers upstream:
``mhc_pre_tilelang``, ``mhc_post_tilelang``, ``mhc_fused_post_pre_tilelang``,
and ``hc_head_fused_kernel_tilelang``.  None of these run on Kunlun without a
dispatcher that calls the native XPU variants defined in
:mod:`vllm_kunlun.ops.hyper_connection`.  This adapter simply substitutes the
Kunlun native Python wrappers at import time.

Additionally, the V4 ``DeepseekV4MoE.forward`` on some model builds dispatches to
a fused :meth:`~RoutedExperts.apply_weights` path that expects grouped-topk args;
Kunlun keeps the per-call routing behaviour but supplies arguments compatible with
the current MoE implementation, so we patch only when needed and leave other
variants untouched.
"""
import logging
from typing import List

from vllm_kunlun.adapters.runtime_utils import WarningOnce
from vllm_kunlun.patches.registry import _register_lazy

LOGGER = logging.getLogger("vllm_kunlun.ops.fused_moe.mhc_hyperconnection")
_TILELANG_SENTINEL = "_dsv4_mhc_tilelang_wired"
_MODEL_SENTINEL = "_dsv4_mhc_model_wired"

# Targets where aliases are published for community code paths.
_MODULE_TARGETS = (
    "vllm.models.deepseek_v4.nvidia.model",
)
_TILE_TARGET = "vllm.model_executor.kernels.mhc.tilelang"


def _apply_to_moe_if_needed(mod):
    """Patch DeepseekV4MoE.forward only if it still uses `apply_weights` direct call."""
    moe_cls = getattr(mod, "DeepseekV4MoE", None)
    if moe_cls is None:
        return []
    forward_fn = getattr(moe_cls, "forward", None)
    if getattr(forward_fn, "_dsv4_mhc_router_patched", False):
        return [f"{mod.__name__}.deepseek_v4_moe.forward.already_set"]
    co_names = getattr(getattr(forward_fn, "__code__", None), "co_names", ())
    if "apply_weights" not in co_names or hasattr(moe_cls, "_dsv4_no_apply_weights_path"):
        # Variant already dispatching through another mechanism; do nothing.
        return []

    def _kunlun_v4_moe_forward(self, hidden_states):
        router_input = hidden_states[0] if isinstance(hidden_states, (list, tuple)) else hidden_states
        router_logits = self.gate(router_input)
        return self.moe.apply_weights(
            x=router_input,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=True,
            num_expert_group=getattr(self, "n_group", self.n_group),
            topk_group=getattr(self, "topk_group", self.topk_group),
            custom_routing_function=getattr(self, "custom_routing_function", None),
            scoring_func=getattr(self, "scoring_func", "softmax"),
            e_score_correction_bias=getattr(self, "e_score_correction_bias", None),
            activation=getattr(self, "activation", None),
            apply_router_weight_on_input=getattr(self, "apply_router_weight_on_input", False),
            enable_eplb=False,
        )

    setattr(_kunlun_v4_moe_forward, "_dsv4_mhc_router_patched", True)
    try:
        old_doc = getattr(forward_fn, "__doc__", None)
        if old_doc:
            _kunlun_v4_moe_forward.__doc__ = old_doc
    except Exception:
        pass

    moe_cls._dsv4_original_moe_forward = forward_fn
    moe_cls.forward = _kunlun_v4_moe_forward
    LOGGER.info("Patched DeepseekV4MoE.forward -> apply_weights path")
    return [f"{mod.__name__}.deepseek_v4_moe.forward.grouped_router_override"]


def _tilelang_applier(tile_mod) -> List[str]:
    from ...ops.hyper_connection import (
        mhc_fused_post_pre_tilelang,
        mhc_post_tilelang,
        mhc_pre_tilelang,
    )

    if getattr(tile_mod, _TILELANG_SENTINEL, False):
        return []

    tile_mod.mhc_pre_tilelang = mhc_pre_tilelang
    tile_mod.mhc_post_tilelang = mhc_post_tilelang
    tile_mod.mhc_fused_post_pre_tilelang = mhc_fused_post_pre_tilelang
    setattr(tile_mod, _TILELANG_SENTINEL, True)
    LOGGER.info("Wired DSV4 TileLang-mHC aliases into %s", tile_mod.__name__)
    return [_TILE_TARGET]


def _tilelang_predicate(tile_mod) -> bool:
    """Static-spec contract predicate paired with `_tilelang_applier`."""
    return bool(getattr(tile_mod, _TILELANG_SENTINEL, False))


def _model_applied(mod) -> bool:
    fn = getattr(mod, "mhc_pre_tilelang", None)
    mhc_okay = fn is not None and getattr(fn, "__module__", "") == "vllm_kunlun.ops.hyper_connection"
    head_fn = getattr(mod, "hc_head_fused_kernel_tilelang", None)
    head_okay = head_fn is not None and getattr(head_fn, "__module__", "") == "vllm_kunlun.ops.hyper_connection"
    return bool(getattr(mod, _MODEL_SENTINEL, False)) and mhc_okay and head_okay


def _model_applier(mod) -> List[str]:
    from ...ops.hyper_connection import (
        hc_head_fused_kernel_kunlun,
        mhc_fused_post_pre_tilelang,
        mhc_post_tilelang,
        mhc_pre_tilelang,
    )

    labels = []

    if not getattr(mod, _MODEL_SENTINEL, False):
        mod.mhc_pre_tilelang = mhc_pre_tilelang
        mod.mhc_post_tilelang = mhc_post_tilelang
        mod.mhc_fused_post_pre_tilelang = mhc_fused_post_pre_tilelang
        mod.hc_head_fused_kernel_tilelang = hc_head_fused_kernel_kunlun
        setattr(mod, _MODEL_SENTINEL, True)
        LOGGER.info("Wired DSV4 model-level mHC helpers into %s", mod.__name__)
        labels.append(f"{mod.__name__}.mhc_aliases")

    # DeepseekV4MoE forward override remains independent of RMSNorm/QKV-insert wiring.
    labels.extend(_apply_to_moe_if_needed(mod))
    return labels


def apply(master_enabled_check: bool = True) -> List[str]:
    """Register lazy hooks for DSV4 hyper-connection aliases.

    The native wrappers already contain their own capability-probe/fallback logic,
    so this adapter only needs to substitute the symbols; no env kill-switch other
    than KUNLUN_DSV4_PLUGINS_ENABLED and the implementation-local one.
    """
    if not master_enabled_check:
        return []

    for target in _MODULE_TARGETS:
        _register_lazy(target, _model_applied, _model_applier)
    _register_lazy(_TILE_TARGET, _tilelang_predicate, _tilelang_applier)
    return []
