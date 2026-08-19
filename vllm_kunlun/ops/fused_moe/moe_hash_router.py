"""DeepSeek-V4 MoE hash-router + sqrt(softplus) scoring accelerator.

V4 route computation differs from earlier DeepSeek checkpoints:

* Non-hash layers use ``sqrt(softplus(gating))`` implemented upstream as
  ``torch.sqrt(F.softplus(gating.float()))``. This adapter swaps in the XPU
  pointwise operator :obj:`torch.ops.xspeedgate_ops.act_sqrt_softplus`, falling
  back to the exact Torch formula when unavailable.
* Hash layers are routed according to input-token hashes followed by top-k
  selection over fixed expert indices. Where available,
  :obj:`torch.ops.xspeedgate_ops.moe_hash_topk_fused` fuses lookup + activation +
  renormalization in one launch; otherwise we emulate with ``hash_indices_table``
  indexing plus softmax-renorm logic.
"""
import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F

from vllm_kunlun.runtime_utils import WarningOnce
from vllm_kunlun.patches.registry import _register_lazy

LOGGER = logging.getLogger("vllm_kunlun.ops.fused_moe.moe_hash_router")
_HASH_TOPK_MODULE = "vllm._custom_ops"
_ROUTER_MODULE = (
    "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
)
_APPLIED_ATTR = "_dsv4_wired_by_moe_hash_router"
_FALSE = object()

_op_cache: dict[str, object] = {}
_warned_missing_keys: set[str] = set()
_flag_cache: dict[int, object] = {}


def _get_flags():
    """Parse feature gates once per process at first routing call."""
    if not _flag_cache:
        # Lazy local import avoids cycles during adapter package startup.
        from vllm_kunlun.config.deepseek_v4 import FeatureFlags
        _flag_cache[0] = FeatureFlags()
    return _flag_cache[0]


def _xsg(op_name):
    key = f"xspeedgate_ops.{op_name}"
    value = _op_cache.get(key, _FALSE)
    if value is not _FALSE:
        return None if value is None else value
    handle = getattr(getattr(torch.ops, "xspeedgate_ops", object()), op_name, None)
    handle = handle if callable(handle) else None
    _op_cache[key] = handle
    return handle


def _warn_missing(kind):
    if kind not in _warned_missing_keys:
        _warned_missing_keys.add(kind)
        WarningOnce.emit(
            f"moe-hash-missing:{kind}",
            "%s missing/disabled for V4 MoE routing; continuing with PyTorch reference",
            kind,
        )


# ---------------------------------------------------------------------------
# sqrt(softplus) scorer shared across routes
# ---------------------------------------------------------------------------
def sqrt_softplus_scores(gating_output):
    """Return scores equivalent to torch.sqrt(F.softplus(x.float()))."""
    flags = _get_flags()
    x = gating_output.float()
    if flags.activation_routing_accel:
        op_handle = _xsg("act_sqrt_softplus")
        if op_handle is not None:
            try:
                out = op_handle(x)
                if isinstance(out, torch.Tensor) and out.shape == x.shape:
                    return out.to(torch.float32)
            except Exception as exc:  # noqa: BLE001
                WarningOnce.emit(
                    "moe-sqrt-softplus-op-failed-once",
                    "act_sqrt_softplus failed (%s); falling back to torch.sqrt(softplus())",
                    str(exc),
                )
                pass
        elif hasattr(gating_output, "device") or gating_output is gating_output:
            _warn_missing("torch.ops.xspeedgate_ops.act_sqrt_softplus")
    return torch.sqrt(F.softplus(x))


# ---------------------------------------------------------------------------
# Main route/top-k replacement
# ---------------------------------------------------------------------------
def _make_kunlun_topk_fn():
    def kunlun_route(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del token_expert_indices
        scores = sqrt_softplus_scores(gating_output)
        scores_for_choice = scores
        if e_score_correction_bias is not None:
            scores_for_choice = scores + e_score_correction_bias.float()

        flags = _get_flags()
        topk_k = topk_weights.shape[-1]
        hash_fused_kernel = (
            _xsg("moe_hash_topk_fused")
            if hash_indices_table is not None
               and input_tokens is not None
               and flags.hash_topk_fused
            else None
        )

        if hash_fused_kernel is not None:
            try:
                ids_out, weights_out = hash_fused_kernel(
                    gating_output.float().contiguous(),      # [M,E] fp32 logits/gates
                    input_tokens.long(),                     # [M] int64 tokens
                    hash_indices_table.int(),                # [vocab,k] int32 indices
                    0,                                       # num_shared_experts appended later
                    1.0,                                     # scaling applied below
                )
                weights_final = weights_out.float() * routed_scaling_factor
                topk_indices.copy_(ids_out.int())
                topk_weights.copy_(weights_final)
                return topk_weights, topk_indices
            except Exception as exc:  # noqa: BLE001
                WarningOnce.emit(
                    "moe-hash-topk-fused-failed-once",
                    "moe_hash_topk_fused failed (%s); falling through to torch indexing",
                    str(exc),
                )
                pass

        plain_hash_mode = (
            flags.hash_topk_fused and hash_indices_table is not None
            and input_tokens is not None
        )
        if plain_hash_mode:
            chosen_ids = hash_indices_table[input_tokens.long()].long().clamp_min_(0)
            chosen_ids[:, :].clamp_max_(scores.size(-1) - 1)
            topk_indices.copy_(chosen_ids[:, :topk_k])
            gathered_scores = scores.gather(1, chosen_ids[:, :topk_k]).float()
        else:
            _, ids_top = torch.topk(scores_for_choice, k=topk_k, dim=-1, sorted=False)
            ids_top.clamp_min_(0)
            topk_indices.copy_(ids_top.long())
            gathered_scores = scores.gather(1, ids_top)

        if renormalize:
            denom = gathered_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            gathered_scores = gathered_scores / denom
        topk_weights.copy_(gathered_scores * routed_scaling_factor)
        return topk_weights, topk_indices

    setattr(kunlun_route, "_dsv4_wired", True)
    setattr(kunlun_route, "_kunlun_hash_fallback", True)
    setattr(kunlun_route, "_kunlun_sqrt_softplus_accel", True)
    return kunlun_route


# ---------------------------------------------------------------------------
# Lazy-binding wrapper installed into vllm._custom_ops dispatch shim
# ---------------------------------------------------------------------------
def _install_custom_ops_shim(custom_mod):
    attr_name = "topk_hash_softplus_sqrt"
    current = getattr(custom_mod, attr_name, None)
    if current is not None and getattr(current, "_dsv4_wired", False):
        return [f"{custom_mod.__name__}.{attr_name}"]

    fallback_core = _make_kunlun_topk_fn()

    def lazy_wrapper(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        input_tokens=None,
        hash_indices_table=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
                _topk_softplus_sqrt_torch as late_rt,
            )
        except Exception:  # noqa: BLE001
            late_rt = fallback_core
        return late_rt(
            topk_weights=topk_weights,
            topk_indices=topk_indices,
            token_expert_indices=token_expert_indices,
            gating_output=gating_output,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            input_tokens=input_tokens,
            hash_indices_table=hash_indices_table,
            routed_scaling_factor=routed_scaling_factor,
        )

    setattr(lazy_wrapper, "_dsv4_wired", True)
    setattr(lazy_wrapper, "_kunlun_hash_fallback", True)
    setattr(custom_mod, attr_name, lazy_wrapper)
    LOGGER.info("Installed lazy %s.%s", custom_mod.__name__, attr_name)
    return [f"{custom_mod.__name__}.{attr_name}"]


# ---------------------------------------------------------------------------
# Direct patch on fused_topk_bias_router module itself
# ---------------------------------------------------------------------------
def _install_direct_router(router_mod):
    existing_vts_name = "vllm_topk_softplus_sqrt"
    current_entrypoint = getattr(router_mod, existing_vts_name, None)
    if current_entrypoint is not None and getattr(current_entrypoint, "_dsv4_wired", False):
        return []
    replacement = _make_kunlun_topk_fn()
    try:
        setattr(replacement, "_upstream_original_impl", current_entrypoint)
    except Exception:  # noqa: BLE001
        pass
    setattr(router_mod, existing_vts_name, replacement)
    setattr(router_mod, "_topk_softplus_sqrt_torch", replacement)
    LOGGER.info("Patched direct router in %s", router_mod.__name__)
    return [f"{router_mod.__name__}.{_topk_symbol_pretty()}"]


def _topk_symbol_pretty() -> str:
    return "vllm_topk_softplus_sqrt/_topk_softplus_sqrt_torch"


def _predicate_custom(mod):
    return getattr(getattr(mod, "topk_hash_softplus_sqrt", None), "_dsv4_wired", False)


def _predicate_router(mod):
    return getattr(getattr(mod, "vllm_topk_softplus_sqrt", None), "_dsv4_wired", False)


# ---------------------------------------------------------------------------
def apply(master_enabled_check=True) -> List[str]:
    """Install DSV4 sqrt-softplus / hash-router acceleration where selected.

    The subsystem requires at least one of the two feature bits -- native
    act_sqrt_softplus scoring or moe_hash_topk_fused hashing -- to be enabled;
    disabling both leaves upstream community routing untouched.
    """
    if not master_enabled_check:
        return []

    from vllm_kunlun.config.deepseek_v4 import FeatureFlags

    flags = FeatureFlags()
    if not (flags.hash_topk_fused or flags.activation_routing_accel):
        WarningOnce.emit(
            "dsv4-moe-hash-router-disabled",
            "Both DSV4 MoE-routing switches are disabled; skipping hash/softplus accelerator patches",
        )
        return []

    _register_lazy(_HASH_TOPK_MODULE, _predicate_custom, _install_custom_ops_shim)
    _register_lazy(_ROUTER_MODULE, _predicate_router, _install_direct_router)
    LOGGER.debug("Registered DSV4 MOE hash/softplus lazy hooks")
    return []
