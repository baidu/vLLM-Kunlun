"""Feature flags and deprecated-alias table for DeepSeek-V4 Kunlun adapters.

Each flag follows a uniform ``KUNLUN_DSV4_*`` naming convention.  Old names
introduced during the early bring-up are kept readable for one release cycle but
emit a deprecation warning telling the user the canonical replacement.
"""
from typing import Dict, Optional, Tuple, Union

from vllm_kunlun.adapters.runtime_utils import env_bool, env_int, env_str, WarningOnce

# ---------------------------------------------------------------------------
# Legacy alias lookup: old name -> (canonical primary name)
_DEPRECATED_ALIASES: "dict[str, tuple[str]]" = {
    # hash router / sqrtsoftplus routing acceleration
    "KUNLUN_V4_HASH_TOPK_FUSED":            ("KUNLUN_DSV4_HASH_TOPK_FUSED",),
    "KUNLUN_V4_ACT_SQRT_SOFTPLUS":          (
        "KUNLUN_DSV4_ACTIVATION_ROUTING_ACCEL",
    ),
    # output projection inverse RoPE fast path
    "KUNLUN_V4_OPROJ_NATIVE":               ("KUNLUN_DSV4_OPROJ_NATIVE",),
    # FP8 / INT8 MoE native paths; aliases differ only in prefix.
    "KUNLUN_FP8_MOE_NATIVE":                (
        "KUNLUN_DSV4_FP8_MOE_GROUPED_BF16_NATIVE",
    ),
    "KUNLUN_INT8_MOE_NATIVE":               (
        "KUNLUN_DSV4_INT8_W8A8_ROUTE_METHOD",
    ),
}


def _aliases(primary_name: str) -> Optional[Tuple[str, ...]]:
    """Return every legacy alias that maps to *primary_name*."""
    out = []
    for k, v in _DEPRECATED_ALIASES.items():
        if v == (primary_name,) or list(v) == [primary_name]:
            out.append(k)
    return tuple(out) if out else None


class FeatureFlags:
    """Runtime-readable DSV4 feature switches with backwards-compatible legacy naming.

    Reading a property lazily resolves environment values exactly once per process,
    using :func:`env_bool`/:func:`env_str`. The class is intentionally stateless so it can be
    instantiated cheaply anywhere without import side effects beyond logging.
    """

    __slots__ = ()

    @staticmethod
    def enabled(master_default: bool = True) -> bool:
        return env_bool(
            "KUNLUN_DSV4_PLUGINS_ENABLED", default=master_default
        )

    @property
    def platform_policy(self) -> bool:
        return self._bool("KUNLUN_DSV4_PLATFORM_POLICY")

    @property
    def forced_mla_block_size(self) -> Optional[int]:
        raw = env_str("KUNLUN_DSV4_FORCE_MLA_BLOCK_SIZE", default="")
        try:
            val = int(raw.strip())
            if val not in {64, 128, 256}:
                WarningOnce.emit(
                    "bad-block-size",
                    "%s=%r ignored because valid choices are 64/128/256",
                    "KUNLUN_DSV4_FORCE_MLA_BLOCK_SIZE",
                    raw,
                )
                return None
            return val
        except Exception:  # noqa: BLE001
            return None

    @property
    def flashmla_sparse_backend(self) -> bool:
        return self._bool("KUNLUN_DSV4_FLASHMLA_SPARSE_BACKEND")

    @property
    def qkv_cache_insert_native(self) -> bool:
        return self._bool("KUNLUN_DSV4_QKV_CACHE_INSERT_NATIVE")

    @property
    def rmsnorm_shortcut(self) -> bool:
        return self._bool("KUNLUN_DSV4_RMSNORM_SHORTCUT")

    @property
    def indexer_decode_native(self) -> bool:
        return self._bool("KUNLUN_DSV4_INDEXER_DECODE_NATIVE")

    @property
    def mhc_tilelang_native(self) -> bool:
        return self._bool("KUNLUN_DSV4_MHC_TILELANG_NATIVE")

    @property
    def oproj_native(self) -> bool:
        return self._bool("KUNLUN_DSV4_OPROJ_NATIVE")

    @property
    def activation_routing_accel(self) -> bool:
        return self._bool("KUNLUN_DSV4_ACTIVATION_ROUTING_ACCEL")

    @property
    def hash_topk_fused(self) -> bool:
        return self._bool("KUNLUN_DSV4_HASH_TOPK_FUSED")

    @property
    def compressor_save_native(self) -> bool:
        return self._bool("KUNLUN_DSV4_COMPRESSOR_SAVE_NATIVE")

    @property
    def compressor_vectorized_fallback(self) -> bool:
        return self._bool("KUNLUN_DSV4_COMPRESSOR_VECTORIZED_FALLBACK")

    @property
    def fp8_moe_grouped_bf16_native(self) -> bool:
        return self._bool("KUNLUN_DSV4_FP8_MOE_GROUPED_BF16_NATIVE")

    @property
    def int8_w8a8_route_method(self) -> bool:
        return self._bool("KUNLUN_DSV4_INT8_W8A8_ROUTE_METHOD")

    @classmethod
    def summarize(cls) -> Dict[str, Union[bool, str]]:
        inst = cls()
        keys = [
            "enabled", "platform_policy", "forced_mla_block_size",
            "flashmla_sparse_backend", "qkv_cache_insert_native",
            "rmsnorm_shortcut", "indexer_decode_native", "oproj_native",
            "activation_routing_accel", "hash_topk_fused",
            "compressor_save_native", "compressor_vectorized_fallback",
            "fp8_moe_grouped_bf16_native", "int8_w8a8_route_method",
        ]
        res = {}
        for key in keys:
            if key == "enabled":
                continue
            res[key] = getattr(inst, key)
        master = env_bool("KUNLUN_DSV4_PLUGINS_ENABLED", default=True)
        res["enabled"] = master
        return res

    @staticmethod
    def _bool(name: str, *, default: bool = True) -> bool:
        canon = name.replace("_ENABLED", "") != name  # unused stub for uniformity
        assert isinstance(default, bool), type(default).__name__
        return env_bool(name, default=default, aliases=_aliases(name))
