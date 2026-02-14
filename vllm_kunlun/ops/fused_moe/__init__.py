"""
Kunlun FusedMoE CustomOp registration
"""


def register_kunlun_fused_moe_ops():
    """Register Kunlun FusedMoE CustomOp"""
    from .layer import KunlunUnquantizedFusedMoEMethod  # noqa: F401

    print(
        "[Kunlun Plugin] FusedMoE CustomOp registered: "
        "UnquantizedFusedMoEMethod -> KunlunUnquantizedFusedMoEMethod"
    )


__all__ = ["register_kunlun_fused_moe_ops"]
