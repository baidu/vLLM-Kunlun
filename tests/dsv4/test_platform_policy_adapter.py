import os
from unittest.mock import patch

# This module is importable even before full vLLM model loading, because it uses runtime-utils only.
def test_apply_sets_block_size_for_dsv4():
    # Import here so earlier tests can run without needing the platform module.
    from vllm_kunlun.models.deepseek_v4_policy import apply as policy_apply
    from vllm_kunlun.platforms.kunlun import KunlunPlatform

    class FakeHFConfig:
        model_type = "deepseek_v4"

    class FakeModelConfig:
        use_mla = True
        hf_config = FakeHFConfig()

    class CacheConfig:
        block_size = 16

    class Vcfg:
        cache_config = CacheConfig()
        model_config = FakeModelConfig()

    old_fn = KunlunPlatform.check_and_update_config
    try:
        label = policy_apply()
        assert len(label) == 1 and "dsv4_wrap" in label[0]

        # Simulate baseline decisions first.
        Vcfg.cache_config.block_size = 64
        KunlunPlatform.check_and_update_config(KunlunPlatform, Vcfg)
        assert Vcfg.cache_config.block_size == 256, (
            f"expected DSV4 override to 256 got {Vcfg.cache_config.block_size}"
        )

        # Explicit force should still win.
        with patch.dict(os.environ, {"KUNLUN_DSV4_FORCE_MLA_BLOCK_SIZE": "128"}, clear=False):
            from vllm_kunlun.config.deepseek_v4 import FeatureFlags
            _ff = FeatureFlags()
            Vcfg.cache_config.block_size = 64
            KunlunPlatform.check_and_update_config(KunlunPlatform, Vcfg)
            assert Vcfg.cache_config.block_size == 128, Vcfg.cache_config.block_size
    finally:
        KunlunPlatform.check_and_update_config = old_fn
