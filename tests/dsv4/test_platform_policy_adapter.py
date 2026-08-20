import os
from unittest.mock import patch


def test_apply_labels_and_policy_hook_sets_block_size():
    from vllm_kunlun.models.deepseek_v4_policy import (
        _policy_post_hook,
        apply as policy_apply,
    )
    from vllm_kunlun.platforms.kunlun import KunlunPlatform

    class FakeHFConfig:
        model_type = "deepseek_v4"
        index_topk = 512

    class FakeModelConfig:
        use_mla = True
        hf_config = FakeHFConfig()

    class CacheConfig:
        block_size = 16

    class Vcfg:
        cache_config = CacheConfig()
        model_config = FakeModelConfig()

    label = policy_apply()
    assert len(label) == 1 and "dsv4_wrap" in label[0]
    assert policy_apply() == []  # idempotent

    Vcfg.cache_config.block_size = 64
    _policy_post_hook(KunlunPlatform, Vcfg)
    assert Vcfg.cache_config.block_size == 256

    with patch.dict(
        os.environ,
        {"KUNLUN_DSV4_FORCE_MLA_BLOCK_SIZE": "128"},
        clear=False,
    ):
        Vcfg.cache_config.block_size = 64
        _policy_post_hook(KunlunPlatform, Vcfg)
        assert Vcfg.cache_config.block_size == 128
