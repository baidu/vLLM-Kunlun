"""
Copyright (c) 2026 Baidu, Inc. All Rights Reserved.

This file is a part of the vllm-kunlun project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from types import SimpleNamespace

import torch


class _FakeBackend:
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [8]

    @staticmethod
    def get_kv_cache_block_dim(
        kernel_block_size,
        num_kv_heads,
        head_size,
        cache_dtype_str=None,
    ):
        return 0


def _ensure_torch251_compat_shims():
    import vllm_kunlun  # noqa: F401
    import vllm_kunlun.schema  # noqa: F401
    from vllm_kunlun.compat import apply_torch251_compat_shims

    apply_torch251_compat_shims()


def test_prepare_kernel_block_sizes_preserves_kv_cache_group_ids():
    _ensure_torch251_compat_shims()

    from vllm.v1.kv_cache_interface import (
        EncoderOnlyAttentionSpec,
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    from vllm_kunlun.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes

    encoder_spec = EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["encoder"], kv_cache_spec=encoder_spec),
            KVCacheGroupSpec(layer_names=["decoder"], kv_cache_spec=attn_spec),
        ],
    )
    attn_groups = [
        [],
        [
            AttentionGroup(
                backend=_FakeBackend,
                layer_names=["decoder"],
                kv_cache_spec=attn_spec,
                kv_cache_group_id=1,
            )
        ],
    ]

    kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, attn_groups)

    assert kernel_block_sizes == [None, 8]


def test_kv_block_zeroer_init_meta_uses_aligned_kernel_block_sizes():
    _ensure_torch251_compat_shims()

    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_kunlun.v1.worker.utils import AttentionGroup, KVBlockZeroer

    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    attn_group = AttentionGroup(
        backend=_FakeBackend,
        layer_names=["decoder"],
        kv_cache_spec=attn_spec,
        kv_cache_group_id=1,
    )
    zeroer = KVBlockZeroer(device=torch.device("cpu"), pin_memory=False)
    static_forward_context = {
        "decoder": SimpleNamespace(kv_cache=torch.zeros((4, 4), dtype=torch.float32))
    }

    zeroer.init_meta(
        [attn_group],
        [None, 8],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context=static_forward_context,
    )

    assert zeroer._meta is not None
