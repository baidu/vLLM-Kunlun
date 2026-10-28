# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""qwen4_exp QSA 需要、但随包 vLLM 里还没有的 KV cache spec。

随包 vLLM 的 ``vllm/v1/kv_cache_interface.py`` 没有 ``CircularBufferSpec``，
它来自上游那个带 qwen4_exp 的 PR。这里按上游语义原样回填，基类仍然从随包
vLLM import，避免改动 vllm。

随包 vLLM 升级到带该 spec 的版本后，本模块应删除，QSA 侧改回直接 import
上游位置。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec
from vllm.v1.request import Request


@dataclass(frozen=True, kw_only=True)
class CircularBufferSpec(AttentionSpec):
    """One block per request holding the raw keys of the token group that
    is still being compressed.

    ``block_size`` is the ring capacity. It must exceed the compression ratio
    by the speculative lookahead: a speculative step stores all of its rows,
    drafts included, before acceptance is known, while the next step still
    reads the open group's committed keys from the ring.
    """

    # 随包 vLLM 的 AttentionSpec 基类没有 head_size_v（只有 FullAttentionSpec /
    # MLAAttentionSpec 有），而上游 qwen4_exp 的 QSAKeyStateCache 会传
    # `head_size_v=0`（环形缓冲是 key-only），不接就 TypeError。
    head_size_v: int = 0

    @property
    def real_page_size_bytes(self) -> int:
        # 随包 vLLM 的 AttentionSpec.real_page_size_bytes 固定按 K+V 算
        # （2 × block_size × num_kv_heads × head_size × 元素宽），而且根本不看
        # head_size_v，于是环形缓冲的 page 会被算成两倍：unify 之后 raw 张量按
        # 两倍 page 分配，QSAStateBackend.get_kv_cache_shape 给出的是 key-only
        # 的 4 维形状，view 时就会报 "shape ... is invalid for input of size"。
        # 这里按 key-only 算回来。
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # The ring occupies one block per request for its whole lifetime.
        del vllm_config
        return self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        del vllm_config, max_len
        return 1

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, CircularBufferSpec) for spec in kv_cache_specs.values()
        )

    @property
    def prefix_cacheable(self) -> bool:
        return False


class CircularBufferManager(SingleTypeKVCacheManager):
    """Block manager for :class:`CircularBufferSpec`.

    The ring holds exactly one physical block per request for the request's
    whole lifetime: ``circular_qsa_slot_mapping`` only ever reads
    ``block_table[req, 0]`` and wraps the logical position modulo the ring
    capacity. The default token-driven allocation (``cdiv(num_tokens,
    block_size)``) would keep appending blocks as the sequence grows, which the
    block table (one column, see ``max_num_blocks_per_req``) cannot even hold.

    The ring is also not prefix-cacheable: its content is the open group's
    raw keys, not a function of the block-aligned prefix.
    """

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del num_tokens, new_computed_blocks, total_computed_tokens
        del num_tokens_main_model, apply_admission_cap
        return 1 if not self.req_to_blocks[request_id] else 0

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        del num_tokens, num_tokens_main_model
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        return new_blocks

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(
            kv_cache_spec, CircularBufferSpec
        ), "CircularBufferManager can only be used for circular buffer groups"
        del block_hashes, max_length, block_pool, drop_eagle_block
        del alignment_tokens, dcp_world_size, pcp_world_size
        return tuple([] for _ in kv_cache_group_ids)

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        del request, num_tokens, retention_interval

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        del running_request_id
        return 0


def register_custom_kv_cache_specs(vllm_config: VllmConfig | None = None) -> None:
    """Register the out-of-tree specs backfilled for qwen4_exp QSA.

    Called from ``KunlunPlatform.register_custom_kv_cache_specs``, i.e. from the
    tail of vLLM's own ``register_all_kvcache_specs``. Registering earlier than
    that would be a trap: ``KVCacheSpecRegistry._ensure_registered`` only runs
    the built-in registration while the registry is still empty, so a custom
    spec landing first would permanently shadow FullAttentionSpec and friends.
    """
    del vllm_config
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    KVCacheSpecRegistry.register(
        CircularBufferSpec,
        CircularBufferManager,
        uniform_type_base_spec=CircularBufferSpec,
    )
