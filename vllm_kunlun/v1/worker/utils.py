# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product as iprod
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.multimodal.registry import MultiModalRegistry
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import largest_power_of_2_divisor
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.core.encoder_cache_manager import compute_mm_encoder_budget
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention import Attention

logger = init_logger(__name__)


@triton.jit
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    block_ids_ptr,
    n_blocks,
    N_SEGS: tl.constexpr,
    PAGE_SIZE_EL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch."""
    pid = tl.program_id(0)
    chunks = PAGE_SIZE_EL // BLOCK_SIZE
    work_per_block = N_SEGS * chunks
    block_index = pid // work_per_block
    if block_index >= n_blocks:
        return
    remainder = pid % work_per_block
    seg_index = remainder // chunks
    chunk_index = remainder % chunks
    block_id = tl.load(block_ids_ptr + block_index)
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    offset = (
        block_id.to(tl.int64) * PAGE_SIZE_EL + chunk_index.to(tl.int64) * BLOCK_SIZE
    )
    cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    tl.store(ptr + offset + cols, tl.zeros([BLOCK_SIZE], dtype=tl.int32))


class KVBlockZeroer:
    """Manage efficient zeroing of KV cache blocks via a Triton kernel."""

    def __init__(self, device: torch.device, pin_memory: bool):
        self.device = device
        self.pin_memory = pin_memory
        self._meta: tuple[torch.Tensor, int, int, int] | None = None
        self._id_cap: int = 0
        self._ids_pinned: torch.Tensor | None = None
        self._ids_gpu: torch.Tensor | None = None

    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int | None],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
    ) -> None:
        seen_ptrs: set[int] = set()
        seg_addrs: list[int] = []
        page_size_el: int | None = None

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if type(spec) is not FullAttentionSpec:
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            if kernel_bs is None:
                continue
            ratio = spec.block_size // kernel_bs
            block_dim = group.backend.get_kv_cache_block_dim(
                kernel_bs,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=cache_dtype,
            )

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv = static_forward_context[layer_name].kv_cache
                if not isinstance(kv, torch.Tensor):
                    continue
                data_ptr = kv.data_ptr()
                if data_ptr in seen_ptrs:
                    continue
                seen_ptrs.add(data_ptr)

                element_size = kv.element_size()
                cur_bytes = kv.stride(block_dim) * element_size
                assert cur_bytes % 4 == 0
                kernel_block_el = cur_bytes // 4
                cur_page_el = kernel_block_el * ratio
                if page_size_el is None:
                    page_size_el = cur_page_el
                else:
                    assert (
                        page_size_el == cur_page_el
                    ), f"Non-uniform page sizes: {page_size_el} vs {cur_page_el}"

                block_stride_bytes = cur_bytes
                outer_dims = [
                    dim
                    for dim in range(block_dim)
                    if kv.stride(dim) * element_size > block_stride_bytes
                ]
                outer_strides = [kv.stride(dim) * element_size for dim in outer_dims]
                for outer in iprod(*(range(kv.shape[dim]) for dim in outer_dims)):
                    off_bytes = sum(
                        index * stride for index, stride in zip(outer, outer_strides)
                    )
                    seg_addrs.append(data_ptr + off_bytes)

        if not seg_addrs or page_size_el is None:
            self._meta = None
            return

        blk_size = min(largest_power_of_2_divisor(page_size_el), 1024)
        self._id_cap = 8192
        self._ids_pinned = torch.empty(
            self._id_cap, dtype=torch.int64, pin_memory=self.pin_memory
        )
        self._ids_gpu = torch.empty(self._id_cap, dtype=torch.int64, device=self.device)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            page_size_el,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        if not block_ids or self._meta is None:
            return
        seg_addrs, page_size_el, blk_size, n_segs = self._meta
        n_blocks = len(block_ids)
        if n_blocks > self._id_cap:
            self._id_cap = n_blocks * 2
            self._ids_pinned = torch.empty(
                self._id_cap, dtype=torch.int64, pin_memory=self.pin_memory
            )
            self._ids_gpu = torch.empty(
                self._id_cap, dtype=torch.int64, device=self.device
            )
        assert self._ids_pinned is not None and self._ids_gpu is not None
        self._ids_pinned[:n_blocks].numpy()[:] = block_ids
        idx = self._ids_gpu[:n_blocks]
        idx.copy_(self._ids_pinned[:n_blocks], non_blocking=True)
        grid = (n_blocks * n_segs * (page_size_el // blk_size),)
        _zero_kv_blocks_kernel[grid](
            seg_addrs,
            idx,
            n_blocks,
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
        )


class MultiModalBudget:
    """Helper class to calculate budget information for multi-modal models."""

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        mm_registry: MultiModalRegistry,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.mm_registry = mm_registry
        # vLLM 0.19 moved MM processor cache creation onto the registry and
        # requires a full VllmConfig. This utility only receives ModelConfig /
        # SchedulerConfig, so keep the legacy helper import-safe here.
        self.cache = cache = None

        self.max_model_len = model_config.max_model_len
        self.max_num_reqs = scheduler_config.max_num_seqs

        self.mm_limits = mm_registry.get_mm_limits_per_prompt(model_config, cache=cache)
        max_tokens_by_modality = (
            mm_registry.get_max_tokens_per_item_by_nonzero_modality(
                model_config, cache=cache
            )
        )

        encoder_compute_budget, encoder_cache_size = compute_mm_encoder_budget(
            scheduler_config,
            max_tokens_by_modality,
        )

        self.encoder_compute_budget = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        max_items_per_prompt_by_modality = {}
        max_items_per_batch_by_modality = {}

        for modality, max_tokens in max_tokens_by_modality.items():
            max_items_per_prompt, max_items_per_batch = self.get_max_items(
                modality, max_tokens
            )
            max_items_per_prompt_by_modality[modality] = max_items_per_prompt
            max_items_per_batch_by_modality[modality] = max_items_per_batch

        self.max_tokens_by_modality = max_tokens_by_modality
        self.max_items_per_prompt_by_modality = max_items_per_prompt_by_modality
        self.max_items_per_batch_by_modality = max_items_per_batch_by_modality

    def get_modality_with_max_tokens(self) -> str:
        modality, _ = max(self.max_tokens_by_modality.items(), key=lambda item: item[1])
        return modality

    def get_encoder_budget(self) -> int:
        return min(self.encoder_compute_budget, self.encoder_cache_size)

    def get_max_items(
        self,
        modality: str,
        max_tokens_per_item: int,
    ) -> tuple[int, int]:
        if max_tokens_per_item == 0:
            return 0, 0

        encoder_budget = self.get_encoder_budget()
        if encoder_budget == 0:
            return 0, 0

        max_encoder_items_per_batch = encoder_budget // max_tokens_per_item
        mm_limit = self.mm_limits[modality]
        max_items_per_prompt = max(
            1,
            min(mm_limit, self.max_model_len // max_tokens_per_item),
        )

        scheduler_config = self.scheduler_config
        max_num_reqs = self.max_num_reqs
        if not scheduler_config.enable_chunked_prefill:
            max_num_reqs = min(
                max_num_reqs,
                scheduler_config.max_num_batched_tokens // max_tokens_per_item,
            )

        max_decoder_items_per_batch = max_num_reqs * max_items_per_prompt
        max_items_per_batch = max(
            1,
            min(max_encoder_items_per_batch, max_decoder_items_per_batch),
        )
        return max_items_per_prompt, max_items_per_batch


@dataclass
class AttentionGroup:
    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=lambda: []
    )

    def create_metadata_builders(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ) -> None:
        kv_cache_spec_builder = (
            self.kv_cache_spec.copy_with_new_block_size(kernel_block_size)
            if kernel_block_size is not None
            else self.kv_cache_spec
        )
        self.metadata_builders = [
            self.backend.get_builder_cls()(
                kv_cache_spec_builder,
                self.layer_names,
                vllm_config,
                device,
            )
            for _ in range(num_metadata_builders)
        ]

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        assert len(self.metadata_builders) > ubatch_id
        return self.metadata_builders[ubatch_id]


def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
) -> int:
    """Select a block size supported by all attention backends."""

    def block_size_is_supported(
        backends: list[type[AttentionBackend]], block_size: int
    ) -> bool:
        for backend in backends:
            is_supported = False
            for supported_size in backend.get_supported_kernel_block_sizes():
                if isinstance(supported_size, int):
                    if block_size == supported_size:
                        is_supported = True
                elif isinstance(supported_size, MultipleOf):
                    if block_size % supported_size.base == 0:
                        is_supported = True
                else:
                    raise ValueError(f"Unknown supported size: {supported_size}")
            if not is_supported:
                return False
        return True

    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    all_int_supported_sizes = set(
        supported_size
        for backend in backends
        for supported_size in backend.get_supported_kernel_block_sizes()
        if isinstance(supported_size, int)
    )

    for supported_size in sorted(all_int_supported_sizes, reverse=True):
        if kv_manager_block_size % supported_size != 0:
            continue
        if block_size_is_supported(backends, supported_size):
            return supported_size
    raise ValueError(f"No common block size for {kv_manager_block_size}.")


def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list[AttentionGroup]]
) -> list[int | None]:
    """Generate kernel block sizes matching each KV cache group."""
    kernel_block_sizes: list[int | None] = []
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            kernel_block_sizes.append(None)
            continue
        if isinstance(kv_cache_spec, AttentionSpec):
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [group.backend for group in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)
        elif isinstance(kv_cache_spec, MambaSpec):
            kernel_block_sizes.append(kv_cache_spec.block_size)
        else:
            raise NotImplementedError(
                f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
            )
    return kernel_block_sizes


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """
    Perform sanity checks for the result of multimodal embedding generation.
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} instead. "
        "This is most likely due to incorrect implementation of the model's "
        "`embed_multimodal` method."
    )

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of input "
        f"items: {expected_num_items}, but got {len(mm_embeddings)=} instead. "
        "This is most likely due to incorrect implementation of the model's "
        "`embed_multimodal` method."
    )

    assert all(embed.ndim == 2 for embed in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, but "
        f"got tensors with shapes {[embed.shape for embed in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation of the "
        "model's `embed_multimodal` method."
    )


def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """Validate that startup free memory satisfies the requested utilization."""
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    if init_snapshot.free_memory < requested_memory:
        if hasattr(current_platform, "is_kunlun") and current_platform.is_kunlun():
            logger.warning(
                "Kunlun reported free memory %s GiB below requested %s GiB; "
                "clamping requested memory to the observed free memory and "
                "continuing initialization.",
                format_gib(init_snapshot.free_memory),
                format_gib(requested_memory),
            )
            return int(init_snapshot.free_memory)
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(init_snapshot.free_memory)}/"
            f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
            f"utilization or reduce GPU memory used by other processes."
        )

    return requested_memory


def scatter_mm_placeholders(
    embeds: torch.Tensor,
    is_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Scatter multimodal embeddings into a contiguous placeholder tensor.
    """
    if is_embed is None:
        return embeds

    placeholders = embeds.new_full(
        (is_embed.shape[0], embeds.shape[-1]),
        fill_value=torch.nan,
    )
    placeholders[is_embed] = embeds
    return placeholders


def gather_mm_placeholders(
    placeholders: torch.Tensor,
    is_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reconstruct embeddings from placeholder tokens."""
    if is_embed is None:
        return placeholders
    return placeholders[is_embed]


def add_kv_sharing_layers_to_kv_cache_groups(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    """
    Reuse allocated KV caches for layers participating in KV sharing.
    """
    layer_to_kv_cache_group: dict[str, KVCacheGroupSpec] = {}
    for kv_cache_group in kv_cache_groups:
        for layer_name in kv_cache_group.layer_names:
            layer_to_kv_cache_group[layer_name] = kv_cache_group

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        tgt_kv_cache_group = layer_to_kv_cache_group[target_layer_name]
        tgt_kv_cache_group.layer_names.append(layer_name)

        if runner_only_attn_layers is not None:
            runner_only_attn_layers.add(layer_name)


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, "Attention"],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: Optional[int] = 1,
) -> None:
    """
    Bind allocated KV cache tensors to the runner and forward context.
    """
    assert len(runner_kv_caches) == 0

    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            if (
                current_platform.is_kunlun()
                or current_platform.is_cuda_alike()
                or current_platform.is_xpu()
                or current_platform.is_cpu()
            ):
                pass
            else:
                raise NotImplementedError
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


def is_residual_scattered_for_sp(
    vllm_config: VllmConfig, num_input_tokens: int
) -> bool:
    """Check whether the residual tensor is sequence-parallel scattered."""
    if not vllm_config.compilation_config.pass_config.enable_sp:
        return False

    tp = vllm_config.parallel_config.tensor_parallel_size
    if tp == 1:
        return False

    assert num_input_tokens % tp == 0

    if (
        not vllm_config.compilation_config.splitting_ops
        or vllm_config.compilation_config.use_inductor_graph_partition
    ):
        return True

    compile_sizes = vllm_config.compilation_config.compile_sizes
    if compile_sizes is None:
        return False
    return num_input_tokens in compile_sizes
