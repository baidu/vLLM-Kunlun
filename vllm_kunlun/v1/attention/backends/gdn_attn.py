# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends import gdn_attn as _upstream_gdn_attn
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

logger = init_logger(__name__)


def _to_cpu(t: torch.Tensor | None) -> torch.Tensor | None:
    """Force a metadata mirror onto the host, no-op if it is already there.

    Several ``*_cpu`` locals in ``build()`` are produced by indexing the *device*
    block table, so they are not actually on the host despite the name. Their
    consumers pass them to kunlun ops as host pointers.
    """
    if t is None or t.device.type == "cpu":
        return t
    return t.to("cpu")


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None
    # [Kunlun] CPU mirror — consumed by causal_conv1d_fn(has_initial_state_cpu=...)
    has_initial_state_cpu: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )
    # [Kunlun] CPU mirror — consumed by causal_conv1d_fn(query_start_loc_cpu=...)
    non_spec_query_start_loc_cpu: torch.Tensor | None = None

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    spec_state_indices_tensor_cpu: torch.Tensor | None = (
        None  # shape: [batch, num_spec]
    )
    spec_conv_state_indices_tensor: torch.Tensor | None = None  # shape: [batch,]
    spec_conv_state_indices_tensor_cpu: torch.Tensor | None = None  # shape: [batch,]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    # [Kunlun] CPU mirror — consumed as cache_indices_cpu / conv_state_indices_cpu
    non_spec_state_indices_tensor_cpu: torch.Tensor | None = None

    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    # [Kunlun] bool mask used by qwen3_next.py to split spec / non-spec tokens
    # via mixed_qkv[spec_token_masks] / g[:, spec_token_masks] etc.
    spec_token_masks: torch.Tensor | None = None
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    num_accepted_tokens_cpu: torch.Tensor | None = None  # shape: [batch,]

    # Variable-length speculative decode uses a padded layout only for the
    # fixed-width Kunlun conv kernel, then restores the real token layout before
    # the recurrent kernel. Both are None for uniform MTP batches.
    spec_pad_gather_idx: torch.Tensor | None = None
    spec_unpad_idx: torch.Tensor | None = None

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


def build_spec_pad_indices(
    spec_query_lens_cpu: torch.Tensor,
    spec_width: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Map variable-length spec tokens to fixed-width rows and back."""
    num_spec_decodes = spec_query_lens_cpu.size(0)
    lens = spec_query_lens_cpu.to(torch.int64)
    if int(lens.sum().item()) == num_spec_decodes * spec_width:
        return None

    cu = torch.zeros(num_spec_decodes + 1, dtype=torch.int64)
    torch.cumsum(lens, dim=0, out=cu[1:])
    starts = cu[:-1].unsqueeze(1)
    pos = torch.arange(spec_width, dtype=torch.int64).unsqueeze(0)
    last_real = (lens - 1).clamp_(min=0).unsqueeze(1)

    # Replicate the final real token so the conv input stays finite. These tail
    # outputs are removed before the recurrent kernel and cannot be accepted.
    pad_gather_idx = (starts + torch.minimum(pos, last_real)).reshape(-1)
    real_mask = pos < lens.unsqueeze(1)
    unpad_idx = (
        torch.arange(num_spec_decodes, dtype=torch.int64).unsqueeze(1) * spec_width
        + pos
    )[real_mask]
    return pad_gather_idx.to(torch.int32), unpad_idx.to(torch.int32)


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        # [Kunlun] Track which capture sizes have had their spec-decode
        # fused_recurrent kernel eagerly warmed up (see
        # _maybe_warmup_spec_kernel).
        self._spec_kernel_warmed: set[int] = set()

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.spec_conv_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_conv_state_indices_tensor_cpu = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device="cpu",
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        spec_conv_state_indices_tensor: torch.Tensor | None = None
        spec_conv_state_indices_tensor_cpu: torch.Tensor | None = None
        spec_sequence_masks_cpu: torch.Tensor | None = None
        spec_pad_gather_idx: torch.Tensor | None = None
        spec_unpad_idx: torch.Tensor | None = None
        # [Kunlun] These CPU mirrors are only produced on some branches below but
        # are always read when the metadata is built, so default them here.
        spec_state_indices_tensor_cpu: torch.Tensor | None = None
        non_spec_state_indices_tensor_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = spec_sequence_masks_cpu.to(
                    query_start_loc.device, non_blocking=True
                )

        if spec_sequence_masks is None:
            (
                num_decodes,
                num_prefills,
                num_decode_tokens,
                num_prefill_tokens,
            ) = split_decodes_and_prefills(m, decode_threshold=1)
            num_spec_decode_tokens = 0
            spec_token_indx = None
            spec_token_masks = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            spec_conv_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            non_spec_state_indices_tensor_cpu = (
                block_table_tensor[:, 0] if block_table_tensor is not None else None
            )
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                # [Kunlun] all tokens are spec tokens in this branch
                spec_token_masks = torch.ones(
                    spec_token_size,
                    dtype=torch.bool,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                spec_state_indices_tensor_cpu = (
                    block_table_tensor[spec_sequence_masks_cpu, : self.num_spec + 1]
                    if block_table_tensor is not None
                    else None
                )
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                spec_state_indices_tensor_cpu = (
                    block_table_tensor[spec_sequence_masks_cpu, : self.num_spec + 1]
                    if block_table_tensor is not None
                    else None
                )
                non_spec_state_indices_tensor = block_table_tensor[
                    ~spec_sequence_masks_cpu, 0
                ]
                non_spec_state_indices_tensor_cpu = (
                    block_table_tensor[~spec_sequence_masks_cpu, 0]
                    if block_table_tensor is not None
                    else None
                )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

            pad_indices = build_spec_pad_indices(
                query_lens_cpu[spec_sequence_masks_cpu], self.num_spec + 1
            )
            if pad_indices is not None:
                logger.warning_once(
                    "[KunlunPlugin] GDN spec decode hit non-uniform query "
                    "lengths; using the padded-conv fallback"
                )
                spec_pad_gather_idx = pad_indices[0].to(
                    query_start_loc.device, non_blocking=True
                )
                spec_unpad_idx = pad_indices[1].to(
                    query_start_loc.device, non_blocking=True
                )

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        if num_prefills > 0:
            # Only prefill batches use FLA chunk ops.
            # Pre-compute on CPU and async-copy to GPU to avoid
            # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
            from vllm.model_executor.layers.fla.ops.index import (
                prepare_chunk_indices,
                prepare_chunk_offsets,
            )
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            gpu_device = query_start_loc.device
            chunk_indices = prepare_chunk_indices(
                non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE
            ).to(device=gpu_device, non_blocking=True)
            chunk_offsets = prepare_chunk_offsets(
                non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE
            ).to(device=gpu_device, non_blocking=True)

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            (
                nums_dict,
                batch_ptr,
                token_chunk_offset_ptr,
            ) = compute_causal_conv1d_metadata(
                non_spec_query_start_loc_cpu,
                device=query_start_loc.device,
            )
        else:
            has_initial_state = None

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (
            num_decodes > 0 and num_spec_decodes > 0
        ), f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and spec_pad_gather_idx is None
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            self.spec_conv_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor[:, 0], non_blocking=True
            )
            self.spec_conv_state_indices_tensor_cpu[:num_spec_decodes].copy_(
                spec_state_indices_tensor_cpu[:num_spec_decodes, 0]
                if spec_state_indices_tensor_cpu is not None
                else spec_state_indices_tensor[:, 0].to(device="cpu", dtype=torch.int32)
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)
            spec_conv_state_indices_tensor = self.spec_conv_state_indices_tensor[
                :batch_size
            ]
            spec_conv_state_indices_tensor[num_spec_decodes:].fill_(0)
            spec_conv_state_indices_tensor_cpu = (
                self.spec_conv_state_indices_tensor_cpu[:batch_size]
            )
            spec_conv_state_indices_tensor_cpu[num_spec_decodes:].fill_(0)
            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[:num_spec_decodes].fill_(True)
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)
        elif spec_state_indices_tensor is not None:
            spec_conv_state_indices_tensor = (
                spec_state_indices_tensor[:, 0].to(torch.int32).contiguous()
            )
            spec_conv_state_indices_tensor_cpu = (
                spec_state_indices_tensor_cpu[:, 0].contiguous()
                if spec_state_indices_tensor_cpu is not None
                else spec_conv_state_indices_tensor.to(device="cpu", dtype=torch.int32)
            )
            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens.to(dtype=torch.int32)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)
        elif (
            non_spec_state_indices_tensor is not None
            and non_spec_state_indices_tensor_cpu is None
        ):
            non_spec_state_indices_tensor_cpu = non_spec_state_indices_tensor.to(
                device="cpu", dtype=torch.int32
            )

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            has_initial_state_cpu=(
                has_initial_state.to("cpu", non_blocking=True)
                if has_initial_state is not None
                else None
            ),
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            non_spec_query_start_loc_cpu=non_spec_query_start_loc_cpu,
            spec_state_indices_tensor=spec_state_indices_tensor,
            # [Kunlun] These locals are named ``*_cpu`` but are produced by
            # indexing the *device* ``block_table_tensor`` (see the branches
            # above), so they can still live on the device. The consumers hand
            # them to kunlun ops as host pointers, so they must be forced to CPU
            # -- passing a device tensor there faults with
            # "an illegal memory access was encountered".
            spec_state_indices_tensor_cpu=_to_cpu(spec_state_indices_tensor_cpu),
            spec_conv_state_indices_tensor=spec_conv_state_indices_tensor,
            spec_conv_state_indices_tensor_cpu=_to_cpu(
                spec_conv_state_indices_tensor_cpu
            ),
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            non_spec_state_indices_tensor_cpu=_to_cpu(
                non_spec_state_indices_tensor_cpu
            ),
            spec_sequence_masks=spec_sequence_masks,
            spec_token_masks=spec_token_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            # [Kunlun] No consumer reads this: ``causal_conv1d_update`` accepts a
            # ``num_accepted_tokens_cpu`` argument but never forwards it to the
            # kernel. Dropping the blocking ``.cpu()`` removes one device sync
            # per spec-decode step per attention group.
            num_accepted_tokens_cpu=None,
            spec_pad_gather_idx=spec_pad_gather_idx,
            spec_unpad_idx=spec_unpad_idx,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def _maybe_warmup_spec_kernel(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> None:
        """[Kunlun] Eagerly warm up the spec-decode fused_recurrent kernel
        before it is recorded into a FULL CUDA graph.

        Why: with MTP, the pre-capture warmup pass (_warmup_and_capture ->
        _dummy_run(cudagraph_runtime_mode=NONE)) builds metadata via the normal
        ``build()`` path. Because the dummy run never populates
        ``num_decode_draft_tokens``, ``build()`` takes the *non-spec* branch and
        the uniform (query_len = 1 + num_spec) requests are treated as prefill
        (chunk path). The actual capture, however, goes through
        ``build_for_cudagraph_capture`` which synthesizes *spec* metadata from
        ``diff(query_start_loc)`` and runs the *spec* ``fused_recurrent``. That
        spec kernel is therefore launched for the very first time inside the
        capture region.

        The Kunlun native op does capture-illegal work on its first launch for a
        new shape (load-based kernel auto-selection, L3 scratch allocation,
        module setup), which fails during capture with
        "CUDA error: unrecognized error code" (dump shows l3_size=0). Running it
        once in eager beforehand moves that first-launch cost outside the graph.

        This is invoked from ``build_for_cudagraph_capture``, which runs *before*
        the ``torch.cuda.graph`` capture region begins; an
        ``is_current_stream_capturing()`` guard makes sure we never launch the
        kernel while a capture is in progress.

        How to apply: only relevant for FULL cudagraph + spec decode; a no-op
        otherwise. Warms once per capture batch size.
        """
        if not (self.use_full_cuda_graph and self.use_spec_decode):
            return
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            return

        num_reqs = int(common_attn_metadata.num_reqs)
        if num_reqs <= 0 or num_reqs in self._spec_kernel_warmed:
            return

        try:
            from vllm_kunlun.ops.fla.fused_recurrent import (
                fused_recurrent_gated_delta_rule,
            )

            hf_config = self.vllm_config.model_config.hf_config
            tc = getattr(hf_config, "text_config", hf_config)
            num_k_heads = tc.linear_num_key_heads
            num_v_heads = tc.linear_num_value_heads
            head_k_dim = tc.linear_key_head_dim
            head_v_dim = tc.linear_value_head_dim

            spec_width = self.num_spec + 1
            total = num_reqs * spec_width
            dev = self.device
            # Match the real spec call: fp16 io / g / beta, fp16 state, and a
            # non-contiguous (transposed) h0 like the page-padded ssm_state view.
            io_dtype = torch.float16
            q = torch.zeros(
                1, total, num_k_heads, head_k_dim, dtype=io_dtype, device=dev
            )
            k = torch.zeros_like(q)
            v = torch.zeros(
                1, total, num_v_heads, head_v_dim, dtype=io_dtype, device=dev
            )
            g = torch.zeros(1, total, num_v_heads, dtype=io_dtype, device=dev)
            beta = torch.zeros(1, total, num_v_heads, dtype=io_dtype, device=dev)
            h0 = torch.zeros(
                num_reqs,
                num_v_heads,
                head_v_dim,
                head_k_dim,
                dtype=io_dtype,
                device=dev,
            ).transpose(-1, -2)
            cu_seqlens = torch.arange(
                0, total + 1, spec_width, dtype=torch.int32, device=dev
            )
            ssm_state_indices = (
                torch.arange(total, dtype=torch.int32, device=dev)
                .remainder(num_reqs)
                .reshape(num_reqs, spec_width)
            )
            num_accepted_tokens = torch.full(
                (num_reqs,), spec_width, dtype=torch.int32, device=dev
            )

            fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=h0,
                inplace_final_state=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=ssm_state_indices,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
            torch.cuda.synchronize()
            self._spec_kernel_warmed.add(num_reqs)
            logger.info(
                "[KunlunPlugin] warmed spec GDN fused_recurrent kernel for "
                "cudagraph capture (num_reqs=%d, tokens=%d)",
                num_reqs,
                total,
            )
        except Exception as e:  # warmup must never break startup
            logger.warning(
                "[KunlunPlugin] spec GDN kernel warmup failed (num_reqs=%s): %r",
                num_reqs,
                e,
            )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        # Warm the spec fused_recurrent kernel before graph capture.
        self._maybe_warmup_spec_kernel(common_attn_metadata)

        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )
        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)


# Monkey-patch upstream vllm so the GDN_ATTN backend registry resolves to
# Kunlun classes. Required because qwen3_next.py does
# `isinstance(attn_metadata, GDNAttentionMetadata)` against the Kunlun class,
# while the backend lookup via MambaAttentionBackendEnum.GDN_ATTN otherwise
# instantiates the upstream class.
_upstream_gdn_attn.GDNAttentionMetadata = GDNAttentionMetadata
_upstream_gdn_attn.GDNAttentionMetadataBuilder = GDNAttentionMetadataBuilder
_upstream_gdn_attn.GDNAttentionBackend = GDNAttentionBackend
