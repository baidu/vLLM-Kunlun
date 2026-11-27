# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass, replace
from math import prod

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


def get_gdn_sdnn_cache_spec(
    spec: MambaSpec,
    config: VllmConfig,
    num_spec: int,
    key_heads: int,
    value_heads: int,
    head_dims: tuple[int, int],
) -> MambaSpec:
    speculative = config.speculative_config
    if speculative is None or not config.use_v2_model_runner:
        return spec
    if speculative.method == "mtp":
        proposer = "mtp"
    elif speculative.method == "dflash":
        draft = getattr(speculative, "draft_model_config", None)
        architectures = getattr(draft, "architectures", None) or ()
        if "DFlash2T1ResidualDraftModel" in architectures:
            proposer = "dmtp"
        elif "DFlash2DraftModel" in architectures:
            proposer = "dflash2"
        else:
            return spec
    else:
        return spec
    if spec.mamba_cache_mode not in ("none", "align"):
        logger.info_once(
            "GDN SDNN single-slot fallback for Mamba cache mode %s.",
            spec.mamba_cache_mode,
        )
        return spec

    # A tree needs a state per branch. This verifier commits one linear prefix.
    backend = config.attention_config.backend
    if (
        getattr(speculative, "speculative_token_tree", None)
        or getattr(speculative, "tree_choices", None)
        or getattr(speculative, "use_tree_attention", False)
        or getattr(backend, "name", backend) == "TREE_ATTN"
        or not 1 <= num_spec <= 7
        or config.model_config.dtype != torch.float16
        or spec.dtypes[1] != torch.float16
        or head_dims != (128, 128)
        or not (0 < key_heads <= value_heads <= 48 and value_heads % key_heads == 0)
        or spec.page_size_padded is None
    ):
        return spec

    width = num_spec + 1
    # Put the factors in the same physical page as the recurrent state and
    # convolution window so allocation and zeroing cover the whole cache.
    shapes = spec.shapes + (
        (key_heads, width, 128),
        (value_heads, width, 128),
        (value_heads, width),
        (1,),
    )
    dtypes = spec.dtypes + (torch.float16, torch.float16, torch.float32, torch.int32)
    raw_bytes = sum(
        prod(shape) * dtype.itemsize for shape, dtype in zip(shapes, dtypes)
    )
    if raw_bytes > spec.page_size_padded:
        return spec

    from vllm_kunlun.ops.fla.fused_recurrent import gdn_sdnn_is_available

    if not gdn_sdnn_is_available(align=spec.mamba_cache_mode == "align"):
        return spec
    logger.info_once(
        "GDN SDNN single-slot verification enabled (proposer=%s, K=%d, mode=%s).",
        proposer,
        num_spec,
        spec.mamba_cache_mode,
    )
    return replace(spec, shapes=shapes, dtypes=dtypes, num_speculative_blocks=0)


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

    spec_conv_state_indices_tensor: torch.Tensor | None = None
    spec_pad_gather_idx: torch.Tensor | None = None
    spec_unpad_idx: torch.Tensor | None = None
    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    # [Kunlun] Prefill-only mirror, consumed by causal_conv1d_fn.
    # The current decode causal_conv1d_update ABI consumes device indices only.
    non_spec_state_indices_tensor_cpu: torch.Tensor | None = None

    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_sequence_masks_cpu: torch.Tensor | None = None  # shape: [batch,]
    # [Kunlun] bool mask used by qwen3_next.py to split spec / non-spec tokens
    # via mixed_qkv[spec_token_masks] / g[:, spec_token_masks] etc.
    spec_token_masks: torch.Tensor | None = None
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    non_spec_accepted: torch.Tensor | None = None
    non_spec_commit_slots: torch.Tensor | None = None
    # Batch-level handoff data shared by compatible GDN cache groups.
    non_spec_commit_live: int | None = None
    # Compact prefill only. A missing slot vector means all rows start at zero.
    prefill_state_slots: torch.Tensor | None = None
    prefill_fresh_mask: torch.Tensor | None = None
    prefill_zero_state: torch.Tensor | None = None
    prefill_history_rows: torch.Tensor | None = None

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

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self.compact_gdn = len(kv_cache_spec.shapes) == 6
        self._prefill_zeros: dict[torch.cuda.Stream | None, torch.Tensor] = {}
        self.state_columns = 1 if self.compact_gdn else self.num_spec + 1
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
            (self.decode_cudagraph_max_bs, self.state_columns),
            dtype=torch.int32,
            device=device,
        )
        self.spec_conv_state_indices_tensor = torch.empty(
            self.decode_cudagraph_max_bs, dtype=torch.int32, device=device
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
        accepted_rows = self.decode_cudagraph_max_bs
        if self.compact_gdn:
            # MRV2 hands off speculative state on ordinary eager steps as well.
            # The graph capture capacity can be zero when eager mode is forced.
            accepted_rows = max(
                accepted_rows, self.vllm_config.scheduler_config.max_num_seqs
            )
            self._non_spec_commit_slots = torch.empty(
                accepted_rows, dtype=torch.int32, device=device
            )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (accepted_rows,),
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

        spec_pad_gather_idx = spec_unpad_idx = None
        spec_sequence_masks_cpu: torch.Tensor | None = None
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

        batch_accepted = num_accepted_tokens
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
            padding = build_spec_pad_indices(
                query_lens_cpu[spec_sequence_masks_cpu], self.num_spec + 1
            )
            if padding is not None:
                spec_pad_gather_idx, spec_unpad_idx = (
                    x.to(query_start_loc.device, non_blocking=True) for x in padding
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

            has_initial_state = (context_lens_tensor > 0).to(torch.int32)
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

        has_initial_state_cpu = (
            has_initial_state.cpu() if has_initial_state is not None else None
        )

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
            and num_prefills == 0
            and num_decodes == 0
            and spec_pad_gather_idx is None
            and 0 < num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
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
            spec_num_query_tokens = num_spec_decode_tokens
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        ordinary_decode_graph = (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and batch_size <= self.decode_cudagraph_max_bs
        )
        if ordinary_decode_graph:
            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc[: num_decodes + 1], non_blocking=True
            )
            # fill_ with a device scalar extracts its value on the host. The
            # matching CPU query boundaries are already available here.
            non_spec_num_query_tokens = int(non_spec_query_start_loc_cpu[-1])
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        non_spec_accepted = non_spec_commit_live = None
        prefill_fresh_mask = prefill_zero_state = prefill_history_rows = None
        if self.compact_gdn and num_prefills:
            # This CPU mirror is also required by causal_conv1d_fn. Decide once
            # for all layers, excluding graph padding with no input tokens.
            history = has_initial_state_cpu.bool() & (
                non_spec_query_start_loc_cpu.diff() > 0
            )
            history_rows = history.nonzero(as_tuple=True)[0]
            if history_rows.numel():
                device = query_start_loc.device
                all_history = history_rows.numel() == history.numel()
                if not all_history:
                    prefill_fresh_mask = (
                        (~history).to(device, non_blocking=True).view(-1, 1, 1, 1)
                    )
                if batch_accepted is not None:
                    if not all_history:
                        prefill_history_rows = history_rows.to(
                            device, non_blocking=True
                        )
                    if all_history and spec_sequence_masks_cpu is None:
                        non_spec_accepted = batch_accepted
                    elif spec_sequence_masks_cpu is None:
                        non_spec_accepted = batch_accepted.index_select(
                            0, prefill_history_rows
                        )
                    else:
                        non_spec_rows = (~spec_sequence_masks_cpu).nonzero(
                            as_tuple=True
                        )[0]
                        batch_rows = non_spec_rows[history_rows]
                        non_spec_accepted = batch_accepted.index_select(
                            0, batch_rows.to(device, non_blocking=True)
                        )
            else:
                # The XPU chunk kernel requires an explicit h0 to return the
                # final state. All layers can read the same immutable zeros.
                device = query_start_loc.device
                stream = (
                    torch.cuda.current_stream(device) if device.type == "cuda" else None
                )
                zeros = self._prefill_zeros.get(stream)
                rows = history.numel()
                if zeros is None or zeros.size(0) < rows:
                    heads, v_dim, k_dim = self.kv_cache_spec.shapes[1]
                    zeros = torch.zeros(
                        (rows, heads, k_dim, v_dim),
                        device=device,
                        dtype=self.kv_cache_spec.dtypes[1],
                    )
                    self._prefill_zeros[stream] = zeros
                prefill_zero_state = zeros[:rows]
        elif self.compact_gdn and num_decodes and batch_accepted is not None:
            # Ordinary FULL decode replays with different slots and accept
            # counts at the same addresses.
            live = min(num_decodes, batch_accepted.numel())
            rows = batch_size if ordinary_decode_graph else live
            non_spec_commit_live = live
            non_spec_accepted = self.num_accepted_tokens[:rows]
            non_spec_accepted[:live].copy_(batch_accepted[:live], non_blocking=True)
            non_spec_accepted[live:].zero_()
        attn_metadata = GDNAttentionMetadata(
            non_spec_accepted=non_spec_accepted,
            non_spec_commit_live=non_spec_commit_live,
            prefill_fresh_mask=prefill_fresh_mask,
            prefill_zero_state=prefill_zero_state,
            prefill_history_rows=prefill_history_rows,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            has_initial_state_cpu=has_initial_state_cpu,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            non_spec_query_start_loc_cpu=non_spec_query_start_loc_cpu,
            spec_pad_gather_idx=spec_pad_gather_idx,
            spec_unpad_idx=spec_unpad_idx,
            spec_sequence_masks=spec_sequence_masks,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            spec_token_masks=spec_token_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return self.reuse_metadata(attn_metadata, m)

    def reuse_metadata(
        self,
        metadata: GDNAttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> GDNAttentionMetadata:
        """Select state IDs from this group's table and reuse the other fields."""
        m = metadata
        blk_table = mamba_get_block_table_tensor(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        spec_masks = m.spec_sequence_masks_cpu
        spec_indices = spec_conv_indices = non_spec_indices = None
        if spec_masks is None:
            non_spec_indices = blk_table[:, 0].contiguous()
        else:
            if m.num_prefills == 0 and m.num_decodes == 0:
                spec_indices = blk_table[: m.num_spec_decodes, : self.state_columns]
                spec_indices = spec_indices.contiguous()
            else:
                spec_indices = blk_table[spec_masks, : self.state_columns]
                non_spec_indices = blk_table[~spec_masks, 0]
            spec_conv_indices = spec_indices[:, 0].contiguous()
            if self.compact_gdn:
                spec_conv_indices = spec_conv_indices.masked_fill(
                    spec_conv_indices == NULL_BLOCK_ID, -1
                )

        # CUDA Graph replay uses this group's state buffers.
        batch_size = m.num_actual_tokens
        if (
            self.use_full_cuda_graph
            and m.num_prefills == 0
            and m.num_decodes == 0
            and m.spec_pad_gather_idx is None
            and 0 < m.num_spec_decodes <= self.decode_cudagraph_max_bs
            and m.num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_indices is not None and spec_conv_indices is not None
            num_spec_decodes = m.num_spec_decodes
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_indices, non_blocking=True
            )
            spec_indices = self.spec_state_indices_tensor[:batch_size]
            spec_indices[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_conv_state_indices_tensor[:num_spec_decodes].copy_(
                spec_conv_indices, non_blocking=True
            )
            spec_conv_indices = self.spec_conv_state_indices_tensor[:batch_size]
            spec_conv_indices[num_spec_decodes:].fill_(
                -1 if self.compact_gdn else NULL_BLOCK_ID
            )
        elif (
            self.use_full_cuda_graph
            and m.num_prefills == 0
            and m.num_spec_decodes == 0
            and batch_size <= self.decode_cudagraph_max_bs
        ):
            assert non_spec_indices is not None
            self.non_spec_state_indices_tensor[: m.num_decodes].copy_(
                non_spec_indices[: m.num_decodes], non_blocking=True
            )
            non_spec_indices = self.non_spec_state_indices_tensor[:batch_size]
            non_spec_indices[m.num_decodes :].fill_(NULL_BLOCK_ID)

        non_spec_commit_slots = prefill_state_slots = None
        if self.compact_gdn and m.num_prefills:
            assert non_spec_indices is not None
            non_spec_indices = non_spec_indices.contiguous()
            if m.prefill_zero_state is None:
                prefill_state_slots = non_spec_indices.long()
                if m.non_spec_accepted is not None:
                    non_spec_commit_slots = non_spec_indices
                    if m.prefill_history_rows is not None:
                        non_spec_commit_slots = non_spec_commit_slots.index_select(
                            0, m.prefill_history_rows
                        )
                    non_spec_commit_slots = non_spec_commit_slots.masked_fill(
                        non_spec_commit_slots == NULL_BLOCK_ID, -1
                    )
        elif self.compact_gdn and m.num_decodes and m.non_spec_accepted is not None:
            assert non_spec_indices is not None
            rows = m.non_spec_accepted.numel()
            live = (
                m.non_spec_commit_live
                if m.non_spec_commit_live is not None
                else min(m.num_decodes, rows)
            )
            non_spec_commit_slots = self._non_spec_commit_slots[:rows]
            live_slots = non_spec_commit_slots[:live]
            live_slots.copy_(non_spec_indices[:live], non_blocking=True)
            live_slots.masked_fill_(live_slots == NULL_BLOCK_ID, -1)
            non_spec_commit_slots[live:].fill_(-1)

        return replace(
            m,
            spec_state_indices_tensor=spec_indices,
            spec_conv_state_indices_tensor=spec_conv_indices,
            non_spec_commit_slots=non_spec_commit_slots,
            prefill_state_slots=prefill_state_slots,
            non_spec_state_indices_tensor=non_spec_indices,
            non_spec_state_indices_tensor_cpu=(
                non_spec_indices.cpu()
                if m.num_prefills and non_spec_indices is not None
                else None
            ),
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
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
        num_decode_draft_tokens_cpu = torch.diff(m.query_start_loc_cpu).sub_(1)

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)


# Monkey-patch upstream vllm so the GDN_ATTN backend registry resolves to
# Kunlun classes. Required because qwen3_next.py does
# `isinstance(attn_metadata, GDNAttentionMetadata)` against the Kunlun class,
# while the backend lookup via MambaAttentionBackendEnum.GDN_ATTN otherwise
# instantiates the upstream class.
_upstream_gdn_attn.GDNAttentionMetadata = GDNAttentionMetadata
_upstream_gdn_attn.GDNAttentionMetadataBuilder = GDNAttentionMetadataBuilder
_upstream_gdn_attn.GDNAttentionBackend = GDNAttentionBackend
