# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass

import os
import torch
from vllm.config import VllmConfig
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



# __Q38_NST_PERM__
# block_table_tensor[:, 0] 是 stride=(cols,) 的非稠密视图。MTP 下
# cols = num_spec + 1 > 1，而 Kunlun 的 kernel（reshape_and_cache_flash /
# causal_conv1d_*）把 int32 索引张量按稠密数组读，于是第 i>0 个请求的槽位会
# 被读成 block_table_tensor.flat[i] == block_table[0][i]，即请求 0 的投机
# scratch 块。必须先物化成稠密张量再交给 kernel。
def _q38_dense_indices(t):
    if t is None or t.is_contiguous():
        return t
    return t.contiguous()


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


# __Q38_NOSYNC__: prefill step 上把 CPU 镜像提前到 build 阶段算。
# Q38_MIRROR_EAGER=0 回到纯惰性行为。
import os as _q38_os_me

_Q38_MIRROR_EAGER = _q38_os_me.environ.get("Q38_MIRROR_EAGER", "1") == "1"
_Q38_SPEC_PAD_LOG = _q38_os_me.environ.get("Q38_SPEC_PAD_LOG", "0") == "1"
_Q38_SPEC_PAD_MAX = int(_q38_os_me.environ.get("Q38_SPEC_PAD_LOG_N", "20"))
_Q38_SPEC_PAD_N = [0]


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
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    # [Kunlun] CPU mirror — consumed as cache_indices_cpu / conv_state_indices_cpu
    # Q38 PERF: 改为惰性 property（见类尾），build() 不再做阻塞 .cpu()。

    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    # [Kunlun] bool mask used by qwen3_next.py to split spec / non-spec tokens
    # via mixed_qkv[spec_token_masks] / g[:, spec_token_masks] etc.
    spec_token_masks: torch.Tensor | None = None
    # [Kunlun] gather/unpad indices for non-uniform spec query lengths
    # (None when every spec request has exactly num_spec + 1 tokens)
    spec_pad_gather_idx: torch.Tensor | None = None
    spec_unpad_idx: torch.Tensor | None = None
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


    # Q38 PERF: 惰性 CPU 镜像。纯 decode FULL-graph 回放不执行 Python 前向，
    # 永远不会访问本属性 => 每步省一次阻塞 D2H 同步；eager/capture 路径首次
    # 访问时才同步，语义与原实现一致。
    @property
    def non_spec_state_indices_tensor_cpu(self) -> "torch.Tensor | None":
        cached = self.__dict__.get("_q38_nssit_cpu_cache")
        if cached is None and self.non_spec_state_indices_tensor is not None:
            cached = self.non_spec_state_indices_tensor.cpu()
            self.__dict__["_q38_nssit_cpu_cache"] = cached
        return cached


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
        # Q38 PERF: context_lens 仅 prefill 分支使用，推迟到那里再算，
        # 省掉纯 decode 步的 2 个 eager GPU 减法（XPU 派发开销大）。
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

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

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            spec_token_masks = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = _q38_dense_indices(
                block_table_tensor[:, 0])
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
                spec_state_indices_tensor = _q38_dense_indices(
                    block_table_tensor[spec_sequence_masks_cpu,
                                       : self.num_spec + 1])
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
                non_spec_mask_cpu = (~spec_sequence_masks_cpu) & (query_lens_cpu > 0)
                non_spec_state_indices_tensor = _q38_dense_indices(
                    block_table_tensor[non_spec_mask_cpu, 0])

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
                    int(non_spec_mask_cpu.sum().item()) + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[non_spec_mask_cpu.to(query_lens.device)],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    int(non_spec_mask_cpu.sum().item()) + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_mask_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

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
            context_lens_tensor = m.compute_num_computed_tokens()
            has_initial_state = (context_lens_tensor > 0).to(torch.int32)
            if spec_sequence_masks_cpu is not None:
                # Keep the validity mask aligned with the non-spec rows passed
                # to the prefill kernel; padded zero-length rows are omitted.
                non_spec_mask_cpu = (~spec_sequence_masks_cpu) & (
                    query_lens_cpu > 0
                )
                has_initial_state = has_initial_state[non_spec_mask_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None

        # __Q38_SLOTCOLL__: spec 的投机 scratch 列是否撞到 non-spec 的槽位


        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (
            num_decodes > 0 and num_spec_decodes > 0
        ), f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        # [Kunlun] Speculative tokens can be clipped for a step (for example the
        # last steps before max_model_len), which makes the per-request query
        # lengths non-uniform. Pad them to the fixed conv width and drop the
        # padded tail afterwards instead of assuming a uniform width.
        spec_pad_gather_idx = None
        spec_unpad_idx = None
        if spec_sequence_masks_cpu is not None:
            _pad_indices = build_spec_pad_indices(
                query_lens_cpu[spec_sequence_masks_cpu], self.num_spec + 1
            )
            if _pad_indices is not None:
                spec_pad_gather_idx = _pad_indices[0].to(
                    query_start_loc.device, non_blocking=True
                )
                spec_unpad_idx = _pad_indices[1].to(
                    query_start_loc.device, non_blocking=True
                )
                # __Q38_SPEC_PAD_LOG__: off by default; set Q38_SPEC_PAD_LOG=1
                # to confirm the clipped-spec path is being taken.
                if _Q38_SPEC_PAD_LOG and _Q38_SPEC_PAD_N[0] < _Q38_SPEC_PAD_MAX:
                    _Q38_SPEC_PAD_N[0] += 1
                    print(
                        "[Q38_SPEC_PAD] padded non-uniform spec lens "
                        f"{query_lens_cpu[spec_sequence_masks_cpu].tolist()} "
                        f"-> width {self.num_spec + 1} "
                        f"(hit {_Q38_SPEC_PAD_N[0]})",
                        flush=True,
                    )

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
            and spec_pad_gather_idx is None
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

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
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)
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

        # __Q38_VDUMP__: 每步 spec 元数据时间线（Q38_VDUMP=1 开）

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
                has_initial_state.cpu() if has_initial_state is not None else None
            ),
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            non_spec_query_start_loc_cpu=non_spec_query_start_loc_cpu,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_masks=spec_token_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            spec_pad_gather_idx=spec_pad_gather_idx,
            spec_unpad_idx=spec_unpad_idx,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        # __Q38_NOSYNC__: 惰性 CPU 镜像本是给纯 decode 全图省一次 D2H 的，
        # 但 mixed step 每层 forward 里首次访问时命令队列很深，一次只搬
        # 几个 int 也要等 ~800us；build 阶段队列浅，同样的拷贝只要几 us。
        # 各 group 的 state 索引不同，无法跨 group 共享，只能提前。
        if _Q38_MIRROR_EAGER and num_prefills > 0:
            attn_metadata.non_spec_state_indices_tensor_cpu  # noqa: B018
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
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



# __Q38_GDN_BUILD_CACHE__: 纯 spec-decode 步的跨 group 元数据缓存。
# 36 个 GDN group 每步构建的元数据除 spec_state_indices（各组 block table
# 不同）外完全相同；首个 builder 全量构建并缓存，其余复用，省 ~30 组 x
# 十余个小 GPU 算子的派发开销。Q38_GDN_BUILD_CACHE=0 可关闭。
import dataclasses as _q38_dc
import os as _q38_os_c

# __Q38_GDN_FASTIDX__: 跨 group 复用 gather index + gather out= + 免 fill。
_Q38_GO_OK = [True]
_Q38_GS = {"full": 0, "fast_out": 0, "fast_cp": 0, "fill": 0, "step": 0}
_Q38_GS_N = int(_q38_os_c.environ.get("Q38_GDN_STAT", "0") or 0)


def _q38_gs_step():
    if not _Q38_GS_N:
        return
    _Q38_GS["step"] += 1
    if _Q38_GS["step"] % _Q38_GS_N:
        return
    print("[Q38-GDN] step=%d per%d full=%d fast_out=%d fast_cp=%d fill=%d"
          % (_Q38_GS["step"], _Q38_GS_N, _Q38_GS["full"], _Q38_GS["fast_out"],
             _Q38_GS["fast_cp"], _Q38_GS["fill"]), flush=True)
    for k in ("full", "fast_out", "fast_cp", "fill"):
        _Q38_GS[k] = 0


def _q38_step_idx(pc, builder, m, n_spec):
    """align 模式下的 gather 列号，每步算一次，所有 group 共用。"""
    bsz = int(builder.kv_cache_spec.block_size)
    ns = int(builder.num_spec)
    key = (bsz, ns, n_spec)
    idx = pc["idx"].get(key)
    if idx is not None:
        return idx
    start = torch.clamp((m.seq_lens[:n_spec] - 1) // bsz, min=0)
    offs = torch.arange(ns + 1, device=m.seq_lens.device, dtype=torch.int32)
    idx = (start.unsqueeze(1) + offs).to(torch.int64)
    pc["idx"][key] = idx
    return idx



# __Q38_GDN_VERIFY__: 快路径与参考路径逐元素比对（Q38_GDN_VERIFY=N）。
_Q38_GV_N = int(_q38_os_c.environ.get("Q38_GDN_VERIFY", "0") or 0)
_Q38_GV = {"n": 0, "bad": 0, "tbad": 0}


def _q38_gdn_verify(builder, m, mask_cpu, n_spec, sst_view):
    mode = builder.vllm_config.cache_config.mamba_cache_mode
    ref_bt = mamba_get_block_table_tensor(
        m.block_table_tensor, m.seq_lens, builder.kv_cache_spec, mode
    )
    ref = ref_bt[mask_cpu, : builder.num_spec + 1]
    bad = int((sst_view[:n_spec] != ref).sum().item())
    tail = sst_view[n_spec:]
    tbad = int((tail != NULL_BLOCK_ID).sum().item()) if tail.numel() else 0
    _Q38_GV["n"] += 1
    _Q38_GV["bad"] += bad
    _Q38_GV["tbad"] += tbad
    if bad or tbad:
        print("[Q38-GDNV] BAD n_spec=%d bad=%d tail_bad=%d shape=%s"
              % (n_spec, bad, tbad, tuple(sst_view.shape)), flush=True)
    if _Q38_GV["n"] % _Q38_GV_N == 0:
        print("[Q38-GDNV] checks=%d mismatch=%d tail_bad=%d"
              % (_Q38_GV["n"], _Q38_GV["bad"], _Q38_GV["tbad"]), flush=True)


def _q38_gather_out(src, index, out):
    """gather 直写持久缓冲；设备不支持 out= 就永久退回 gather+copy_。"""
    if not _Q38_GO_OK[0]:
        return False
    try:
        torch.gather(src, 1, index, out=out)
        return True
    except Exception:                                      # noqa: BLE001
        _Q38_GO_OK[0] = False
        return False

if _q38_os_c.environ.get("Q38_GDN_BUILD_CACHE", "1") == "1":
    _q38_orig_build_c = GDNAttentionMetadataBuilder.build

    def _q38_cached_build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        cls = GDNAttentionMetadataBuilder
        c = getattr(cls, "_q38_bcache", None)
        if c is None:
            cls._q38_bcache = c = {"seen": set(), "meta": None}
        bid = id(self)
        if bid in c["seen"]:
            # 同一 builder 再次出现 => 新的一步，缓存失效
            c["seen"].clear()
            c["meta"] = None
            c["plan"] = None                              # __Q38_GDN_FASTIDX__
            _q38_gs_step()                                # __Q38_GDN_FASTIDX__
        c["seen"].add(bid)

        m = common_attn_metadata
        meta0 = c["meta"]
        if (
            meta0 is not None
            and self.use_full_cuda_graph
            and num_decode_draft_tokens_cpu is not None
            and num_accepted_tokens is not None
            and meta0.num_prefills == 0
            and meta0.num_decodes == 0
            and meta0.num_spec_decodes > 0
            and meta0.num_actual_tokens == m.num_actual_tokens
        ):
            # __Q38_GDN_FASTIDX__: mask/n_spec/all 每步算一次，37 个 group 共用
            _pc = c.get("plan")
            if _pc is None:
                _mk = num_decode_draft_tokens_cpu >= 0
                _n = int(_mk.sum().item())
                c["plan"] = _pc = {
                    "mask": _mk,
                    "n": _n,
                    "all": bool(_mk[:_n].all()),
                    "idx": {},
                }
            mask_cpu = _pc["mask"]
            n_spec = _pc["n"]
            if n_spec == meta0.num_spec_decodes:
                _q38_mode = self.vllm_config.cache_config.mamba_cache_mode
                if (_q38_mode in ("all", "none")
                        and _pc["all"]):
                    # __Q38_FASTBT_NONE__: none/all 模式下 block_table 即
                    # 状态索引表（mamba_get 原样返回），前 n_spec 行连续时
                    # 等价于纯 narrow 切片，零内核零拷贝
                    sst = m.block_table_tensor[:n_spec, : self.num_spec + 1]
                elif (_q38_mode == "align"
                        and _pc["all"]):
                    # __Q38_FASTBT__: 稳态全 spec 批，免布尔 gather，
                    # 直接按 seq_lens 计算 align 槽位（等价于
                    # mamba_get_block_table_tensor + [mask] 切片）
                    # __Q38_GDN_FASTIDX__: index 跨 group 复用，gather 直接
                    # out= 写进持久缓冲；每 group 从 ~7 次发射降到 1 次。
                    blk = m.block_table_tensor[:n_spec]
                    idx = _q38_step_idx(_pc, self, m, n_spec)
                    if _q38_gather_out(
                        blk, idx, self.spec_state_indices_tensor[:n_spec]
                    ):
                        sst = None
                        _Q38_GS["fast_out"] += 1
                    else:
                        sst = torch.gather(blk, 1, idx)
                        _Q38_GS["fast_cp"] += 1
                else:
                    _c = getattr(self, "_q38_dbg_n", 0)
                    if _c < 3:
                        self._q38_dbg_n = _c + 1
                    block_table_tensor = mamba_get_block_table_tensor(
                        m.block_table_tensor,
                        m.seq_lens,
                        self.kv_cache_spec,
                        self.vllm_config.cache_config.mamba_cache_mode,
                    )
                    sst = block_table_tensor[mask_cpu, : self.num_spec + 1]
                bs = m.num_actual_tokens
                if sst is not None:
                    self.spec_state_indices_tensor[:n_spec].copy_(
                        sst, non_blocking=True
                    )
                sst_view = self.spec_state_indices_tensor[:bs]
                # __Q38_GDN_FASTIDX__: 只写过 [:n_spec]，[n_spec:bs] 若上次
                # 就已经是 NULL 便无需重填（bs 变大、或上次的填充边界比这次
                # 靠后时才可能残留脏值）。稳态下这次发射也省掉。
                _tail = getattr(self, "_q38_tail", None)
                if _tail is None or bs > _tail[0] or _tail[1] > n_spec:
                    sst_view[n_spec:].fill_(NULL_BLOCK_ID)
                    self._q38_tail = (bs, n_spec)
                    _Q38_GS["fill"] += 1
                if _Q38_GV_N:                     # __Q38_GDN_VERIFY__
                    _q38_gdn_verify(self, m, mask_cpu, n_spec, sst_view)
                return _q38_dc.replace(
                    meta0, spec_state_indices_tensor=sst_view
                )

        self._q38_tail = None                             # __Q38_GDN_FASTIDX__
        _Q38_GS["full"] += 1                              # __Q38_GDN_FASTIDX__
        meta = _q38_orig_build_c(
            self,
            common_prefix_len,
            m,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
        )
        if (
            self.use_full_cuda_graph
            and meta.num_prefills == 0
            and meta.num_decodes == 0
            and meta.num_spec_decodes > 0
            # [Kunlun] padded (non-uniform) spec steps keep their own index
            # tensors; reusing them for the other groups would pair the padded
            # metadata with a staged state-index buffer.
            and meta.spec_pad_gather_idx is None
        ):
            c["meta"] = meta
        return meta

    GDNAttentionMetadataBuilder.build = _q38_cached_build


import os as _q38_os
