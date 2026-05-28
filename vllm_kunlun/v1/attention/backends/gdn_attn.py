# # SPDX-License-Identifier: Apache-2.0
# # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# """Backend for GatedDeltaNet attention."""

# import os
# from dataclasses import dataclass

# import torch
# from vllm.config import VllmConfig
# from vllm.v1.attention.backend import (
#     AttentionBackend,
#     AttentionCGSupport,
#     AttentionMetadataBuilder,
#     CommonAttentionMetadata,
# )
# from vllm.v1.attention.backends import gdn_attn
# from vllm.v1.attention.backends.utils import (
#     PAD_SLOT_ID,
#     compute_causal_conv1d_metadata,
#     mamba_get_block_table_tensor,
#     split_decodes_and_prefills,
# )
# from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

# from vllm_kunlun.v1.worker.mamba_utils import get_kunlun_running_slot_ctx


# class GDNAttentionBackend(AttentionBackend):
#     @staticmethod
#     def get_name() -> str:
#         return "GDN_ATTN"

#     @staticmethod
#     def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
#         return GDNAttentionMetadataBuilder


# @dataclass
# class GDNAttentionMetadata:
#     num_prefills: int
#     num_prefill_tokens: int
#     num_decodes: int
#     num_decode_tokens: int
#     num_spec_decodes: int
#     num_spec_decode_tokens: int
#     num_actual_tokens: int

#     has_initial_state: torch.Tensor | None = None
#     has_initial_state_cpu: torch.Tensor | None = None

#     spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
#     non_spec_query_start_loc: torch.Tensor | None = (
#         None  # shape: [batch - num_spec_decodes + 1,]
#     )
#     non_spec_query_start_loc_cpu: torch.Tensor | None = None
#     spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
#     non_spec_state_indices_tensor: torch.Tensor | None = (
#         None  # shape: [batch - num_spec_decodes,]
#     )
#     non_spec_state_indices_tensor_cpu: torch.Tensor | None = (
#         None  # shape: [batch - num_spec_decodes,]
#     )

#     spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]

#     spec_token_masks: torch.Tensor | None = (
#         None  # shape: [num_prefill_tokens + num_decode_tokens,]
#     )

#     spec_token_indx: torch.Tensor | None = None
#     non_spec_token_indx: torch.Tensor | None = None

#     num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

#     # The following attributes are for triton implementation of causal_conv1d
#     nums_dict: dict | None = None
#     batch_ptr: torch.Tensor | None = None
#     token_chunk_offset_ptr: torch.Tensor | None = None


# class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
#     _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

#     reorder_batch_threshold: int = 1

#     def __init__(
#         self,
#         kv_cache_spec: AttentionSpec,
#         layer_names: list[str],
#         vllm_config: VllmConfig,
#         device: torch.device,
#     ):
#         assert isinstance(kv_cache_spec, MambaSpec)
#         self.vllm_config = vllm_config
#         self.compilation_config = vllm_config.compilation_config
#         self.speculative_config = vllm_config.speculative_config
#         self.kv_cache_spec = kv_cache_spec
#         # [Kunlun APC AUDIT V2] remember our own layer_names so we can look up
#         # this group's expected phys block in _KUNLUN_PER_LAYER_EXPECTED.
#         self.layer_names: list[str] = list(layer_names)

#         if self.speculative_config:
#             assert self.speculative_config.num_speculative_tokens is not None
#             self.num_spec: int = self.speculative_config.num_speculative_tokens
#         else:
#             self.num_spec = 0
#         self.use_spec_decode = self.num_spec > 0
#         self._init_reorder_batch_threshold(1, self.use_spec_decode)

#         self.use_full_cuda_graph = (
#             self.compilation_config.cudagraph_mode.has_full_cudagraphs()
#         )

#         self.decode_cudagraph_max_bs = (
#             self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
#         )
#         if self.compilation_config.max_cudagraph_capture_size is not None:
#             self.decode_cudagraph_max_bs = min(
#                 self.decode_cudagraph_max_bs,
#                 self.compilation_config.max_cudagraph_capture_size,
#             )

#         self.spec_state_indices_tensor = torch.empty(
#             (self.decode_cudagraph_max_bs, self.num_spec + 1),
#             dtype=torch.int32,
#             device=device,
#         )
#         self.non_spec_state_indices_tensor = torch.empty(
#             (self.decode_cudagraph_max_bs,),
#             dtype=torch.int32,
#             device=device,
#         )
#         self.spec_sequence_masks = torch.empty(
#             (self.decode_cudagraph_max_bs,),
#             dtype=torch.bool,
#             device=device,
#         )
#         self.spec_token_masks = torch.empty(
#             (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
#             dtype=torch.bool,
#             device=device,
#         )
#         self.spec_token_indx = torch.empty(
#             (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
#             dtype=torch.int32,
#             device=device,
#         )
#         self.non_spec_token_indx = torch.empty(
#             (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
#             dtype=torch.int32,
#             device=device,
#         )
#         self.spec_query_start_loc = torch.empty(
#             (self.decode_cudagraph_max_bs + 1,),
#             dtype=torch.int32,
#             device=device,
#         )
#         self.non_spec_query_start_loc = torch.empty(
#             (self.decode_cudagraph_max_bs + 1,),
#             dtype=torch.int32,
#             device=device,
#         )
#         self.num_accepted_tokens = torch.empty(
#             (self.decode_cudagraph_max_bs,),
#             dtype=torch.int32,
#             device=device,
#         )

#     def build(  # type: ignore[override]
#         self,
#         common_prefix_len: int,
#         common_attn_metadata: CommonAttentionMetadata,
#         num_accepted_tokens: torch.Tensor | None = None,
#         num_decode_draft_tokens_cpu: torch.Tensor | None = None,
#         fast_build: bool = False,
#     ) -> GDNAttentionMetadata:
#         m = common_attn_metadata

#         query_start_loc = m.query_start_loc
#         query_start_loc_cpu = m.query_start_loc_cpu
#         context_lens_tensor = m.compute_num_computed_tokens()
#         nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
#         block_table_tensor = mamba_get_block_table_tensor(
#             m.block_table_tensor,
#             m.seq_lens,
#             self.kv_cache_spec,
#             self.vllm_config.cache_config.mamba_cache_mode,
#         )

#         # ------------------------------------------------------------------
#         # [Kunlun APC split-brain diagnostics & fix]
#         #
#         # Upstream's mamba_get_block_table_tensor("align") derives the
#         # running-slot column via `(seq_lens-1)//block_size`, while
#         # preprocess_mamba writes via `len(block_ids)-1-num_speculative_blocks`.
#         # These can diverge at block boundaries / APC hits, causing kernel
#         # to read a DIFFERENT physical slot than where preprocess_mamba
#         # wrote -> cross-request pollution.
#         #
#         # KUNLUN_APC_SLOT_DIFF=1 : print both for comparison (no behavior change)
#         # KUNLUN_APC_AUTH_SLOT=1 : actually override upstream with authoritative
#         # ------------------------------------------------------------------
#         _diff_log = os.environ.get("KUNLUN_APC_SLOT_DIFF", "0") == "1"
#         _auth_override = os.environ.get("KUNLUN_APC_AUTH_SLOT", "0") == "1"
#         if (
#             (_diff_log or _auth_override)
#             and self.vllm_config.cache_config.mamba_cache_mode == "align"
#         ):
#             _ctx = get_kunlun_running_slot_ctx()
#             _auth_non_spec = _ctx.get("non_spec_block_ids")
#             _auth_req_ids = _ctx.get("req_ids")
#             if (
#                 _auth_non_spec is not None
#                 and _auth_req_ids is not None
#                 and _auth_non_spec.shape[0] == block_table_tensor.shape[0]
#             ):
#                 if _diff_log:
#                     _upstream_col0 = block_table_tensor[:, 0].cpu().tolist()
#                     _auth_list = _auth_non_spec.cpu().tolist()
#                     _seq_lens = m.seq_lens.cpu().tolist() if m.seq_lens is not None else []
#                     _raw_bt = m.block_table_tensor.cpu().tolist() if m.block_table_tensor is not None else []
#                     for _i, _rid in enumerate(_auth_req_ids):
#                         _u = _upstream_col0[_i] if _i < len(_upstream_col0) else "?"
#                         _a = _auth_list[_i] if _i < len(_auth_list) else "?"
#                         _sl = _seq_lens[_i] if _i < len(_seq_lens) else "?"
#                         _row = _raw_bt[_i] if _i < len(_raw_bt) else []
#                         _bs = self.kv_cache_spec.block_size
#                         _start_idx = max((_sl - 1) // _bs, 0) if isinstance(_sl, int) else "?"
#                         _mark = " *** DIFF ***" if _u != _a else ""
#                         print(
#                             f"[SLOT_DIFF] req={_rid} seq_len={_sl} "
#                             f"block_size={_bs} "
#                             f"upstream_start_idx={_start_idx} "
#                             f"upstream_blk={_u} auth_blk={_a} "
#                             f"raw_bt={_row[:8]}{_mark}",
#                             flush=True,
#                         )
#                 if _auth_override:
#                     block_table_tensor = block_table_tensor.clone()
#                     block_table_tensor[:, 0] = _auth_non_spec.to(
#                         block_table_tensor.device
#                     )
#                     _auth_spec = _ctx.get("spec_block_ids")
#                     if (
#                         _auth_spec is not None
#                         and _auth_spec.shape[1] == block_table_tensor.shape[1]
#                     ):
#                         block_table_tensor[:, :] = _auth_spec.to(
#                             block_table_tensor.device
#                         )

#         spec_sequence_masks_cpu: torch.Tensor | None = None
#         if (
#             not self.use_spec_decode
#             or num_decode_draft_tokens_cpu is None
#             or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
#             .sum()
#             .item()
#             == 0
#         ):
#             spec_sequence_masks = None
#             num_spec_decodes = 0
#         else:
#             spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
#             num_spec_decodes = spec_sequence_masks_cpu.sum().item()
#             if num_spec_decodes == 0:
#                 spec_sequence_masks = None
#                 spec_sequence_masks_cpu = None
#             else:
#                 spec_sequence_masks = spec_sequence_masks_cpu.to(
#                     query_start_loc.device, non_blocking=True
#                 )

#         if spec_sequence_masks is None:
#             num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
#                 split_decodes_and_prefills(m, decode_threshold=1)
#             )
#             num_spec_decode_tokens = 0
#             spec_token_indx = None
#             spec_token_masks = None
#             non_spec_token_indx = None
#             spec_state_indices_tensor = None
#             non_spec_state_indices_tensor = block_table_tensor[:, 0]
#             spec_query_start_loc = None
#             non_spec_query_start_loc = query_start_loc
#             non_spec_query_start_loc_cpu = query_start_loc_cpu
#             num_accepted_tokens = None
#         else:
#             query_lens = query_start_loc[1:] - query_start_loc[:-1]
#             assert spec_sequence_masks_cpu is not None
#             query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

#             non_spec_query_lens = query_lens[~spec_sequence_masks]
#             num_decodes = (non_spec_query_lens == 1).sum().item()
#             num_prefills = non_spec_query_lens.size(0) - num_decodes
#             num_decode_tokens = num_decodes
#             num_prefill_tokens = non_spec_query_lens.sum().item() - num_decode_tokens
#             num_spec_decode_tokens = (
#                 query_lens.sum().item() - num_prefill_tokens - num_decode_tokens
#             )

#             if num_prefills == 0 and num_decodes == 0:
#                 spec_token_masks = torch.ones(
#                     (
#                         min(
#                             num_spec_decodes * (self.num_spec + 1),
#                             query_start_loc[-1].item(),
#                         )
#                     ),
#                     dtype=torch.bool,
#                     device=query_start_loc.device,
#                 )
#                 spec_token_size = min(
#                     num_spec_decodes * (self.num_spec + 1),
#                     query_start_loc[-1].item(),
#                 )
#                 spec_token_indx = torch.arange(
#                     spec_token_size,
#                     dtype=torch.int32,
#                     device=query_start_loc.device,
#                 )
#                 non_spec_token_indx = torch.empty(
#                     0, dtype=torch.int32, device=query_start_loc.device
#                 )
#                 spec_state_indices_tensor = block_table_tensor[:, : self.num_spec + 1]
#                 non_spec_state_indices_tensor = None
#                 spec_query_start_loc = query_start_loc
#                 non_spec_query_start_loc = None
#                 non_spec_query_start_loc_cpu = None
#             else:
#                 spec_token_masks = torch.repeat_interleave(
#                     spec_sequence_masks, query_lens
#                 )
#                 index = torch.argsort(spec_token_masks, stable=True)
#                 num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
#                 non_spec_token_indx = index[:num_non_spec_tokens]
#                 spec_token_indx = index[num_non_spec_tokens:]

#                 spec_state_indices_tensor = block_table_tensor[
#                     spec_sequence_masks, : self.num_spec + 1
#                 ]
#                 non_spec_state_indices_tensor = block_table_tensor[
#                     ~spec_sequence_masks, 0
#                 ]

#                 spec_query_start_loc = torch.zeros(
#                     num_spec_decodes + 1,
#                     dtype=torch.int32,
#                     device=query_start_loc.device,
#                 )
#                 torch.cumsum(
#                     query_lens[spec_sequence_masks], dim=0, out=spec_query_start_loc[1:]
#                 )
#                 non_spec_query_start_loc = torch.zeros(
#                     query_lens.size(0) - num_spec_decodes + 1,
#                     dtype=torch.int32,
#                     device=query_start_loc.device,
#                 )
#                 torch.cumsum(
#                     query_lens[~spec_sequence_masks],
#                     dim=0,
#                     out=non_spec_query_start_loc[1:],
#                 )
#                 non_spec_query_start_loc_cpu = torch.zeros(
#                     query_lens_cpu.size(0) - num_spec_decodes + 1,
#                     dtype=torch.int32,
#                 )
#                 torch.cumsum(
#                     query_lens_cpu[~spec_sequence_masks_cpu],
#                     dim=0,
#                     out=non_spec_query_start_loc_cpu[1:],
#                 )

#             assert num_accepted_tokens is not None
#             num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

#         # [Kunlun APC AUDIT V2 — per-mamba-group]
#         # Compare this group's kernel-read phys block id (from
#         # non_spec_state_indices_tensor) against preproc's published
#         # per-layer expected phys block id. A mismatch is a real per-group
#         # split-brain.  Enable via KUNLUN_APC_AUDIT_V2=1.
#         if (
#             os.environ.get("KUNLUN_APC_AUDIT_V2", "0") == "1"
#             and non_spec_state_indices_tensor is not None
#         ):
#             try:
#                 from vllm_kunlun.v1.worker.mamba_utils import (
#                     get_kunlun_per_layer_expected,
#                 )
#                 _ctx = get_kunlun_per_layer_expected()
#                 _req_ids = _ctx.get("req_ids")
#                 _per_layer = _ctx.get("per_layer_phys", {})
#                 _ln = self.layer_names[0] if self.layer_names else None
#                 _expected = _per_layer.get(_ln) if _ln else None
#                 if _req_ids is not None and _expected is not None:
#                     _kernel_read = non_spec_state_indices_tensor.cpu().tolist()
#                     # non_spec may have fewer rows than batch (filtered by
#                     # ~spec_sequence_masks). Align by row order: _expected
#                     # was published in input_batch.req_ids order. When
#                     # spec_sequence_masks is not None we'd need to filter;
#                     # for now audit only the no-spec-decode common path.
#                     if len(_kernel_read) == len(_expected):
#                         _step = _ctx.get("step", -1)
#                         for _i, (_k, _e) in enumerate(
#                             zip(_kernel_read, _expected)
#                         ):
#                             _rid = _req_ids[_i] if _i < len(_req_ids) else "?"
#                             _mark = (
#                                 " *** DIFF ***" if int(_k) != int(_e) else ""
#                             )
#                             print(
#                                 f"[APC_AUDIT_V2] step={_step} "
#                                 f"layer={_ln} req={_rid} row={_i} "
#                                 f"kernel_read={_k} preproc_expected={_e}"
#                                 f"{_mark}",
#                                 flush=True,
#                             )
#             except Exception as _e:
#                 print(f"[APC_AUDIT_V2] audit failed: {_e}", flush=True)

#         if num_prefills > 0:
#             # has_initial_state = context_lens_tensor > 0
#             has_initial_state = (context_lens_tensor > 0).to(torch.int32)
#             if spec_sequence_masks is not None:
#                 has_initial_state = has_initial_state[~spec_sequence_masks]
#                 assert non_spec_query_start_loc_cpu is not None
#             nums_dict, batch_ptr, token_chunk_offset_ptr = (
#                 compute_causal_conv1d_metadata(
#                     non_spec_query_start_loc_cpu,
#                     device=query_start_loc.device,
#                 )
#             )
#         else:
#             has_initial_state = None

#         # Prepare tensors for cudagraph
#         # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
#         batch_size = m.num_actual_tokens

#         if (
#             self.use_full_cuda_graph
#             and num_prefills == 0
#             and num_decodes == 0
#             and num_spec_decodes <= self.decode_cudagraph_max_bs
#             and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
#         ):
#             self.spec_state_indices_tensor[:num_spec_decodes].copy_(
#                 spec_state_indices_tensor, non_blocking=True
#             )
#             spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
#             spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

#             self.spec_sequence_masks[:num_spec_decodes].copy_(
#                 spec_sequence_masks, non_blocking=True
#             )
#             spec_sequence_masks = self.spec_sequence_masks[:batch_size]
#             spec_sequence_masks[num_spec_decodes:].fill_(False)

#             assert spec_token_masks is not None
#             self.spec_token_masks[: spec_token_masks.size(0)].copy_(
#                 spec_token_masks, non_blocking=True
#             )
#             spec_token_masks = self.spec_token_masks[: m.num_actual_tokens]
#             spec_token_masks[spec_token_masks.size(0) :].fill_(False)

#             assert non_spec_token_indx is not None and spec_token_indx is not None
#             self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
#                 non_spec_token_indx, non_blocking=True
#             )
#             non_spec_token_indx = self.non_spec_token_indx[
#                 : non_spec_token_indx.size(0)
#             ]

#             self.spec_token_indx[: spec_token_indx.size(0)].copy_(
#                 spec_token_indx, non_blocking=True
#             )
#             spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

#             self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
#                 spec_query_start_loc, non_blocking=True
#             )
#             spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
#             spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
#             spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

#             self.num_accepted_tokens[:num_spec_decodes].copy_(
#                 num_accepted_tokens, non_blocking=True
#             )
#             num_accepted_tokens = self.num_accepted_tokens[:batch_size]
#             num_accepted_tokens[num_spec_decodes:].fill_(1)

#         if (
#             self.use_full_cuda_graph
#             and num_prefills == 0
#             and num_spec_decodes == 0
#             and num_decodes <= self.decode_cudagraph_max_bs
#         ):
#             self.non_spec_state_indices_tensor[:num_decodes].copy_(
#                 non_spec_state_indices_tensor, non_blocking=True
#             )
#             non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
#                 :batch_size
#             ]
#             non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

#             self.non_spec_query_start_loc[: num_decodes + 1].copy_(
#                 non_spec_query_start_loc, non_blocking=True
#             )
#             non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
#             non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
#             non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

#         # ------------------------------------------------------------------
#         # [Kunlun APC AUDIT] Kernel-side READ-slot audit.
#         # Checks:
#         #   1. duplicates in non_spec_state_indices_tensor   → two reqs read/write
#         #      the same physical mamba block in this step (kernel collision)
#         #   2. per-req mismatch: kernel read block != preprocess write block
#         #      → split-brain between slot formulas
#         # Enable via KUNLUN_APC_AUDIT=1. Read-only.
#         # ------------------------------------------------------------------
#         if (
#             os.environ.get("KUNLUN_APC_AUDIT", "0") == "1"
#             and non_spec_state_indices_tensor is not None
#         ):
#             try:
#                 _kbids = non_spec_state_indices_tensor.detach().cpu().tolist()
#             except Exception:
#                 _kbids = []
#             # (1) duplicates
#             _seen: dict[int, int] = {}
#             _ctx2 = get_kunlun_running_slot_ctx()
#             _auth_reqs = _ctx2.get("req_ids") or []
#             for _i, _bid in enumerate(_kbids):
#                 if _bid < 0:
#                     continue
#                 _rname = _auth_reqs[_i] if _i < len(_auth_reqs) else f"idx{_i}"
#                 if _bid in _seen:
#                     _prev_i = _seen[_bid]
#                     _prev_r = (
#                         _auth_reqs[_prev_i]
#                         if _prev_i < len(_auth_reqs)
#                         else f"idx{_prev_i}"
#                     )
#                     print(
#                         f"[APC_AUDIT_KERNEL] *** KERNEL-DUP *** "
#                         f"block={_bid} reqs=({_prev_r},{_rname})",
#                         flush=True,
#                     )
#                 else:
#                     _seen[_bid] = _i
#             # (2) split-brain check
#             _auth_non_spec2 = _ctx2.get("non_spec_block_ids")
#             if (
#                 _auth_non_spec2 is not None
#                 and _auth_non_spec2.shape[0] == len(_kbids)
#             ):
#                 _auth_list2 = _auth_non_spec2.cpu().tolist()
#                 _seq_lens_list = (
#                     m.seq_lens.cpu().tolist() if m.seq_lens is not None else []
#                 )
#                 _raw_bt = (
#                     m.block_table_tensor.cpu().tolist()
#                     if m.block_table_tensor is not None
#                     else []
#                 )
#                 _align_bt = block_table_tensor.cpu().tolist()
#                 _bs = self.kv_cache_spec.block_size
#                 _nspec = self.kv_cache_spec.num_speculative_blocks
#                 for _i in range(len(_kbids)):
#                     if _kbids[_i] != _auth_list2[_i]:
#                         _rname = (
#                             _auth_reqs[_i] if _i < len(_auth_reqs) else f"idx{_i}"
#                         )
#                         _sl = (
#                             _seq_lens_list[_i]
#                             if _i < len(_seq_lens_list)
#                             else -1
#                         )
#                         _start = (_sl - 1) // _bs if _sl > 0 else -1
#                         _row_raw = _raw_bt[_i] if _i < len(_raw_bt) else []
#                         _row_align = _align_bt[_i] if _i < len(_align_bt) else []
#                         # find position of auth_block in raw & align row
#                         _pos_raw = (
#                             _row_raw.index(_auth_list2[_i])
#                             if _auth_list2[_i] in _row_raw
#                             else -1
#                         )
#                         _pos_align = (
#                             _row_align.index(_auth_list2[_i])
#                             if _auth_list2[_i] in _row_align
#                             else -1
#                         )
#                         print(
#                             f"[APC_AUDIT_KERNEL] *** SPLIT-BRAIN *** "
#                             f"req={_rname} "
#                             f"kernel_read={_kbids[_i]} "
#                             f"preproc_wrote={_auth_list2[_i]} "
#                             f"diff={_kbids[_i] - _auth_list2[_i]} "
#                             f"seq_len={_sl} bs={_bs} num_spec={_nspec} "
#                             f"kernel_start_idx={_start} "
#                             f"auth_pos_in_raw={_pos_raw} "
#                             f"auth_pos_in_align={_pos_align} "
#                             f"align_row={_row_align} "
#                             f"raw_tail={_row_raw[-5:] if len(_row_raw) >= 5 else _row_raw}",
#                             flush=True,
#                         )

#         attn_metadata = GDNAttentionMetadata(
#             num_prefills=num_prefills,
#             num_prefill_tokens=num_prefill_tokens,
#             num_decodes=num_decodes,
#             num_decode_tokens=num_decode_tokens,
#             num_spec_decodes=num_spec_decodes,
#             num_spec_decode_tokens=num_spec_decode_tokens,
#             num_actual_tokens=m.num_actual_tokens,
#             has_initial_state=has_initial_state,
#             has_initial_state_cpu=(
#                 has_initial_state.cpu() if has_initial_state is not None else None
#             ),
#             spec_query_start_loc=spec_query_start_loc,
#             non_spec_query_start_loc=non_spec_query_start_loc,
#             non_spec_query_start_loc_cpu=(
#                 non_spec_query_start_loc.cpu()
#                 if non_spec_query_start_loc is not None
#                 else None
#             ),
#             spec_state_indices_tensor=spec_state_indices_tensor,
#             non_spec_state_indices_tensor=non_spec_state_indices_tensor,
#             non_spec_state_indices_tensor_cpu=(
#                 non_spec_state_indices_tensor.cpu()
#                 if non_spec_state_indices_tensor is not None
#                 else None
#             ),
#             spec_sequence_masks=spec_sequence_masks,
#             spec_token_masks=spec_token_masks,
#             spec_token_indx=spec_token_indx,
#             non_spec_token_indx=non_spec_token_indx,
#             num_accepted_tokens=num_accepted_tokens,
#             nums_dict=nums_dict,
#             batch_ptr=batch_ptr,
#             token_chunk_offset_ptr=token_chunk_offset_ptr,
#         )
#         return attn_metadata

#     def build_for_cudagraph_capture(
#         self, common_attn_metadata: CommonAttentionMetadata
#     ):
#         """
#         This method builds the metadata for full cudagraph capture.
#         Currently, only decode is supported for full cudagraphs with Mamba.
#         """
#         m = common_attn_metadata

#         assert (
#             m.num_reqs <= self.decode_cudagraph_max_bs
#             and m.num_actual_tokens <= self.decode_cudagraph_max_bs
#         ), (
#             f"GDN only supports decode-only full CUDAGraph capture. "
#             f"Make sure batch size ({m.num_reqs}) <= "
#             f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
#             f"and number of tokens ({m.num_actual_tokens}) <= "
#             f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
#         )

#         num_accepted_tokens = torch.diff(m.query_start_loc)
#         num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

#         return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)


# gdn_attn.GDNAttentionMetadata = GDNAttentionMetadata
# gdn_attn.GDNAttentionMetadataBuilder = GDNAttentionMetadataBuilder


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass

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

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


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
        context_lens_tensor = m.compute_num_computed_tokens()
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
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
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
                non_spec_state_indices_tensor = block_table_tensor[
                    ~spec_sequence_masks_cpu, 0
                ]

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
            has_initial_state = (context_lens_tensor > 0).to(torch.int32)
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
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
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
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
            non_spec_state_indices_tensor_cpu=(
                non_spec_state_indices_tensor.cpu()
                if non_spec_state_indices_tensor is not None
                else None
            ),
            spec_sequence_masks=spec_sequence_masks,
            spec_token_masks=spec_token_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
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
