# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import os
from typing import Any

import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch

# ---------------------------------------------------------------------------
# [Kunlun APC fix — Split-brain elimination]
# Per-forward running-slot context populated by preprocess_mamba and consumed
# by GDNAttentionMetadataBuilder.build() so that the model-side
# `curr_state_idx = num_blocks - 1 - num_speculative_blocks`
# (used by preprocess_mamba / postprocess_mamba to WRITE state) and the
# metadata-side `non_spec_state_indices_tensor` (used by the kernel to READ
# / INPLACE-WRITE state) refer to **the same physical slot** for every
# request, every step.
#
# Without this, the upstream `mamba_get_block_table_tensor("align")` derives
# slot columns from `seq_lens` which can diverge from `len(block_ids)` in
# APC-hit multi-request batches, causing cross-request state pollution that
# amplifies across precision-test runs via persistent-block write-back.
# ---------------------------------------------------------------------------
_KUNLUN_RUNNING_SLOT_CTX: dict[str, Any] = {
    "req_ids": None,  # list[str] in input_batch.req_ids order
    "non_spec_block_ids": None,  # int32 tensor [B], physical running slot per req
    "spec_block_ids": None,  # int32 tensor [B, 1+num_spec_blocks], or None
}


# ---------------------------------------------------------------------------
# [Kunlun APC AUDIT V2 — per-mamba-group]
# Published by preprocess_mamba, consumed by GDNAttentionMetadataBuilder.build.
# Maps layer_name -> [phys_block_id per req in input_batch.req_ids order].
# The phys block id is where preprocess_mamba wrote (or would write) the
# running SSM state for that (layer, req). Kernel-side compares against its
# own non_spec_state_indices_tensor to catch real per-group split-brain.
# Enable via KUNLUN_APC_AUDIT_V2=1. Zero cost when disabled.
# ---------------------------------------------------------------------------
_KUNLUN_PER_LAYER_EXPECTED: dict[str, Any] = {
    "step": -1,
    "req_ids": None,
    "per_layer_phys": {},  # layer_name -> list[int]
}


def _per_layer_audit_enabled() -> bool:
    return os.environ.get("KUNLUN_APC_AUDIT_V2", "0") == "1"


def get_kunlun_per_layer_expected() -> dict[str, Any]:
    return _KUNLUN_PER_LAYER_EXPECTED


def get_kunlun_running_slot_ctx() -> dict[str, Any]:
    """Accessor for the per-forward authoritative running-slot context."""
    return _KUNLUN_RUNNING_SLOT_CTX


def clear_kunlun_running_slot_ctx() -> None:
    _KUNLUN_RUNNING_SLOT_CTX["req_ids"] = None
    _KUNLUN_RUNNING_SLOT_CTX["non_spec_block_ids"] = None
    _KUNLUN_RUNNING_SLOT_CTX["spec_block_ids"] = None


# ---------------------------------------------------------------------------
# [Kunlun APC AUDIT]
# Per-step ledger that records every dst physical block written by
# collect_mamba_copy_meta (preprocess + postprocess), tagged by (stage, req_id).
# At step end the auditor scans for:
#   WRITE-WRITE : same block written by two different reqs in this step
#   WRITE-READ  : block written by req A == running slot of req B (kernel read)
# Enable via KUNLUN_APC_AUDIT=1. Zero cost when disabled.
# ---------------------------------------------------------------------------
_APC_AUDIT_STEP = 0
_APC_AUDIT_WRITES: list[tuple[str, str, int, int, int]] = []
# each entry: (stage, req_id, layer_idx, dest_block_idx, dest_block_id)
_APC_AUDIT_LAST_WRITER: dict[int, tuple[int, str, str]] = {}
# physical_block_id -> (step, req_id, stage) of the LAST write (across all past steps)


def _apc_audit_enabled() -> bool:
    return os.environ.get("KUNLUN_APC_AUDIT", "0") == "1"


def _apc_audit_record(
    stage: str, req_id: str, layer_idx: int, dest_block_idx: int, dest_block_id: int
) -> None:
    if not _apc_audit_enabled():
        return
    _APC_AUDIT_WRITES.append((stage, req_id, layer_idx, dest_block_idx, dest_block_id))


def _apc_audit_flush_step(
    input_batch_req_ids: list[str], running_non_spec_block_ids: list[int]
) -> None:
    """Call once per forward, AFTER both preprocess and postprocess finished."""
    if not _apc_audit_enabled():
        return
    global _APC_AUDIT_STEP
    step = _APC_AUDIT_STEP
    _APC_AUDIT_STEP += 1

    writes = _APC_AUDIT_WRITES
    if not writes and not running_non_spec_block_ids:
        return

    # --- WRITE-WRITE: same block written by >1 reqs within this step
    dst_to_writers: dict[int, list[tuple[str, str, int]]] = {}
    for stage, req_id, layer_idx, _didx, dbid in writes:
        dst_to_writers.setdefault(dbid, []).append((stage, req_id, layer_idx))
    for dbid, writers in dst_to_writers.items():
        reqs_touching = {r for (_s, r, _l) in writers}
        if len(reqs_touching) > 1:
            print(
                f"[APC_AUDIT] step={step} *** WRITE-WRITE *** "
                f"block={dbid} writers={writers}",
                flush=True,
            )

    # --- WRITE-READ: someone writes into a block that is another req's running slot
    req_to_running: dict[str, int] = {}
    for r, bid in zip(input_batch_req_ids, running_non_spec_block_ids):
        req_to_running[r] = bid
    run_to_req: dict[int, str] = {v: k for k, v in req_to_running.items()}

    for stage, wreq, layer_idx, _didx, dbid in writes:
        owner = run_to_req.get(dbid)
        if owner is not None and owner != wreq:
            print(
                f"[APC_AUDIT] step={step} *** WRITE-READ *** "
                f"block={dbid} written_by=(req={wreq},stage={stage},layer={layer_idx}) "
                f"running_slot_of=req={owner}",
                flush=True,
            )

    # --- Cross-step last-writer tracking: print when same block is re-targeted
    #     by a DIFFERENT req within a short window.
    for stage, req_id, layer_idx, _didx, dbid in writes:
        prev = _APC_AUDIT_LAST_WRITER.get(dbid)
        if prev is not None:
            prev_step, prev_req, prev_stage = prev
            if prev_req != req_id and (step - prev_step) <= 64:
                print(
                    f"[APC_AUDIT] step={step} cross-step block reuse "
                    f"block={dbid} prev=({prev_stage},req={prev_req},step={prev_step}) "
                    f"curr=({stage},req={req_id},layer={layer_idx})",
                    flush=True,
                )
        _APC_AUDIT_LAST_WRITER[dbid] = (step, req_id, stage)

    # Reset per-step ledger
    _APC_AUDIT_WRITES.clear()


def _make_uint8_view_from_ptr(
    ptr: int, size: int, device: torch.device
) -> torch.Tensor:
    storage = torch._C._construct_storage_from_data_pointer(ptr, device, size)
    tensor = torch.empty(0, dtype=torch.uint8, device=device)
    return tensor.set_(storage, 0, (size,), (1,))


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    if batch == 0:
        return

    device = src_ptrs.device
    src_ptrs_cpu = src_ptrs.detach().cpu().tolist()
    dst_ptrs_cpu = dst_ptrs.detach().cpu().tolist()
    sizes_cpu = sizes.detach().cpu().tolist()

    for src_ptr, dst_ptr, size in zip(src_ptrs_cpu, dst_ptrs_cpu, sizes_cpu):
        if size <= 0 or src_ptr == dst_ptr:
            continue
        src = _make_uint8_view_from_ptr(src_ptr, size, device)
        dst = _make_uint8_view_from_ptr(dst_ptr, size, device)
        dst.copy_(src, non_blocking=False)
        torch.cuda.synchronize()


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i in range(len(kv_cache_config.kv_cache_groups)):
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
        if isinstance(kv_cache_spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(kv_cache_spec)
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"
    assert all(mamba_specs[0] == spec for spec in mamba_specs)
    return mamba_group_ids, mamba_specs[0]


def collect_mamba_copy_meta(
    src_state_list: list[int],
    dest_state_list: list[int],
    num_elements_list: list[int],
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
    _audit_stage: str = "",
    _audit_req_id: str = "",
):
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    _layer_idx = 0
    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache[0]
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )

                src_state_list.append(copy_spec.start_addr)
                dest_state_list.append(state[dest_block_id].data_ptr())
                num_elements_list.append(copy_spec.num_elements * state.element_size())
            # audit at the (req, layer) granularity (not per state-tensor)
            if _audit_stage:
                _apc_audit_record(
                    _audit_stage,
                    _audit_req_id,
                    _layer_idx,
                    dest_block_idx,
                    int(dest_block_id),
                )
            _layer_idx += 1


def do_mamba_copy_block(
    src_state_list: list[int],
    dest_state_list: list[int],
    num_elements_list: list[int],
):
    if len(src_state_list) == 0:
        return
    assert len(src_state_list) == len(dest_state_list)
    assert len(src_state_list) == len(num_elements_list)
    src_state_ptrs = torch.tensor(src_state_list, device="cuda", dtype=torch.int64)
    dst_state_ptrs = torch.tensor(dest_state_list, device="cuda", dtype=torch.int64)
    num_elements = torch.tensor(num_elements_list, device="cuda", dtype=torch.int64)

    # batch_memcpy(src_state_ptrs, dst_state_ptrs, num_elements)
    torch.ops.xspeedgate_ops.batch_memcpy(src_state_ptrs, dst_state_ptrs, num_elements)


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)
    src_state_list: list[int] = []
    dest_state_list: list[int] = []
    num_elements_list: list[int] = []
    # [APC AUDIT V3] Per-req copy metadata for content-level fingerprinting.
    # Captured BEFORE do_mamba_copy_block so we can fingerprint src pre-copy
    # and dst post-copy to verify APC-restore semantics.
    _v3_enabled = os.environ.get("KUNLUN_APC_AUDIT_V3", "0") == "1"
    _v3_copy_meta: list[dict[str, Any]] = []  # per (req, group0-only) trace entry

    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        _had_mamba_state_idx = prev_state_idx is not None
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        # [Kunlun fix] Compute num_blocks from tokens (matching upstream),
        # NOT from len(block_ids). In "align" mode under multi-request APC,
        # len(block_ids) can diverge from cdiv(tokens, block_size), causing
        # the COPY destination (ssm_state slot) to differ from the slot the
        # kernel reads via non_spec_state_indices_tensor → garbled output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        # = cdiv(num_computed+num_scheduled, block_size) - 1
        # This matches metadata builder's (seq_lens-1)//block_size formula
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            collect_mamba_copy_meta(
                src_state_list,
                dest_state_list,
                num_elements_list,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
                _audit_stage="pre",
                _audit_req_id=req_id,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
            if _v3_enabled:
                _v3_copy_meta.append(
                    {
                        "req_id": req_id,
                        "num_computed": req_state.num_computed_tokens,
                        "num_sched": num_scheduled_tokens,
                        "prev_state_idx": prev_state_idx,
                        "curr_state_idx": curr_state_idx,
                        "had_prev_mamba_idx": _had_mamba_state_idx,
                        "num_accepted": int(input_batch.num_accepted_tokens_cpu[i]),
                    }
                )

    # [APC AUDIT V3] Fingerprint src (pre-copy) state for every pending copy.
    if _v3_enabled and _v3_copy_meta:
        try:
            for _m in _v3_copy_meta:
                _rid = _m["req_id"]
                _rs = requests[_rid]
                _src_fps: list[str] = []
                _dst_fps_pre: list[str] = []
                for _gid in mamba_group_ids:
                    _b = _rs.block_ids[_gid]
                    _src_blk = (
                        _b[_m["prev_state_idx"]]
                        if _m["prev_state_idx"] < len(_b)
                        else -1
                    )
                    _dst_blk = (
                        _b[_m["curr_state_idx"]]
                        if _m["curr_state_idx"] < len(_b)
                        else -1
                    )
                    _ln = kv_cache_config.kv_cache_groups[_gid].layer_names[0]
                    _attn = forward_context[_ln]
                    _kv: list[torch.Tensor] = _attn.kv_cache[0]
                    # kv[0]=conv_state, kv[1]=ssm_state typically
                    _ssm = _kv[-1]  # last tensor is ssm_state by convention
                    _src_t = _ssm[_src_blk] if 0 <= _src_blk < _ssm.shape[0] else None
                    _dst_t = _ssm[_dst_blk] if 0 <= _dst_blk < _ssm.shape[0] else None

                    def _fp(t):
                        if t is None:
                            return "N/A"
                        _tf = t.detach().float()
                        return (
                            f"norm={_tf.norm().item():.4g} "
                            f"mean={_tf.mean().item():.4g} "
                            f"first={_tf.flatten()[:3].tolist()} "
                            f"nan={torch.isnan(t).sum().item()}"
                        )

                    _m[f"src_blk_g{_gid}"] = int(_src_blk)
                    _m[f"dst_blk_g{_gid}"] = int(_dst_blk)
                    _src_fps.append(f"g{_gid}[src_blk={int(_src_blk)}]:{_fp(_src_t)}")
                    _dst_fps_pre.append(
                        f"g{_gid}[dst_blk={int(_dst_blk)}]:{_fp(_dst_t)}"
                    )
                _m["_src_fps"] = _src_fps
                _m["_dst_fps_pre"] = _dst_fps_pre
        except Exception as _e:
            print(f"[APC_AUDIT_V3] pre-copy fp failed: {_e}", flush=True)

    do_mamba_copy_block(src_state_list, dest_state_list, num_elements_list)

    # [APC AUDIT V3 dup-detect] Cross-req src block collision within the batch.
    # If two different reqs read from the SAME physical src block in the same
    # mamba group this step → strong evidence of block reuse / eviction race.
    if _v3_enabled and _v3_copy_meta:
        try:
            for _gid in mamba_group_ids:
                _key = f"src_blk_g{_gid}"
                _seen: dict[int, list[str]] = {}
                for _m in _v3_copy_meta:
                    _blk = _m.get(_key, -1)
                    if _blk < 0:
                        continue
                    _seen.setdefault(_blk, []).append(_m["req_id"])
                for _blk, _reqs in _seen.items():
                    if len(_reqs) >= 2:
                        print(
                            f"[APC_AUDIT_V3] *** DUP-SRC g{_gid} blk={_blk} "
                            f"reqs={_reqs} ***",
                            flush=True,
                        )
        except Exception as _e:
            print(f"[APC_AUDIT_V3] dup-detect failed: {_e}", flush=True)

    # [APC AUDIT V3] Fingerprint dst (post-copy) & print side-by-side.
    if _v3_enabled and _v3_copy_meta:
        try:
            for _m in _v3_copy_meta:
                _rid = _m["req_id"]
                _rs = requests[_rid]
                _dst_fps_post: list[str] = []
                for _gid in mamba_group_ids:
                    _dst_blk = _m.get(f"dst_blk_g{_gid}", -1)
                    _ln = kv_cache_config.kv_cache_groups[_gid].layer_names[0]
                    _attn = forward_context[_ln]
                    _ssm = _attn.kv_cache[0][-1]
                    _dst_t = _ssm[_dst_blk] if 0 <= _dst_blk < _ssm.shape[0] else None
                    _tf = _dst_t.detach().float() if _dst_t is not None else None
                    _fp_str = (
                        f"norm={_tf.norm().item():.4g} "
                        f"mean={_tf.mean().item():.4g} "
                        f"first={_tf.flatten()[:3].tolist()} "
                        f"nan={torch.isnan(_dst_t).sum().item()}"
                        if _tf is not None
                        else "N/A"
                    )
                    _dst_fps_post.append(f"g{_gid}[dst_blk={_dst_blk}]:{_fp_str}")
                print(
                    f"[APC_AUDIT_V3] req={_m['req_id']} "
                    f"num_comp={_m['num_computed']} num_sched={_m['num_sched']} "
                    f"prev_idx={_m['prev_state_idx']} curr_idx={_m['curr_state_idx']} "
                    f"had_prev_mamba_idx={_m['had_prev_mamba_idx']} "
                    f"num_accepted_raw={_m['num_accepted']}",
                    flush=True,
                )
                for _s, _dp, _dpost in zip(
                    _m.get("_src_fps", []),
                    _m.get("_dst_fps_pre", []),
                    _dst_fps_post,
                ):
                    print(
                        f"[APC_AUDIT_V3]   src_pre  {_s}",
                        flush=True,
                    )
                    print(
                        f"[APC_AUDIT_V3]   dst_pre  {_dp}",
                        flush=True,
                    )
                    print(
                        f"[APC_AUDIT_V3]   dst_post {_dpost}",
                        flush=True,
                    )
        except Exception as _e:
            print(f"[APC_AUDIT_V3] post-copy fp failed: {_e}", flush=True)

    # [DIAG] Split-brain 检测：打印 preprocess_mamba 拷贝到的物理 slot，
    # 以及 metadata builder 给 kernel 读入用的物理 slot。
    # 这两者必须一致，否则 kernel 从错误的 slot 读入脏状态 → 乱码。
    # 设 KUNLUN_MAMBA_DEBUG=1 启用。
    if os.environ.get("KUNLUN_MAMBA_DEBUG", "0") == "1":
        idx = 0
        for i, req_id in enumerate(input_batch.req_ids):
            req_state = requests[req_id]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_blocks = (
                cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
                + num_speculative_blocks
            )
            curr_state_idx = num_blocks - 1 - num_speculative_blocks
            dest_block_id = req_state.block_ids[mamba_group_ids[0]][curr_state_idx]
            prev_si = mamba_state_idx.get(req_id)
            if prev_si is None:
                prev_si = (req_state.num_computed_tokens - 1) // block_size
            n_c = 0
            if prev_si != -1 and prev_si != curr_state_idx:
                n_c = (
                    len(mamba_group_ids)
                    * len(
                        kv_cache_config.kv_cache_groups[mamba_group_ids[0]].layer_names
                    )
                    * len(mamba_state_copy_funcs)
                )
            print(
                f"[MAMBA_RECONCILE] req={req_id} "
                f"num_comp={req_state.num_computed_tokens} "
                f"num_sched={num_scheduled_tokens} "
                f"block_size={block_size} "
                f"auth_dest={dest_block_id} "
                f"upstream_slot_idx={(req_state.num_computed_tokens+num_scheduled_tokens-1)//block_size} "
                f"blockids={req_state.block_ids[mamba_group_ids[0]]}",
                flush=True,
            )
            idx += n_c

    # [Kunlun APC fix] Publish authoritative running-slot physical block IDs
    # so gdn_attn.py can compare with the upstream block_table slice.
    _KUNLUN_RUNNING_SLOT_CTX["req_ids"] = list(input_batch.req_ids)
    auth_non_spec_list: list[int] = []
    for req_id in input_batch.req_ids:
        req_state = requests[req_id]
        bidx = mamba_state_idx.get(req_id)
        if bidx is None:
            bidx = (req_state.num_computed_tokens - 1) // block_size
        if bidx < 0:
            bidx = 0
        b = req_state.block_ids[mamba_group_ids[0]]
        auth_non_spec_list.append(int(b[bidx]) if bidx < len(b) else -1)
    _KUNLUN_RUNNING_SLOT_CTX["non_spec_block_ids"] = torch.tensor(
        auth_non_spec_list,
        dtype=torch.int32,
        device="cpu",
    )

    # [Kunlun APC AUDIT V2] Publish per-layer expected phys block id list
    # (one int per req, aligned with input_batch.req_ids). Each mamba group
    # has its OWN physical block pool, so we publish per-layer values from
    # that group's req_state.block_ids[group_id][curr_state_idx]. GDN builder
    # looks up by self.layer_names[0].
    if _per_layer_audit_enabled():
        per_layer: dict[str, list[int]] = {}
        for gid in mamba_group_ids:
            layer_names = kv_cache_config.kv_cache_groups[gid].layer_names
            phys_list: list[int] = []
            for req_id in input_batch.req_ids:
                req_state = requests[req_id]
                bidx = mamba_state_idx.get(req_id)
                if bidx is None:
                    bidx = (req_state.num_computed_tokens - 1) // block_size
                if bidx < 0:
                    bidx = 0
                b = req_state.block_ids[gid]
                phys_list.append(int(b[bidx]) if bidx < len(b) else -1)
            for ln in layer_names:
                per_layer[ln] = phys_list
        _KUNLUN_PER_LAYER_EXPECTED["step"] = (
            _KUNLUN_PER_LAYER_EXPECTED.get("step", -1) + 1
        )
        _KUNLUN_PER_LAYER_EXPECTED["req_ids"] = list(input_batch.req_ids)
        _KUNLUN_PER_LAYER_EXPECTED["per_layer_phys"] = per_layer

    return mamba_state_idx


def postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
):
    """
    If a blocks is converted from partial block to full block in this step, copy the
    state from the block for running state to the new full block.
    """
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    # NOTE: can be optimized as this function always returns the same result
    mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
    src_state_list: list[int] = []
    dest_state_list: list[int] = []
    num_elements_list: list[int] = []
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = (
            num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        )
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        aligned_new_computed_tokens = (
            new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        )
        # TODO: how to ensure all blocks that cache_blocks called are cached here?
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            collect_mamba_copy_meta(
                src_state_list,
                dest_state_list,
                num_elements_list,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
                _audit_stage="post",
                _audit_req_id=req_id,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(src_state_list, dest_state_list, num_elements_list)

    # [Kunlun APC AUDIT] per-step collision scan
    if _apc_audit_enabled():
        mamba_group_ids_audit, _ = get_mamba_groups(kv_cache_config)
        running_blks: list[int] = []
        for req_id in input_batch.req_ids:
            req_state = requests[req_id]
            bidx = mamba_state_idx.get(req_id, -1)
            b = req_state.block_ids[mamba_group_ids_audit[0]]
            running_blks.append(int(b[bidx]) if 0 <= bidx < len(b) else -1)
        _apc_audit_flush_step(list(input_batch.req_ids), running_blks)
