# SPDX-License-Identifier: Apache-2.0
"""Q38 XPU PORT: QSA 算子的昆仑实现，替掉 ``ops/qsa.py`` 里的 triton kernel。

XPU 上没有 triton，``ops/qsa.py`` 的 8 个 kernel 全部不可用（QSA 因此只能靠
``Q38_DISABLE_QSA=1`` 退回 dense）。本模块提供同名替身：

- indexer 打分（``qsa_mqa_paged``）→ ``kunlun_ops.c4a_paged_mqa_logits``
- block 级 top-k（``qsa_select_paged_tokens`` 内部）→ ``kunlun_ops.fast_topkv2_dynamic``
- 主稀疏 attention（``qsa_sparse_paged_attention``）→ ``kunlun_ops.msa_sparse_attention``
- 其余三件（block→token 展开、压缩组池化、cache 行写入）是 O(rows × head_dim)
  的整数/搬运操作，kunlun_ops 里没有语义对得上的算子，先用 torch 写；
  profile 里冒头再下沉成 XSpeedGate kernel。

已实测的硬约束（详见算子 #5 结论文档）：

- ``c4a_paged_mqa_logits`` 的 ``head_num`` 必须 >= 16，而 ``indexer_n_heads=4``
  → q/weight 零填充到 16（``relu(0·k)=0`` 且 weight=0，贡献恒为 0，实测 rel 1.7e-07）；
- ``weight`` 必须 fp32，``k_scale`` 全 1 即可（bf16 未量化 cache）；
- q 必须 4 维、logits 必须 3 维。docstring 写的"变长 3 维 q"在 pybind 层被
  ``q.sizes().size() == 4`` 直接拒掉 → 把**每个 query token 当成一条独立 batch**，
  ``context_lens = query_position + 1``（未压缩长度），``max_seq_q`` 恒为 1；
- ``fast_topkv2_dynamic`` 只保证前 ``min(length, topk)`` 个槽位是合法下标，
  后面是**未初始化内存**；框架侧 expand 只读前 ``complete_blocks`` 个，正好安全；
- ``msa_sparse_attention`` 会跳过 ``topk_idx`` 里的 ``-1``（与位置无关），
  可以直接吃 expand 产出的 -1 填充。
"""

from __future__ import annotations

import math
import os

import torch

# 出问题时置 Q38_KL_QSA=0 让 QSAIndexer / QSA backend 走回原来的 triton 导入
# （XPU 上等价于直接报错），用来区分"昆仑实现算错了"和"框架侧接线错了"。
_FORCE_OFF = os.environ.get("Q38_KL_QSA", "1") != "1"

_C4A_MIN_HEADS = 16
# 打分中间结果按行分块，和 triton 版同一个预算，避免长 context 时 logits 爆显存。
_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
# expand 的中间张量是 [rows, token_topk + CR - 1]，行数再单独收一次。
_EXPAND_ROW_CHUNK = 512
# Q38_QSA_SYNC=1：每个算子后面插一次同步，把 XPU 的异步 illegal memory access
# 钉到具体调用点（XPU 上 CUDA_LAUNCH_BLOCKING 不起作用）。默认关。
_SYNC = os.environ.get("Q38_QSA_SYNC", "0") == "1"

# msa_sparse_attention 的 topk 实测上限：<= 200 正常，>= 201 一律设备 fault
# （invalid program counter / illegal memory access），与 seq_len、batch、
# prefill/decode、head_dim、page_size、blocks_per_req、topk_idx 张量宽度都无关，
# 只跟传进去的 topk 参数有关。QSA 要的是 2051，远超上限。
# 二分过程和最小复现见 doc/qsa-接kunlun_ops实现.md。
_MSA_TOPK_LIMIT = 200
# Q38_QSA_TOPK_CAP=<=200：上下文超过上限时把稀疏预算降到 CAP 列继续跑（结果是近似，
# 只保留 top-k 算子返回的前 CAP 个下标），默认 0 = 直接报错，不静默算错。
_TOPK_CAP = int(os.environ.get("Q38_QSA_TOPK_CAP", "0") or 0)
# Q38_QSA_DUMP=N：把前 N 次 qsa_sparse_paged_attention 的下标集合摘要打出来
# （每行有效列数 vs 因果上界），用来判断"稀疏下标集合是否等于完整因果前缀"。
_DUMP_LEFT = [int(os.environ.get("Q38_QSA_DUMP", "0") or 0)]


def _sync(tag: str) -> None:
    if not _SYNC:
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"[Q38_QSA_SYNC] device fault right after {tag}") from exc


def _block_table_scale(key_cache: torch.Tensor) -> int:
    """逻辑 block id 到"按块连续"物理 block id 的倍数。

    混合 KV cache 下 ``key_cache`` 是打包布局 ``[2, blocks, kv_heads, page, dim]``
    的切片，``stride(0)`` 会是块内连续步长的整数倍（K/V 打包时是 2 倍）。
    读侧算子只吃 block id、自己按连续步长算偏移，所以 id 必须放大同样的倍数。
    """

    contiguous = key_cache.shape[1] * key_cache.shape[2] * key_cache.shape[3]
    actual = key_cache.stride(0)
    if actual == contiguous or contiguous <= 0 or actual % contiguous:
        return 1
    return actual // contiguous


# 3 个 int64 的 mrope 位置在 bf16 行里占 3 * 4 个元素。
_BF16_PER_INT64 = 4

_KUNLUN = None


def _kunlun():
    global _KUNLUN
    if _FORCE_OFF:
        raise RuntimeError("Q38_KL_QSA=0：昆仑 QSA 实现被显式关闭")
    if _KUNLUN is None:
        import kunlun_ops

        _KUNLUN = kunlun_ops
    return _KUNLUN


def available() -> bool:
    """kunlun_ops 里三个核心算子是否都在。"""

    if _FORCE_OFF:
        return False
    try:
        mod = _kunlun()
    except Exception:
        return False
    return all(
        hasattr(mod, name)
        for name in (
            "c4a_paged_mqa_logits",
            "fast_topkv2_dynamic",
            "msa_sparse_attention",
        )
    )


def _safe_request(token_to_req: torch.Tensor, num_requests: int) -> torch.Tensor:
    """把 token_to_req 夹到合法区间，并返回可直接 index_select 的 int64 下标。"""

    # 不能用 clamp_：token_to_req 本来就是 int64 时 .to() 会原地返回调用方的张量。
    return token_to_req.to(torch.int64).clamp(0, max(num_requests - 1, 0))


def expand_qsa_block_indices_cuda(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """把 block 级 top-k 展开成 token 下标，并补上 open group 的 causal 尾部。

    逐字对照 ``ops/qsa.py::_expand_qsa_indices_kernel``：
    前 ``complete_blocks * CR`` 列是完整压缩块展开出来的 token，
    紧随其后最多 ``CR - 1`` 列是当前未满组里已经可见的 token，其余填 -1。
    """

    ratio = int(compress_ratio)
    if ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if int(token_topk) % ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = int(token_topk) // ratio
    width = int(token_topk) + ratio - 1
    rows = query_positions.numel()
    if block_indices.shape != (rows, block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions")
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    device = block_indices.device
    if out is None:
        out = torch.empty((rows, width), dtype=torch.int32, device=device)
    elif out.shape != (rows, width):
        raise ValueError("QSA expansion output has an invalid shape")
    if not rows:
        return out

    num_requests = sequence_lengths.shape[0]
    request = _safe_request(token_to_req, num_requests)
    in_range = (token_to_req >= 0) & (token_to_req < num_requests)
    seq_all = torch.where(
        in_range,
        sequence_lengths.to(torch.int32).index_select(0, request),
        torch.zeros_like(sequence_lengths[:1].to(torch.int32)).expand(rows),
    )
    qp_all = query_positions.to(torch.int32)
    cols = torch.arange(width, dtype=torch.int32, device=device)
    rank = (cols // ratio).clamp(max=block_topk - 1).to(torch.int64)
    offset = cols % ratio
    blocks_i32 = block_indices.to(torch.int32)

    for start in range(0, rows, _EXPAND_ROW_CHUNK):
        stop = min(start + _EXPAND_ROW_CHUNK, rows)
        seq = seq_all[start:stop, None]
        qp = qp_all[start:stop, None]
        complete = torch.minimum((qp + 1) // ratio, seq // ratio).clamp(max=block_topk)
        expanded_count = complete * ratio
        tail_start = ((qp + 1) // ratio) * ratio
        tail_count = (qp + 1) - tail_start

        is_expanded = cols[None, :] < expanded_count
        expanded = blocks_i32[start:stop].index_select(1, rank) * ratio + offset[None, :]
        tail_offset = cols[None, :] - expanded_count
        is_tail = (
            ~is_expanded & (tail_offset < tail_count) & (tail_offset < ratio - 1)
        )
        token = torch.where(is_expanded, expanded, tail_start + tail_offset)
        valid = (is_expanded | is_tail) & (token >= 0) & (token < seq)
        out[start:stop].copy_(torch.where(valid, token, torch.full_like(token, -1)))
    return out


# c4a 的 lod 分支实测（见单测 T0）：``qlod_cpu`` 可以传 None，
# 但 ``context_lens_cpu`` **必须**给 CPU 张量，否则 block 层直接返回非零
# （c4a_paged_mqa_logits_block.cc:86）。也就是说这个算子拿不到"纯 XPU 侧 lod"
# 的分支，decode 图重放前必须已经有 host 侧的 context 长度。
# → 调用方最好把 metadata 里现成的 CPU 张量传进来（``query_positions_cpu``），
#   否则这里只能 ``.cpu()``，每层一次 D2H 同步。


def _c4a_paged(
    mod,
    q4: torch.Tensor,
    weight: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale: torch.Tensor,
    logits3: torch.Tensor,
    max_context_len: int,
    qlod_xpu: torch.Tensor,
    context_lens: torch.Tensor,
    context_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    ratio: int,
) -> None:
    mod.c4a_paged_mqa_logits(
        q4,
        weight,
        k_cache,
        k_scale,
        logits3,
        int(max_context_len),
        None,
        qlod_xpu,
        context_lens_cpu,
        context_lens,
        block_table,
        ratio,
        True,
    )
    _sync("c4a_paged_mqa_logits")


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    num_columns: int | None = None,
    score_scale: float | None = None,
    query_positions_cpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """indexer 打分：``logits[t, c] = sum_h relu(q[t,h]·k[c]) / sqrt(head_dim)``。

    返回 ``(logits[rows, columns], visible_blocks[rows])``，语义与
    ``ops/qsa.py::qsa_mqa_paged`` 一致（不可见列被填成 -inf）。
    """

    if q.ndim != 3:
        raise ValueError("QSA query must be [rows, heads, head_dim]")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional")
    if token_to_req.shape != (q.shape[0],) or query_positions.shape != (q.shape[0],):
        raise ValueError("QSA per-row metadata must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    ratio = int(compress_ratio)
    if ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    divisor = math.sqrt(q.shape[2]) if score_scale is None else float(score_scale)
    if divisor <= 0:
        raise ValueError("QSA score scale must be positive")

    rows, heads, dim = q.shape
    device = q.device
    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else int(num_columns)
    logits = torch.empty((rows, columns), dtype=torch.float32, device=device)
    request = _safe_request(token_to_req, page_table.shape[0])
    qp = query_positions.to(torch.int32)
    seq = sequence_lengths.to(torch.int32).index_select(0, request)
    visible = torch.minimum((qp + 1) // ratio, seq // ratio).clamp(max=columns)
    visible = visible.to(torch.int32)
    if not rows or not columns:
        return logits, visible

    heads_pad = max(_C4A_MIN_HEADS, heads)
    q4 = q.new_zeros((rows, 1, heads_pad, dim))
    q4[:, 0, :heads] = q
    weight = torch.zeros((rows, 1, heads_pad), dtype=torch.float32, device=device)
    weight[:, 0, :heads] = 1.0 / divisor
    # 每个 query token 自成一条 batch：context_lens 是"未压缩"的可见长度。
    context_lens = (qp + 1).clamp(max=columns * ratio).contiguous()
    if query_positions_cpu is None:
        context_lens_cpu = context_lens.cpu()
    else:
        context_lens_cpu = (
            (query_positions_cpu.to(torch.int32) + 1)
            .clamp(max=columns * ratio)
            .contiguous()
        )
        if context_lens_cpu.shape != (rows,):
            raise ValueError("QSA host-side query positions must match query rows")
    block_table = page_table.index_select(0, request).to(torch.int32).contiguous()
    k_scale = torch.ones((rows, columns), dtype=torch.float32, device=device)
    _c4a_paged(
        _kunlun(),
        q4,
        weight,
        k_cache,
        k_scale,
        logits.view(rows, 1, columns),
        columns * ratio,
        torch.arange(rows + 1, dtype=torch.int32, device=device),
        context_lens,
        context_lens_cpu,
        block_table,
        ratio,
    )
    return logits, visible


def qsa_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    query_positions_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    """打分 → block 级 top-k → 展开成 token 下标。

    ``query_positions_cpu`` 是 ``query_positions`` 的 host 侧副本；不给的话
    ``qsa_mqa_paged`` 只能自己 ``.cpu()``，每层一次 D2H 同步（decode 图捕获会失败）。
    """

    ratio = int(compress_ratio)
    rows = q.shape[0]
    width = int(token_topk) + ratio - 1
    if out is None:
        out = torch.empty((rows, width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out

    mod = _kunlun()
    columns = page_table.shape[1] * k_cache.shape[1]
    block_topk = int(token_topk) // ratio
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        logits, visible = qsa_mqa_paged(
            q[start:stop],
            k_cache,
            page_table,
            token_to_req[start:stop],
            query_positions[start:stop],
            sequence_lengths,
            ratio,
            query_positions_cpu=(
                None if query_positions_cpu is None
                else query_positions_cpu[start:stop]
            ),
        )
        # fast_topkv2_dynamic 只保证前 min(length, topk) 个槽位合法，
        # 后面是脏数据；expand 只读前 complete_blocks(<= visible) 个，安全。
        # length 夹到 >= 1：visible=0 的行（context 不足一个压缩组）expand 会整行填 -1。
        take = min(block_topk, columns)
        picked = mod.fast_topkv2_dynamic(logits, visible.clamp(min=1), take)
        _sync("fast_topkv2_dynamic")
        if take == block_topk:
            blocks = picked
        else:
            blocks = logits.new_zeros((stop - start, block_topk), dtype=torch.int32)
            blocks[:, :take] = picked
        expand_qsa_block_indices_cuda(
            blocks,
            query_positions[start:stop],
            sequence_lengths,
            token_to_req[start:stop],
            ratio,
            int(token_topk),
            out[start:stop],
        )
    return out


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """按 ``logical_indices`` 做稀疏 GQA。``-1`` 会被算子跳过（实测与位置无关）。

    比 triton 版多要三个 metadata：``seq_lens`` / ``max_query_len`` /
    ``max_seq_len``。前者 msa 算子需要 kv LOD，后两者是 host 侧的形状上界；
    triton 版是逐行独立算的所以不需要。只传 XPU 侧 LOD、CPU 侧给 None，
    避免 decode 图重放时的 D2H 同步。
    """

    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    # msa 要求 cache 是 [num_pages, head_num_kv, paged_block_size, head_dim]。
    # 框架侧的 key_cache 是 kv_cache.transpose(1, 2).split(...) 出来的，正好是这个序；
    # 布局一旦变了这里必须炸出来，而不是静默算错。
    if k_cache.shape[3] != q.shape[2] or q.shape[1] % k_cache.shape[1]:
        raise ValueError(
            "QSA sparse attention expects k_cache "
            f"[pages, kv_heads, page, head_dim]; got {tuple(k_cache.shape)} "
            f"against q {tuple(q.shape)}"
        )
    if out is None:
        out = torch.empty_like(q)
    tokens, heads, dim = q.shape
    heads_kv = k_cache.shape[1]
    if logical_indices.shape[0] != tokens:
        raise ValueError("QSA sparse attention indices must match query rows")
    if not tokens:
        return out

    mod = _kunlun()
    batch = block_table.shape[0]
    # Q38 XPU PORT：昆仑的读侧算子（prefill_attention / paged_attention /
    # msa_sparse_attention）都按"块连续"假设自己算 block 偏移，而混合 KV cache 下
    # key_cache 是 [2, blocks, ...] 打包布局的切片，stride(0) 是块内连续步长的 2 倍。
    # 逻辑 block id 必须按这个倍数放大，否则算子读到的是错页——和
    # KunlunAttentionImpl._get_block_table_scale 同一套逻辑。
    block_table = block_table.to(torch.int32) * _block_table_scale(k_cache)
    topk_num = logical_indices.shape[1]
    # 有效列只落在每行前 (query_pos + 1) 列（见 expand_qsa_block_indices_cuda 的
    # expanded / tail 布局），而 query_pos + 1 <= seq_len <= max_seq_len，所以尾部
    # 整列都是 -1，可以按 host 侧上界裁掉：既省算力，也让短上下文绕开算子的 topk 上限。
    usable = max(1, min(topk_num, int(max_seq_len)))
    if usable < topk_num:
        logical_indices = logical_indices[:, :usable].contiguous()
        topk_num = usable
    if topk_num > _MSA_TOPK_LIMIT:
        if _TOPK_CAP <= 0:
            raise NotImplementedError(
                f"kunlun_ops.msa_sparse_attention 在 topk={topk_num} 下会设备 fault"
                f"（实测上限 {_MSA_TOPK_LIMIT}，max_seq_len={int(max_seq_len)}）。"
                "把上下文压到上限以内，或设 Q38_QSA_TOPK_CAP=<=200 用降级预算"
                "（近似结果），或等算子修复。"
            )
        cap = min(_TOPK_CAP, _MSA_TOPK_LIMIT)
        logical_indices = logical_indices[:, :cap].contiguous()
        topk_num = cap
    counts = torch.bincount(
        _safe_request(token_to_req, batch), minlength=batch
    ).to(torch.int32)
    lod_q = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
    lod_q[1:] = counts.cumsum(0).to(torch.int32)
    lod_kv = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
    lod_kv[1:] = seq_lens.to(torch.int32).cumsum(0).to(torch.int32)
    # topk_idx 要 [tokens, head_num_kv, topk]，QSA 的下标对所有头共享。
    topk_idx = (
        logical_indices.to(torch.int32)
        .unsqueeze(1)
        .expand(tokens, heads_kv, topk_num)
        .contiguous()
    )
    scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)
    dump = _DUMP_LEFT[0] > 0
    if dump:
        _DUMP_LEFT[0] -= 1
        valid = (logical_indices >= 0).sum(dim=1)
        first_rows = logical_indices[: min(4, tokens)]
        contiguous_stride = k_cache.shape[1] * k_cache.shape[2] * k_cache.shape[3]
    mod.msa_sparse_attention(
        q,
        k_cache,
        v_cache,
        block_table,
        None,
        lod_q,
        None,
        lod_kv,
        topk_idx,
        out,
        topk_num,
        1,
        batch,
        heads,
        heads_kv,
        dim,
        dim,
        int(max_query_len),
        int(max_seq_len),
        block_table.shape[1],
        k_cache.shape[2],
        scale,
        -1 if int(max_query_len) <= 1 else int(max_query_len),
    )
    _sync("msa_sparse_attention")
    return out


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """把定宽行写进 ``[pages, page_size, 1, width]`` 的 cache，跳过 slot < 0 的行。"""

    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    if not cache.is_contiguous():
        # Q38 XPU PORT: 必须传整块连续的 kv_cache。key_cache / rope_position_cache
        # 是最后一维的切片（非连续），XPU 上对非连续视图 scatter 会先把整块
        # cache 物化一份（实测 peak 多出整块大小，本模型约 18 GiB）。
        raise ValueError("QSA cache rows must be written through a contiguous cache")
    slots = slot_mapping.to(torch.int64)
    page_size = cache.shape[1]
    limit = cache.shape[0] * page_size
    keep = (slots >= 0) & (slots < limit)
    safe = slots.clamp(0, limit - 1)
    flat = cache.reshape(limit, cache.shape[3])
    # 非法 slot 不能靠"夹到 0 再写回原值"来跳过：0 号槽位可能同时是某个合法行的
    # 目标，index_copy_ 遇到重复下标结果未定义，非法行会把合法行覆盖掉。
    # 改成把非法行整体重定向到某个合法行（"代表行"）的槽位、并写它的值——
    # 重复下标写的是同一个值，结果确定。形状全静态，可进 decode 图。
    lead = keep.to(torch.int32).argmax().reshape(1)
    lead_slot = safe.index_select(0, lead)
    lead_value = torch.where(
        keep.any(),
        rows.index_select(0, lead).to(flat.dtype),
        flat.index_select(0, lead_slot),
    )
    index = torch.where(keep, safe, lead_slot)
    values = torch.where(keep.unsqueeze(1), rows.to(flat.dtype), lead_value)
    flat.index_copy_(0, index, values)
    _sync("qsa_store_cache_rows")


def qsa_pack_ring_rows(
    keys: torch.Tensor,
    positions: torch.Tensor | None,
    width: int,
    rope_position_offset: int,
) -> torch.Tensor:
    """把 raw key 和 mrope 位置拼成 ring cache 的整行 ``[rows, width]``。

    ring cache 的行里前 ``key_width`` 个 bf16 是 key，第 ``rope_position_offset``
    起的 ``3 * 4`` 个 bf16 是 3 个 int64 的 mrope 位置。分开写要对 cache 的列切片
    做 scatter，而列切片是非连续视图——XPU 上会整块物化。拼成整行只写一次。
    """

    flat_keys = keys.reshape(keys.shape[0], -1)
    rows = torch.zeros(
        (flat_keys.shape[0], int(width)), dtype=keys.dtype, device=keys.device
    )
    rows[:, : flat_keys.shape[1]] = flat_keys
    if positions is not None:
        packed = positions.reshape(positions.shape[0], -1)[:, :3].contiguous()
        offset = int(rope_position_offset)
        rows[:, offset : offset + 3 * _BF16_PER_INT64] = packed.view(keys.dtype)
    return rows


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,
    raw_positions: torch.Tensor,
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
    *,
    state_cache_full: torch.Tensor | None = None,
    rope_position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对每行做"压缩组均值池化"，成员可能一半在 ring cache、一半在本步 raw 行里。

    逐字对照 ``ops/qsa.py::_compress_qsa_groups_kernel``：组内第 j 个成员的逻辑
    位置是 ``end_position - (CR - 1 - j)``，位置 >= 本次 chunk 起点的从 raw_keys 取，
    否则从 ring cache 的 ``position % ring_size`` 槽位取；非法行输出 0。
    """

    ratio = int(compress_ratio)
    if ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    rows = token_to_req.numel()
    head_dim = raw_keys.shape[2]
    device = raw_keys.device
    pooled = torch.empty((rows, 1, head_dim), dtype=raw_keys.dtype, device=device)
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=device)
    if not rows:
        return pooled, first_positions

    ring_size = compressor_state_cache.shape[1]
    num_requests = query_start_loc.shape[0] - 1
    num_blocks = compressor_state_cache.shape[0]
    request = _safe_request(token_to_req, num_requests)
    in_range = (token_to_req >= 0) & (token_to_req < num_requests)
    row_start = torch.where(
        in_range, query_start_loc.to(torch.int64).index_select(0, request), 0
    )
    row_end = torch.where(
        in_range, query_start_loc.to(torch.int64).index_select(0, request + 1), 0
    )
    index = torch.arange(rows, dtype=torch.int64, device=device)
    end_position = logical_positions.to(torch.int64)
    chunk_start = end_position - (index - row_start)
    state_block = torch.where(
        in_range,
        compressor_state_block_table.to(torch.int64)[:, 0].index_select(0, request),
        torch.full_like(request, -1),
    )
    valid_block = (state_block >= 0) & (state_block < num_blocks)
    valid_row = (
        in_range
        & (index >= row_start)
        & (index < row_end)
        & (end_position >= ratio - 1)
        & (compressed_slots.to(torch.int64) >= 0)
    )
    safe_block = state_block.clamp(min=0)

    raw_flat = raw_keys.reshape(-1, head_dim)

    # Q38 XPU PORT: ring cache 只能整块连续地 gather。
    # `key_cache` / `rope_position_cache` 都是 kv_cache 最后一维的切片（宽 256
    # 的行里取前 128 / 第 128 起的 12 个），非连续；XPU 上不管是 reshape 还是高
    # 级索引，遇到非连续张量都会先把整块 contiguous 一份——本模型 15 万块 ×
    # 512 行 × 256 宽 bf16 ≈ 18 GiB，直接 OOM。所以统一从整块连续的
    # `kv_cache`（[blocks, ring, 1, width]，transpose 只换了 size-1 维的 stride，
    # 仍然连续）gather，再在只有 rows 行的结果上切列。
    gather_base = state_cache_full if state_cache_full is not None else (
        compressor_state_cache
    )
    if not gather_base.is_contiguous():
        raise ValueError("QSA ring cache must be gathered through a contiguous cache")

    def _gather_from(base: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        if not base.is_contiguous():
            raise ValueError(
                "QSA ring cache must be gathered through a contiguous cache"
            )
        ring = base.shape[1]
        flat = base.reshape(-1, base.shape[3])
        flat_slot = safe_block * ring + (slot % ring)
        return flat.index_select(0, flat_slot.clamp(0, flat.shape[0] - 1))

    def _gather_ring(slot: torch.Tensor) -> torch.Tensor:
        return _gather_from(gather_base, slot)

    accumulator = torch.zeros((rows, head_dim), dtype=torch.float32, device=device)
    for group_offset in range(ratio):
        position = end_position - (ratio - 1 - group_offset)
        use_raw = position >= chunk_start
        raw_row = row_start + position - chunk_start
        raw_ok = (
            valid_row & use_raw & (raw_row >= row_start) & (raw_row < row_end)
            & (raw_row < rows)
        )
        state_ok = valid_row & ~use_raw & valid_block
        picked = torch.where(
            raw_ok.unsqueeze(1),
            raw_flat.index_select(0, raw_row.clamp(0, max(rows - 1, 0))).float(),
            torch.zeros((), dtype=torch.float32, device=device),
        )
        picked = torch.where(
            state_ok.unsqueeze(1),
            _gather_ring(position)[:, :head_dim].float(),
            picked,
        )
        accumulator += picked
    pooled.copy_((accumulator / ratio).reshape(rows, 1, head_dim).to(raw_keys.dtype))
    _sync("qsa_compress_groups_with_ratio/pool")

    first_position = end_position - ratio + 1
    if rope_cache is None:
        first_positions.copy_(
            torch.where(valid_row, first_position, torch.zeros_like(first_position))
            .unsqueeze(1)
            .expand(rows, 3)
        )
        return pooled, first_positions

    first_from_raw = first_position >= chunk_start
    raw_first = row_start + first_position - chunk_start
    raw_ok = (
        valid_row & first_from_raw & (raw_first >= row_start) & (raw_first < row_end)
        & (raw_first < rows)
    )
    raw_pos_flat = raw_positions.reshape(-1, raw_positions.shape[2])
    values = torch.where(
        raw_ok.unsqueeze(1),
        raw_pos_flat.index_select(0, raw_first.clamp(0, max(rows - 1, 0))),
        torch.zeros((), dtype=torch.int64, device=device),
    )
    state_ok = valid_row & ~first_from_raw & valid_block
    if state_cache_full is None:
        # rope 位置在独立的 rope cache 里（int64），不能从 key 的 ring cache 上取。
        state_rope = _gather_from(rope_cache, first_position)[:, :3].to(torch.int64)
    else:
        offset = int(rope_position_offset)
        packed = _gather_ring(first_position)[:, offset : offset + 3 * _BF16_PER_INT64]
        state_rope = packed.contiguous().view(torch.int64)[:, :3]
    values = torch.where(state_ok.unsqueeze(1), state_rope, values)
    first_positions.copy_(values)
    _sync("qsa_compress_groups_with_ratio/positions")
    return pooled, first_positions


__all__ = [
    "available",
    "expand_qsa_block_indices_cuda",
    "qsa_compress_groups_with_ratio",
    "qsa_mqa_paged",
    "qsa_pack_ring_rows",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
]
