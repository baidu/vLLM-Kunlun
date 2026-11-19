from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

_KERNEL_DTYPE = torch.float16

# Only when rows exceeds this threshold is sgmv_*_sdnn cheaper than bgmv_*_cluster.
# Set it to a very large number to fall back to pure bgmv (for A/B comparison).
_SGMV_MIN_ROWS = int(os.environ.get("KUNLUN_EXPERT_LORA_SGMV_MIN_ROWS", "768"))

# Max number of rows fed to a single sgmv call. Beyond this, split by rows and
# pass the clamped segment lengths chunk by chunk.
_SGMV_CHUNK_ROWS = int(os.environ.get("KUNLUN_EXPERT_LORA_SGMV_CHUNK_ROWS", "16384"))

_SGMV_PAD = os.environ.get("KUNLUN_EXPERT_LORA_SGMV_PAD", "1") == "1"

# Padding stretches the last segment to ~CHUNK rows; this shape hits error 719
# on b>=49152 in practice, so reject it upfront.
_SGMV_PAD_MAX_CHUNK = 32768
if _SGMV_PAD and _SGMV_CHUNK_ROWS > _SGMV_PAD_MAX_CHUNK:
    raise ValueError(
        f"KUNLUN_EXPERT_LORA_SGMV_CHUNK_ROWS={_SGMV_CHUNK_ROWS} exceeds the safe "
        f"upper bound {_SGMV_PAD_MAX_CHUNK} of the padding path: padding stretches "
        "the last segment to ~CHUNK rows, and on b>=49152 this shape triggers error "
        "719 that kills the entire XPU context. "
        "Either lower CHUNK back to <=32768, or set KUNLUN_EXPERT_LORA_SGMV_PAD=0 to "
        "disable padding (once disabled, the result of the same row varies with batch size)."
    )

# Threshold that decides which path shrink takes.
_SGMV_SHRINK_MIN_ROWS = int(
    os.environ.get("KUNLUN_EXPERT_LORA_SGMV_SHRINK_MIN_ROWS", "4096")
)

# Scratch pool for the seam: fp16 copy of x, fp16 delta, rank buffer, and the full
# block used for padding. Allocations during cudagraph capture stay in the graph's
# private memory pool (observed to grow to 8~22GiB until OOM); after pooling, memory
# depends only on the largest shape. Pooled by (role, width, dtype, device) so that
# simultaneously live blocks of different roles never collide on the same block.
_POOL: dict[tuple, torch.Tensor] = {}
# Replaced old blocks: captured graphs still point at their addresses, so they
# cannot be freed and are never reused.
_POOL_RETIRED: list[torch.Tensor] = []
_pool_logger = logging.getLogger("vllm_kunlun")


def _pooled_rows(rows: int) -> int:
    """Round row count up to a power of two so the pool grows a bounded number of times."""
    if rows <= 1:
        return max(rows, 1)
    return 1 << (rows - 1).bit_length()


def _pooled(
    role: str, rows: int, width: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Get a persistent ``[rows, width]`` scratch; its content is leftover from the
    previous call, so the caller is responsible for initialization.

    Grows only, never shrinks; row count is rounded to a power of two. Each growth
    retires the old block and emits a warning.
    """
    key = (role, width, dtype, device)
    buf = _POOL.get(key)
    if buf is None or buf.shape[0] < rows:
        old_rows = 0 if buf is None else buf.shape[0]
        if buf is not None:
            _POOL_RETIRED.append(buf)
        buf = torch.empty((_pooled_rows(rows), width), dtype=dtype, device=device)
        _POOL[key] = buf
        _pool_logger.warning(
            "[moe_lora] scratch pool grew: role=%s width=%d rows %d -> %d "
            "(%d retired block(s) held to protect captured cudagraphs)",
            role,
            width,
            old_rows,
            buf.shape[0],
            len(_POOL_RETIRED),
        )
    return buf[:rows]


@dataclass
class ExpertLoRARows:
    """The (adapter, expert) index of each row in the sorted buffer."""

    composite: torch.Tensor  # int32 [rows]
    valid: torch.Tensor  # [rows, 1], same dtype as the shrink buffer
    lod: (
        torch.Tensor | None
    )  # int32 [E+1] prefix sum, used by sgmv; None means bgmv only
    seg_index: torch.Tensor | None  # int32 [E], per-segment weight index used by sgmv
    rows: int
    # bool [rows, 1], True = uninitialized rows at the buffer tail; None when not EP.
    garbage: torch.Tensor | None = None
    _chunks: object = (
        None  # cache of sgmv_chunks(), shared by the three seam calls in one forward
    )
    _invalid: object = None  # mask cache of zero_invalid_(), same as above

    def sanitize_input_(self, x: torch.Tensor) -> None:
        """Zero the uninitialized rows at the EP buffer tail: leftover data cast to fp16 may be inf/nan."""
        if self.garbage is not None:
            x.masked_fill_(self.garbage, 0)

    def zero_invalid_(self, buffer: torch.Tensor) -> None:
        """Zero the rows where ``valid == 0``.

        Uses ``masked_fill_`` instead of multiplication: uninitialized tail rows may
        be inf/nan, and 0 * inf = nan.
        """
        if self._invalid is None:
            self._invalid = self.valid == 0
        buffer.masked_fill_(self._invalid, 0)

    @property
    def use_sgmv(self) -> bool:
        return self.lod is not None and self.rows >= _SGMV_MIN_ROWS

    def sgmv_chunks(self) -> list[tuple[int, int, torch.Tensor]]:
        """Split by rows; each chunk yields ``(start, end, seq_len)``, with segment
        lengths clamped to chunk boundaries by the prefix sum."""
        if self._chunks is None:
            lo, hi = self.lod[:-1], self.lod[1:]
            self._chunks = [
                (
                    s,
                    min(s + _SGMV_CHUNK_ROWS, self.rows),
                    (
                        hi.clamp(s, min(s + _SGMV_CHUNK_ROWS, self.rows))
                        - lo.clamp(s, min(s + _SGMV_CHUNK_ROWS, self.rows))
                    ).contiguous(),
                )
                for s in range(0, self.rows, _SGMV_CHUNK_ROWS)
            ]
        return self._chunks


def build_rows(
    topk_ids: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    token_lora_indices: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    num_experts: int,
    single_lora_slot: int | None,
    local_expert_base: int | None = None,
) -> ExpertLoRARows:
    """Derive the per-row (adapter, expert) index from fused_moe's grouping metadata,
    with no host sync throughout (host reads during cudagraph capture are invalid and
    get baked into the graph, which is a hard requirement).
    """
    device = topk_ids.device
    flat_pos = sorted_tokens_idx.reshape(-1).to(torch.int64)
    rows = flat_pos.numel()

    tokens = token_lora_indices.reshape(-1)
    # cudagraph pads the batch to the captured size, so topk_ids may have more tokens
    # than punica's adapter indices; the difference is padded with -1 and takes the
    # zero-contribution path via valid==0.
    num_tokens = topk_ids.shape[0]
    top_k = rows // num_tokens if num_tokens else 0
    if num_tokens * top_k != rows:
        raise ValueError(
            f"topk_ids has {num_tokens} tokens and the sorted buffer has {rows} rows, "
            "which are not an integer multiple of each other -- MoE's grouping metadata "
            "is itself inconsistent, so per-row indices cannot be derived."
        )
    if tokens.numel() > num_tokens:
        raise ValueError(
            f"there are {tokens.numel()} per-token adapter indices, more than the "
            f"{num_tokens} tokens MoE sees -- meaning MoE is not seeing the same batch "
            "of tokens (sequence parallel / DP split?), in which case per-row indices "
            "would be globally misaligned."
        )
    if tokens.numel() < num_tokens:
        tokens = torch.cat(
            [tokens, tokens.new_full((num_tokens - tokens.numel(),), -1)]
        )
    flat_lora = tokens.view(-1, 1).expand(num_tokens, top_k).reshape(-1).to(torch.int32)
    flat_expert = topk_ids.reshape(-1).to(torch.int32)

    if local_expert_base is None:
        row_lora = torch.empty(rows, dtype=torch.int32, device=device)
        row_expert = torch.empty(rows, dtype=torch.int32, device=device)
        row_expert.scatter_(0, flat_pos, flat_expert)
        row_lora.scatter_(0, flat_pos, flat_lora)
        tail_mask = None
    else:
        # EP: the -1 slots cannot be used for scatter, so dump them all into one extra
        # cell at the buffer tail (the shape stays static throughout, avoiding a
        # data-dependent shape from boolean indexing being baked into the graph); the
        # squeezed-out rows (>= lod[-1]) are garbage. Both arrays are first filled with
        # legal values, since an out-of-range composite would read someone else's
        # weights or even crash.
        dump = flat_pos.masked_fill(flat_pos < 0, rows)
        row_expert = torch.zeros(rows + 1, dtype=torch.int32, device=device)
        row_expert.scatter_(0, dump, flat_expert - int(local_expert_base))
        row_lora = torch.full((rows + 1,), -1, dtype=torch.int32, device=device)
        row_lora.scatter_(0, dump, flat_lora)
        row_expert = row_expert[:rows]
        row_lora = row_lora[:rows]
        tail_mask = torch.arange(rows, device=device) < sorted_tokens_num_lod[-1]

    lod = seg_index = None
    if (
        single_lora_slot is not None
        and single_lora_slot >= 0
        and sorted_tokens_num_lod is not None
        and sorted_tokens_num_lod.numel() == num_experts + 1
    ):
        lod = sorted_tokens_num_lod.to(torch.int32)
        if tail_mask is not None:
            lod = lod.clone()
            lod[-1] = rows
        seg_index = (
            torch.arange(num_experts, dtype=torch.int32, device=device)
            + single_lora_slot * num_experts
        )

    composite = row_lora.clamp_min(0) * num_experts + row_expert
    keep_row = row_lora >= 0
    garbage = None
    if tail_mask is not None:
        keep_row = keep_row & tail_mask
        garbage = (~tail_mask).unsqueeze(1)
    valid = keep_row.to(_KERNEL_DTYPE).unsqueeze(1)
    return ExpertLoRARows(composite, valid, lod, seg_index, rows, garbage)


def rows_for_seam(
    context,
    topk_ids: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    sorted_tokens_num_lod: torch.Tensor,
    num_experts: int,
    local_expert_base: int | None = None,
) -> ExpertLoRARows | None:
    """Seam entry point for ``fused_moe`` / ``fused_moe_ep``; returns None when there is no LoRA to compute."""
    if context is None:
        return None
    punica = getattr(context, "punica_wrapper", None)
    if punica is None or getattr(punica, "no_lora", False):
        return None
    return build_rows(
        topk_ids,
        sorted_tokens_idx,
        context.token_lora_indices,
        sorted_tokens_num_lod,
        num_experts,
        context.single_lora_slot,
        local_expert_base,
    )


def _flatten_experts(weights: torch.Tensor) -> torch.Tensor:
    """``[max_loras, E, a, b]`` -> ``[max_loras * E, a, b]``, to match the composite index."""
    if weights.dim() == 4:
        return weights.reshape(-1, weights.shape[2], weights.shape[3])
    return weights


def _sgmv_shrink(ops, x, lora_a, buffer, rows) -> None:
    """Call ``sgmv_shrink_sdnn`` chunk by chunk; pad the last chunk to
    ``_SGMV_CHUNK_ROWS`` (the kernel result varies with the row count of a single
    call, and padding makes each row's output depend only on its own input).
    """
    chunk = _SGMV_CHUNK_ROWS
    for start, end, seq_len in rows.sgmv_chunks():
        n = end - start
        if n == chunk or not _SGMV_PAD:
            ops.sgmv_shrink_sdnn(
                x[start:end],
                lora_a,
                seq_len,
                rows.seg_index,
                buffer[start:end],
                1.0,
            )
            continue
        pad_x = _pooled("padx", chunk, x.shape[1], x.dtype, x.device)
        pad_x[:n].copy_(x[start:end])
        pad_x[n:].zero_()
        pad_buffer = _pooled(
            "padbuf", chunk, buffer.shape[1], buffer.dtype, buffer.device
        )
        pad_buffer.zero_()
        # The padded rows attach to the last segment, computed with that segment's
        # weights, and discarded afterward.
        pad_seq = seq_len.clone()
        pad_seq[-1] += chunk - n
        ops.sgmv_shrink_sdnn(pad_x, lora_a, pad_seq, rows.seg_index, pad_buffer, 1.0)
        buffer[start:end].copy_(pad_buffer[:n])


def _shrink_expand(
    scratch: torch.Tensor,
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    rows: ExpertLoRARows,
    slice_offset: int,
) -> None:
    """shrink + expand for one slice; the result is accumulated into the fp16 ``scratch``."""
    lora_a = _flatten_experts(lora_a)
    lora_b = _flatten_experts(lora_b)
    if lora_a.dtype != _KERNEL_DTYPE or lora_b.dtype != _KERNEL_DTYPE:
        raise TypeError(
            "expert LoRA stacked weights must be fp16: "
            f"got lora_a={lora_a.dtype} lora_b={lora_b.dtype}. "
            "The sgmv/bgmv kernels reject bf16, and the data movement of converting "
            "each time is unacceptable, so weights are converted to fp16 at set_lora."
        )
    rank = lora_a.shape[1]
    buffer = _pooled("rank", rows.rows, rank, _KERNEL_DTYPE, x.device)
    buffer.zero_()
    ops = torch.ops.xspeedgate_ops
    # scale=1.0: alpha/r has already been folded into lora_b in LoRALayerWeights.optimize().
    if rows.use_sgmv:
        if rows.rows >= _SGMV_SHRINK_MIN_ROWS:
            _sgmv_shrink(ops, x, lora_a, buffer, rows)
        else:
            ops.bgmv_shrink_cluster(x, lora_a, rows.composite, buffer, 1.0)
        rows.zero_invalid_(buffer)
        for start, end, seq_len in rows.sgmv_chunks():
            ops.sgmv_expand_sdnn(
                buffer[start:end],
                lora_b,
                seq_len,
                rows.seg_index,
                scratch[start:end],
                slice_offset,
            )
    else:
        ops.bgmv_shrink_cluster(x, lora_a, rows.composite, buffer, 1.0)
        rows.zero_invalid_(buffer)
        ops.bgmv_expand_cluster(buffer, lora_b, rows.composite, scratch, slice_offset)


def apply_seam(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_a_slices,
    lora_b_slices,
    rows: ExpertLoRARows,
) -> None:
    """Accumulate the expert LoRA of all slices into ``y`` on one moe_fc seam.

    ``y``/``x`` are 2D views in sorted order; ``y`` is accumulated in place. The whole
    seam shares one fp16 scratch and adds back to bf16 once at the end. EP tail garbage
    rows are guaranteed zero contribution by sanitize_input_ / valid==0.
    """
    n = rows.rows
    if n == 0:
        # The behavior of the six kernels on 0 rows is unverified, and a zero-length
        # slice on XPU raises, so return directly.
        return
    y2 = y.view(-1, y.shape[-1])[:n]
    xv = x.view(-1, x.shape[-1])[:n]
    x2 = _pooled("x", n, xv.shape[1], _KERNEL_DTYPE, xv.device)
    x2.copy_(xv)  # bf16 -> fp16: none of the six kernels accept bf16
    rows.sanitize_input_(
        x2
    )  # EP: the tail is uninitialized memory, may be inf/nan after conversion
    direct = y2.dtype == _KERNEL_DTYPE
    if direct:
        scratch = y2
    else:
        scratch = _pooled("delta", n, y2.shape[1], _KERNEL_DTYPE, y2.device)
        scratch.zero_()  # expand has accumulate semantics
    offset = 0
    for lora_a, lora_b in zip(lora_a_slices, lora_b_slices):
        _shrink_expand(scratch, x2, lora_a, lora_b, rows, offset)
        offset += _flatten_experts(lora_b).shape[1]
    if offset != y2.shape[1]:
        raise ValueError(
            f"slice widths sum to {offset}, which does not match target width {y2.shape[1]}"
        )
    if not direct:
        y2.add_(scratch.to(y2.dtype))
