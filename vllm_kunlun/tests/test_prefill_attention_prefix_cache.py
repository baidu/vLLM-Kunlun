#!/usr/bin/env python3
"""Standalone diagnostic for kunlun_ops.prefill_attention(is_prefix_cache=True).

The script builds a tiny paged KV cache with deterministic values, runs the
Kunlun prefix-cache prefill op, compares it with a PyTorch reference attention,
and reports whether changing unwritten tail slots changes the output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

import torch

try:
    import cocopod  # noqa: F401
    import kunlun_ops
except Exception as exc:  # pragma: no cover - diagnostic script
    print(f"[ERROR] failed to import kunlun_ops: {exc}", file=sys.stderr)
    raise


@dataclass
class CaseResult:
    name: str
    layout: str
    block_table_scale: int
    batch_size: int
    q_lens: list[int]
    kv_lens: list[int]
    block_size: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    dtype: str
    key_cache_contiguous: bool
    value_cache_contiguous: bool
    max_abs_err_clean: float
    max_rel_err_clean: float
    max_abs_err_dirty: float
    max_rel_err_dirty: float
    max_dirty_delta: float
    dirty_changes_output: bool
    clean_pass: bool
    dirty_pass: bool


def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("No XPU/CUDA device is available for kunlun_ops")


def make_lod(
    lengths: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    lod_cpu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device="cpu")
    lod_cpu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(dim=0)
    return lod_cpu, lod_cpu.to(device)


def expand_kv_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.shape[1] == num_heads:
        return x
    assert num_heads % x.shape[1] == 0
    return x.repeat_interleave(num_heads // x.shape[1], dim=1)


def reference_attention(
    q_flat: torch.Tensor,
    k_tokens: list[torch.Tensor],
    v_tokens: list[torch.Tensor],
    q_lens: list[int],
    scale: float,
) -> torch.Tensor:
    outputs = []
    q_offset = 0
    for q_len, k_req, v_req in zip(q_lens, k_tokens, v_tokens):
        q_req = q_flat[q_offset : q_offset + q_len].float()
        k_req = expand_kv_heads(k_req.float(), q_req.shape[1])
        v_req = expand_kv_heads(v_req.float(), q_req.shape[1])
        scores = torch.einsum("qhd,khd->hqk", q_req, k_req) * scale
        # Query tokens are assumed to be the last q_len tokens of the full KV
        # sequence. Causal mask only hides future query positions inside tail.
        kv_len = k_req.shape[0]
        q_positions = torch.arange(kv_len - q_len, kv_len, device=q_req.device)
        k_positions = torch.arange(kv_len, device=q_req.device)
        mask = k_positions[None, :] <= q_positions[:, None]
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out_req = torch.einsum("hqk,khd->qhd", probs, v_req)
        outputs.append(out_req)
        q_offset += q_len
    return torch.cat(outputs, dim=0).to(q_flat.dtype)


def make_cache_view(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if layout == "contiguous":
        key_cache = torch.zeros(
            (num_blocks, num_kv_heads, block_size, head_size),
            dtype=dtype,
            device=device,
        )
        value_cache = torch.zeros_like(key_cache)
        return key_cache, value_cache, None

    if layout == "hybrid":
        # Simulate vLLM hybrid attention+mamba layout rewrite:
        # the logical attention KV cache is viewed as (2, num_blocks, ...), but
        # _update_hybrid_attention_mamba_layout changes its stride so key blocks
        # are at even physical pages and value blocks are at odd physical pages.
        raw_kv_cache = torch.zeros(
            (2, num_blocks, block_size, num_kv_heads, head_size),
            dtype=dtype,
            device=device,
        )
        hidden_size = block_size * num_kv_heads * head_size
        raw_kv_cache.as_strided_(
            size=raw_kv_cache.shape,
            stride=(
                hidden_size,
                2 * hidden_size,
                num_kv_heads * head_size,
                head_size,
                1,
            ),
        )
        key_cache = raw_kv_cache[0].permute(0, 2, 1, 3)
        value_cache = raw_kv_cache[1].permute(0, 2, 1, 3)
        return key_cache, value_cache, raw_kv_cache

    raise ValueError(f"unknown layout: {layout}")


def build_cache(
    q_lens: list[int],
    kv_lens: list[int],
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    dirty_tail_value: float | None,
    layout: str,
    block_table_scale: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]
]:
    logical_blocks = sum(math.ceil(kv_len / block_size) for kv_len in kv_lens) + 4
    storage_blocks = logical_blocks * max(block_table_scale, 1) + 4
    key_cache, value_cache, raw_kv_cache = make_cache_view(
        storage_blocks, block_size, num_kv_heads, head_size, dtype, device, layout
    )
    block_tables = torch.full(
        (len(kv_lens), max(math.ceil(kv_len / block_size) for kv_len in kv_lens)),
        -1,
        dtype=torch.int32,
        device=device,
    )

    if dirty_tail_value is not None:
        key_cache.fill_(dirty_tail_value)
        value_cache.fill_(-dirty_tail_value)
        if raw_kv_cache is not None:
            raw_kv_cache.fill_(dirty_tail_value)

    k_tokens_per_req: list[torch.Tensor] = []
    v_tokens_per_req: list[torch.Tensor] = []
    next_logical_block = 0
    for req_idx, kv_len in enumerate(kv_lens):
        num_blocks = math.ceil(kv_len / block_size)
        logical_block_ids = torch.arange(
            next_logical_block, next_logical_block + num_blocks, dtype=torch.int32
        )
        physical_block_ids = logical_block_ids * block_table_scale
        block_tables[req_idx, :num_blocks] = physical_block_ids.to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(1000 + req_idx)
        k_req = torch.randn(
            (kv_len, num_kv_heads, head_size),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        v_req = torch.randn(
            (kv_len, num_kv_heads, head_size),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        for token_idx in range(kv_len):
            logical_block_id = next_logical_block + token_idx // block_size
            physical_block_id = logical_block_id * block_table_scale
            block_offset = token_idx % block_size
            key_cache[physical_block_id, :, block_offset, :] = k_req[token_idx]
            value_cache[physical_block_id, :, block_offset, :] = v_req[token_idx]
        k_tokens_per_req.append(k_req)
        v_tokens_per_req.append(v_req)
        next_logical_block += num_blocks

    return key_cache, value_cache, block_tables, k_tokens_per_req, v_tokens_per_req


def run_case(
    name: str,
    q_lens: list[int],
    kv_lens: list[int],
    block_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    atol: float,
    rtol: float,
    layout: str,
    block_table_scale: int,
) -> CaseResult:
    assert len(q_lens) == len(kv_lens)
    assert all(0 < q_len <= kv_len for q_len, kv_len in zip(q_lens, kv_lens))

    generator = torch.Generator(device=device)
    generator.manual_seed(20240519)
    q = torch.randn(
        (sum(q_lens), num_heads, head_size),
        generator=generator,
        dtype=dtype,
        device=device,
    )
    out_clean = torch.empty_like(q)
    out_dirty = torch.empty_like(q)

    qlod_cpu, qlod_xpu = make_lod(q_lens, device)
    kvlod_cpu, kvlod_xpu = make_lod(kv_lens, device)

    key_cache, value_cache, block_tables, k_tokens, v_tokens = build_cache(
        q_lens,
        kv_lens,
        block_size,
        num_kv_heads,
        head_size,
        dtype,
        device,
        None,
        layout,
        block_table_scale,
    )
    kunlun_ops.prefill_attention(
        q=q,
        k=key_cache,
        v=value_cache,
        out=out_clean,
        is_causal=True,
        is_prefix_cache=True,
        block_table=block_tables,
        context_qlen_lod_cpu=qlod_cpu,
        context_qlen_lod_xpu=qlod_xpu,
        context_kvlen_lod_cpu=kvlod_cpu,
        context_kvlen_lod_xpu=kvlod_xpu,
        alibi_slopes=None,
        softmax_lse=None,
        swa_left=-1,
        swa_right=-1,
        sink=None,
    )

    key_cache_dirty, value_cache_dirty, block_tables_dirty, _, _ = build_cache(
        q_lens,
        kv_lens,
        block_size,
        num_kv_heads,
        head_size,
        dtype,
        device,
        37.0,
        layout,
        block_table_scale,
    )
    kunlun_ops.prefill_attention(
        q=q,
        k=key_cache_dirty,
        v=value_cache_dirty,
        out=out_dirty,
        is_causal=True,
        is_prefix_cache=True,
        block_table=block_tables_dirty,
        context_qlen_lod_cpu=qlod_cpu,
        context_qlen_lod_xpu=qlod_xpu,
        context_kvlen_lod_cpu=kvlod_cpu,
        context_kvlen_lod_xpu=kvlod_xpu,
        alibi_slopes=None,
        softmax_lse=None,
        swa_left=-1,
        swa_right=-1,
        sink=None,
    )

    ref = reference_attention(
        q, k_tokens, v_tokens, q_lens, scale=1.0 / math.sqrt(head_size)
    )
    clean_diff = (out_clean.float() - ref.float()).abs()
    dirty_diff = (out_dirty.float() - ref.float()).abs()
    dirty_delta = (out_dirty.float() - out_clean.float()).abs()
    ref_abs = ref.float().abs().clamp_min(1e-6)

    max_abs_err_clean = clean_diff.max().item()
    max_rel_err_clean = (clean_diff / ref_abs).max().item()
    max_abs_err_dirty = dirty_diff.max().item()
    max_rel_err_dirty = (dirty_diff / ref_abs).max().item()
    max_dirty_delta = dirty_delta.max().item()

    return CaseResult(
        name=name,
        layout=layout,
        block_table_scale=block_table_scale,
        batch_size=len(q_lens),
        q_lens=q_lens,
        kv_lens=kv_lens,
        block_size=block_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=str(dtype),
        key_cache_contiguous=key_cache.is_contiguous(),
        value_cache_contiguous=value_cache.is_contiguous(),
        max_abs_err_clean=max_abs_err_clean,
        max_rel_err_clean=max_rel_err_clean,
        max_abs_err_dirty=max_abs_err_dirty,
        max_rel_err_dirty=max_rel_err_dirty,
        max_dirty_delta=max_dirty_delta,
        dirty_changes_output=max_dirty_delta > atol,
        clean_pass=max_abs_err_clean <= atol or max_rel_err_clean <= rtol,
        dirty_pass=max_abs_err_dirty <= atol or max_rel_err_dirty <= rtol,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--json", default="")
    parser.add_argument(
        "--layouts",
        default="contiguous,hybrid",
        help="comma-separated layouts: contiguous,hybrid",
    )
    args = parser.parse_args()

    device = get_device()
    dtype = getattr(torch, args.dtype)
    cases = [
        ("single_partial_tail", [3], [19]),
        ("single_exact_block", [4], [32]),
        ("multi_mixed_partial", [3, 5, 1], [19, 37, 48]),
        ("apc_like_short_query", [1, 1, 2], [33, 49, 65]),
    ]
    layout_scales = []
    requested_layouts = {
        layout.strip() for layout in args.layouts.split(",") if layout.strip()
    }
    if "contiguous" in requested_layouts:
        layout_scales.append(("contiguous", 1))
    if "hybrid" in requested_layouts:
        # Mirrors kunlun_attn.py non-contiguous path: block_tables * 2.
        layout_scales.append(("hybrid", 2))

    results = []
    print("# kunlun_ops.prefill_attention prefix-cache diagnostic")
    print(f"device={device} dtype={dtype} block_size={args.block_size}")
    print(
        f"heads={args.num_heads} kv_heads={args.num_kv_heads} head_size={args.head_size}"
    )
    for layout, block_table_scale in layout_scales:
        for case in cases:
            result = run_case(
                case[0],
                case[1],
                case[2],
                args.block_size,
                args.num_heads,
                args.num_kv_heads,
                args.head_size,
                dtype,
                device,
                args.atol,
                args.rtol,
                layout,
                block_table_scale,
            )
            results.append(result)
            status = (
                "PASS"
                if result.clean_pass
                and result.dirty_pass
                and not result.dirty_changes_output
                else "FAIL"
            )
            print(
                f"[{status}] {result.layout}/{result.name}: "
                f"scale={result.block_table_scale} "
                f"kc_contig={result.key_cache_contiguous} "
                f"vc_contig={result.value_cache_contiguous} "
                f"clean_abs={result.max_abs_err_clean:.6g} "
                f"clean_rel={result.max_rel_err_clean:.6g} "
                f"dirty_abs={result.max_abs_err_dirty:.6g} "
                f"dirty_rel={result.max_rel_err_dirty:.6g} "
                f"dirty_delta={result.max_dirty_delta:.6g} "
                f"q_lens={result.q_lens} kv_lens={result.kv_lens}"
            )

    report = {
        "device": str(device),
        "dtype": str(dtype),
        "block_size": args.block_size,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_size": args.head_size,
        "atol": args.atol,
        "rtol": args.rtol,
        "layout_scales": layout_scales,
        "results": [asdict(result) for result in results],
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote json report: {args.json}")

    failed = [
        r
        for r in results
        if not (r.clean_pass and r.dirty_pass and not r.dirty_changes_output)
    ]
    if failed:
        print(
            "\nFinding: prefix-cache output differs from reference or depends on dirty tail slots."
        )
        return 1
    print(
        "\nFinding: prefix-cache output matches reference and is insensitive to dirty tail slots."
    )
    return 0


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        print("[WARN] CUDA_VISIBLE_DEVICES is empty", file=sys.stderr)
    raise SystemExit(main())
