"""Unit tests for the DeepSeek-V4 compressor native kernels.

The references are transcribed from the upstream Triton kernel
``_fused_kv_compress_norm_rope_insert_sparse_attn`` in
``vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache``, which is the
semantics the Kunlun adapter has to reproduce:

* ``kunlun_ops.fused_kv_compress_gather`` -- for every token sitting on a
  compression boundary, gather ``(1 + overlap) * compress_ratio`` state rows
  ending at that position, softmax the score half across the window
  (per column), and weight-sum the kv half.
* ``kunlun_ops.dpsk_v4_norm_rope_gptj`` -- RMSNorm over the whole head followed
  by GPT-J (interleaved-pair) RoPE on the trailing ``rope_head_dim`` elements,
  in place.

The last test covers the adapter dispatcher that wires those two kernels
together, asserting the native path and the torch fallback land on the same
paged KV cache.

Reference math runs on CPU in float64 so the assertions measure kernel error
rather than reduction-order noise in a second float32 implementation.
"""
import types

import pytest
import torch

kunlun_ops = pytest.importorskip("kunlun_ops")

_REQUIRED = ("fused_kv_compress_gather", "dpsk_v4_norm_rope_gptj")
_MISSING = [name for name in _REQUIRED if not callable(getattr(kunlun_ops, name, None))]

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Kunlun XPU device is required"
    ),
    pytest.mark.skipif(
        bool(_MISSING), reason=f"kunlun_ops is missing {_MISSING}"
    ),
]

DEVICE = "cuda:0"
ROPE_HEAD_DIM = 64
EPS = 1e-6

# (head_size, compress_ratio, overlap) -- the two shapes DeepSeek-V4-Flash uses:
# the MLA compressor (head 512, ratio 4, overlapped window) and the indexer
# compressor (head 128, ratio 128, no overlap).
LAYER_SHAPES = [
    pytest.param(512, 4, 1, id="mla-head512-ratio4-overlap"),
    pytest.param(128, 128, 0, id="indexer-head128-ratio128"),
]


def _make_cos_sin_cache(max_pos, rope_head_dim):
    """vLLM layout: ``[max_pos, rope_head_dim]`` with cos in the first half."""
    half = rope_head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float64) / half))
    angles = torch.arange(max_pos, dtype=torch.float64).unsqueeze(1) * inv_freq
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


def _interleave_cos_sin(cos_sin_cache):
    """Native ``freqs_cis`` layout: ``[cos0, sin0, cos1, sin1, ...]``."""
    max_pos, rope_head_dim = cos_sin_cache.shape
    half = rope_head_dim // 2
    return (
        torch.stack(
            (cos_sin_cache[:, :half], cos_sin_cache[:, half:]), dim=-1
        )
        .reshape(max_pos, rope_head_dim)
        .contiguous()
    )


def _reference_gather(
    state_cache, token_to_req_indices, positions, slot_mapping, block_table,
    block_size, head_size, state_width, compress_ratio, overlap,
):
    """Loop-faithful port of the Triton gather; rows that early-exit stay zero."""
    num_tokens = positions.shape[0]
    window = (1 + overlap) * compress_ratio
    out = torch.zeros((num_tokens, head_size), dtype=torch.float64)
    for token in range(num_tokens):
        if int(slot_mapping[token]) < 0:
            continue
        position = int(positions[token])
        if (position + 1) % compress_ratio != 0:
            continue
        req = int(token_to_req_indices[token])
        kv_rows = torch.zeros((window, head_size), dtype=torch.float64)
        score_rows = torch.full((window, head_size), float("-inf"), dtype=torch.float64)
        for slot in range(window):
            pos = position - window + 1 + slot
            if pos < 0:
                continue
            head_offset = head_size if slot >= compress_ratio else 0
            block = int(block_table[req, pos // block_size])
            row = state_cache[block, pos % block_size].to(torch.float64)
            kv_rows[slot] = row[head_offset:head_offset + head_size]
            score_rows[slot] = row[
                state_width + head_offset:state_width + head_offset + head_size
            ]
        weights = torch.softmax(score_rows, dim=0)
        out[token] = (kv_rows * weights).sum(dim=0)
    return out


def _reference_norm_rope(kv, norm_weight, positions, cos_sin_cache, rope_head_dim, eps):
    """RMSNorm over the full head, then GPT-J RoPE on the trailing rope dims."""
    x = kv.to(torch.float64)
    out = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
    out = out * norm_weight.to(torch.float64)
    nope = x.shape[-1] - rope_head_dim
    half = rope_head_dim // 2
    cos_sin = cos_sin_cache.to(torch.float64)[positions.cpu().long()]
    cos, sin = cos_sin[:, :half], cos_sin[:, half:]
    even = out[:, nope::2].clone()
    odd = out[:, nope + 1::2].clone()
    out[:, nope::2] = even * cos - odd * sin
    out[:, nope + 1::2] = odd * cos + even * sin
    return out


# Token layout shared by every gather test, expressed in units of
# ``compress_ratio`` so both layer shapes exercise the same situations:
# clipped window at sequence start, a mid-sequence boundary, a boundary on a
# second request with a different block table, a non-boundary position, and a
# padding row.
_TOKENS = [
    # (req, position_in_ratios, position_delta, state_slot)
    (0, 1, -1, 0),    # boundary, window runs off the start of the sequence
    (0, 4, -1, 1),    # boundary, full window
    (1, 2, -1, 5),    # boundary on the other request
    (1, 2, 0, 6),     # NOT a boundary -> must early-exit
    (1, 3, -1, -1),   # padding slot -> must early-exit
]
BLOCK_SIZE = 128
NUM_BLOCKS = 16
BLOCK_TABLE_ROWS = [[7, 3, 11, 5, 1, 9], [2, 13, 6, 15, 4, 8]]


def _make_case(head_size, compress_ratio, overlap, seed=1234):
    state_width = (1 + overlap) * head_size
    generator = torch.Generator().manual_seed(seed)
    state_cache = torch.randn(
        (NUM_BLOCKS, BLOCK_SIZE, 2 * state_width),
        generator=generator,
        dtype=torch.float32,
    )
    positions = torch.tensor(
        [n * compress_ratio + d for _, n, d, _ in _TOKENS], dtype=torch.int64
    )
    return {
        "state_cache": state_cache,
        "token_to_req_indices": torch.tensor(
            [t[0] for t in _TOKENS], dtype=torch.int32
        ),
        "positions": positions,
        "slot_mapping": torch.tensor([t[3] for t in _TOKENS], dtype=torch.int64),
        "block_table": torch.tensor(BLOCK_TABLE_ROWS, dtype=torch.int32),
        "head_size": head_size,
        "state_width": state_width,
        "compress_ratio": compress_ratio,
        "overlap": overlap,
    }


def _run_gather(case):
    out = torch.zeros(
        (case["positions"].shape[0], case["head_size"]),
        dtype=torch.float32,
        device=DEVICE,
    )
    kunlun_ops.fused_kv_compress_gather(
        case["state_cache"].to(DEVICE),
        case["token_to_req_indices"].to(DEVICE),
        case["positions"].to(DEVICE),
        case["slot_mapping"].to(DEVICE),
        case["block_table"].to(DEVICE),
        out,
        BLOCK_SIZE,
        case["head_size"],
        case["state_width"],
        case["compress_ratio"],
        case["overlap"],
    )
    torch.cuda.synchronize()
    return out


def _reference_gather_from_case(case):
    return _reference_gather(
        case["state_cache"], case["token_to_req_indices"], case["positions"],
        case["slot_mapping"], case["block_table"], BLOCK_SIZE,
        case["head_size"], case["state_width"], case["compress_ratio"],
        case["overlap"],
    )


def _boundary_mask(case):
    return torch.tensor(
        [
            slot >= 0 and (int(pos) + 1) % case["compress_ratio"] == 0
            for pos, slot in zip(case["positions"], case["slot_mapping"])
        ]
    )


@pytest.mark.parametrize("head_size, compress_ratio, overlap", LAYER_SHAPES)
def test_compress_gather_matches_reference(head_size, compress_ratio, overlap):
    case = _make_case(head_size, compress_ratio, overlap)
    got = _run_gather(case).cpu().to(torch.float64)
    torch.testing.assert_close(
        got, _reference_gather_from_case(case), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("head_size, compress_ratio, overlap", LAYER_SHAPES)
def test_compress_gather_early_exits_on_padding_and_non_boundary(
    head_size, compress_ratio, overlap
):
    case = _make_case(head_size, compress_ratio, overlap)
    got = _run_gather(case).cpu()
    touched = got.abs().sum(dim=-1) > 0
    torch.testing.assert_close(touched, _boundary_mask(case))


@pytest.mark.parametrize("head_size, compress_ratio, overlap", LAYER_SHAPES)
def test_compress_gather_leaves_preexisting_output_untouched(
    head_size, compress_ratio, overlap
):
    """Skipped rows must not be written, so the caller's buffer survives."""
    case = _make_case(head_size, compress_ratio, overlap)
    num_tokens = case["positions"].shape[0]
    sentinel = torch.full(
        (num_tokens, head_size), -7.5, dtype=torch.float32, device=DEVICE
    )
    kunlun_ops.fused_kv_compress_gather(
        case["state_cache"].to(DEVICE),
        case["token_to_req_indices"].to(DEVICE),
        case["positions"].to(DEVICE),
        case["slot_mapping"].to(DEVICE),
        case["block_table"].to(DEVICE),
        sentinel,
        BLOCK_SIZE,
        head_size,
        case["state_width"],
        compress_ratio,
        overlap,
    )
    torch.cuda.synchronize()
    skipped = ~_boundary_mask(case)
    assert bool((sentinel.cpu()[skipped] == -7.5).all())


@pytest.mark.parametrize("head_size", [512, 128])
def test_norm_rope_gptj_matches_reference(head_size):
    num_tokens, max_pos = 6, 640
    generator = torch.Generator().manual_seed(7)
    kv = torch.randn((num_tokens, head_size), generator=generator, dtype=torch.float32)
    norm_weight = torch.randn(
        (head_size,), generator=generator, dtype=torch.float32
    ).abs() + 0.5
    positions = torch.tensor([0, 1, 4, 63, 128, 639], dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_HEAD_DIM)

    got = kv.to(DEVICE)
    status = kunlun_ops.dpsk_v4_norm_rope_gptj(
        got,
        norm_weight.to(DEVICE),
        positions.to(DEVICE),
        _interleave_cos_sin(cos_sin_cache).to(torch.float32).to(DEVICE),
        mode=2,
        compress_ratio=0,
        eps=EPS,
    )
    torch.cuda.synchronize()
    assert status == 0

    expected = _reference_norm_rope(
        kv, norm_weight, positions, cos_sin_cache, ROPE_HEAD_DIM, EPS
    )
    torch.testing.assert_close(
        got.cpu().to(torch.float64), expected, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("head_size, compress_ratio, overlap", LAYER_SHAPES)
def test_compressor_pipeline_matches_reference(head_size, compress_ratio, overlap):
    """gather -> norm_rope in exactly the order the adapter will call them.

    The RoPE position is the *compressed* position
    ``(position // compress_ratio) * compress_ratio``, matching the upstream
    Triton kernel.  Rows the gather skipped stay zero and survive RMSNorm as
    zero, so the comparison covers every row.
    """
    max_pos = 640
    case = _make_case(head_size, compress_ratio, overlap, seed=99)
    norm_weight = (
        torch.randn(
            (head_size,), generator=torch.Generator().manual_seed(5),
            dtype=torch.float32,
        ).abs()
        + 0.5
    )
    cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_HEAD_DIM)
    compressed_positions = (
        case["positions"] // compress_ratio
    ) * compress_ratio

    got = _run_gather(case)
    status = kunlun_ops.dpsk_v4_norm_rope_gptj(
        got,
        norm_weight.to(DEVICE),
        compressed_positions.to(DEVICE),
        _interleave_cos_sin(cos_sin_cache).to(torch.float32).to(DEVICE),
        mode=2,
        compress_ratio=0,
        eps=EPS,
    )
    torch.cuda.synchronize()
    assert status == 0

    expected = _reference_norm_rope(
        _reference_gather_from_case(case), norm_weight, compressed_positions,
        cos_sin_cache, ROPE_HEAD_DIM, EPS,
    )
    torch.testing.assert_close(
        got.cpu().to(torch.float64), expected, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("head_size", [512, 128])
def test_norm_rope_gptj_rotates_only_the_rope_tail(head_size):
    """The leading nope dims must come out as plain RMSNorm, unrotated."""
    num_tokens, max_pos = 4, 256
    generator = torch.Generator().manual_seed(11)
    kv = torch.randn((num_tokens, head_size), generator=generator, dtype=torch.float32)
    norm_weight = torch.ones((head_size,), dtype=torch.float32)
    positions = torch.tensor([3, 7, 11, 255], dtype=torch.int64)

    got = kv.to(DEVICE)
    kunlun_ops.dpsk_v4_norm_rope_gptj(
        got,
        norm_weight.to(DEVICE),
        positions.to(DEVICE),
        _interleave_cos_sin(_make_cos_sin_cache(max_pos, ROPE_HEAD_DIM))
        .to(torch.float32)
        .to(DEVICE),
        mode=2,
        compress_ratio=0,
        eps=EPS,
    )
    torch.cuda.synchronize()

    x = kv.to(torch.float64)
    normed = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS)
    nope = head_size - ROPE_HEAD_DIM
    torch.testing.assert_close(
        got.cpu().to(torch.float64)[:, :nope], normed[:, :nope],
        rtol=1e-4, atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Adapter-level: the dispatcher that wires the two kernels together must agree
# with the torch fallback it replaces.
# ---------------------------------------------------------------------------
_KV_SLOTS = [1, 2, 3, 4, 5]
_NATIVE_FLAG = "KUNLUN_DSV4_COMPRESSOR_COMPRESS_NATIVE"


def _reference_store(case, norm_weight, cos_sin_cache):
    """Expected paged KV cache: normed rows at their slots, everything else zero."""
    head_size = case["head_size"]
    compress_ratio = case["compress_ratio"]
    normed = _reference_norm_rope(
        _reference_gather_from_case(case),
        norm_weight,
        (case["positions"] // compress_ratio) * compress_ratio,
        cos_sin_cache,
        ROPE_HEAD_DIM,
        EPS,
    )
    out = torch.zeros((NUM_BLOCKS, BLOCK_SIZE, head_size), dtype=torch.float64)
    for token, keep in enumerate(_boundary_mask(case).tolist()):
        if not keep:
            continue
        slot = _KV_SLOTS[token]
        out[slot // BLOCK_SIZE, slot % BLOCK_SIZE] = normed[token]
    return out


def _run_installed_dispatcher(case, norm_weight, cos_sin_cache, native, monkeypatch):
    """Install the adapter's dispatcher with the native flag forced, then call it."""
    from vllm_kunlun.ops.attention import compressor

    monkeypatch.setenv(_NATIVE_FLAG, "1" if native else "0")
    emitted = []
    monkeypatch.setattr(
        compressor.WarningOnce,
        "emit",
        staticmethod(lambda key, *a, **kw: emitted.append(key)),
    )

    fcqc = types.ModuleType("fake_fused_compress_quant_cache")
    compressor._install_compress_norm_rope_store_triton(fcqc)

    num_tokens = case["positions"].shape[0]
    kv_cache = torch.zeros(
        (NUM_BLOCKS, BLOCK_SIZE, case["head_size"]),
        dtype=torch.float32,
        device=DEVICE,
    )
    fcqc.compress_norm_rope_store_triton(
        case["state_cache"].to(DEVICE),
        num_tokens,
        case["token_to_req_indices"].to(DEVICE),
        case["positions"].to(DEVICE),
        case["slot_mapping"].to(DEVICE),
        case["block_table"].to(DEVICE),
        BLOCK_SIZE,
        case["state_width"],
        cos_sin_cache.to(torch.float32).to(DEVICE),
        kv_cache,
        types.SimpleNamespace(
            slot_mapping=torch.tensor(_KV_SLOTS, dtype=torch.int64, device=DEVICE)
        ),
        None,
        case["head_size"],
        ROPE_HEAD_DIM,
        case["compress_ratio"],
        case["overlap"],
        False,
        norm_weight.to(DEVICE),
        EPS,
        0,
        0,
        0,
    )
    torch.cuda.synchronize()
    return kv_cache.cpu().to(torch.float64), emitted


@pytest.mark.parametrize("head_size, compress_ratio, overlap", LAYER_SHAPES)
def test_dispatcher_native_and_fallback_agree(
    head_size, compress_ratio, overlap, monkeypatch
):
    pytest.importorskip("vllm_kunlun")
    case = _make_case(head_size, compress_ratio, overlap, seed=4242)
    norm_weight = (
        torch.randn(
            (head_size,), generator=torch.Generator().manual_seed(8),
            dtype=torch.float32,
        ).abs()
        + 0.5
    )
    cos_sin_cache = _make_cos_sin_cache(640, ROPE_HEAD_DIM)
    expected = _reference_store(case, norm_weight, cos_sin_cache)

    got_native, emitted = _run_installed_dispatcher(
        case, norm_weight, cos_sin_cache, True, monkeypatch
    )
    # A silent fallback would still match the reference, so assert the native
    # path was actually taken.
    assert emitted == [], emitted
    torch.testing.assert_close(got_native, expected, rtol=1e-4, atol=1e-5)

    got_fallback, _ = _run_installed_dispatcher(
        case, norm_weight, cos_sin_cache, False, monkeypatch
    )
    torch.testing.assert_close(got_fallback, expected, rtol=1e-4, atol=1e-5)
