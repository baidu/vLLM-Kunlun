# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun DSpark module contracts and selective Mamba state initialization."""

from types import SimpleNamespace as NS

import pytest
import torch

_CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not _CUDA, reason="requires Kunlun XPU")
if _CUDA:
    from vllm_kunlun.ops.layernorm import rms_norm
    from vllm_kunlun.v1.worker.gpu.spec_decode.dflash.speculator import (
        prepare_dflash_inputs,
    )
    from vllm_kunlun.v1.worker.gpu.spec_decode.dspark import speculator, utils
    from vllm_kunlun.v1.worker.gpu.spec_decode.dspark_buffers import lookup_kv_lod
    from vllm_kunlun.v1.worker.mamba_utils import MambaStateZeroer


def tensor(data, dtype=torch.int32):
    return torch.tensor(data, dtype=dtype, device="cuda")


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("anchor", [False, True])
def test_input_preparation_packs_requests_and_all_kv_groups(groups, anchor):
    capacity, queries = 4, 3
    steps = queries if anchor else queries - 1
    buffers = NS(
        input_ids=tensor([-77] * 12),
        positions=tensor([-77] * 12, torch.int64),
        query_start_loc=tensor([-77] * 5),
        seq_lens=tensor([-77] * 4),
    )
    batch = NS(
        num_reqs=2,
        num_tokens=5,
        positions=tensor([10, 11, 12, 20, 21], torch.int64),
        query_start_loc=tensor([0, 3, 5]),
        idx_mapping=tensor([1, 0]),
    )
    qslots = torch.full((groups, 12), -77, dtype=torch.int64, device="cuda")
    cslots = torch.full((groups, 5), -77, dtype=torch.int64, device="cuda")
    cpos = tensor([-77] * 5, torch.int64)
    sidx = tensor([-77] * (capacity * steps), torch.int64)
    spos = torch.empty_like(sidx)
    smap = tensor([-77] * (capacity * steps))
    for group in range(groups):
        prepare_dflash_inputs(
            buffers,
            qslots[group],
            cpos,
            cslots[group],
            sidx,
            spos,
            smap,
            batch,
            tensor([1, 0]),
            tensor([1, 0]),
            tensor([100, 200], torch.int64),
            tensor([300, 400]),
            torch.full((2, 16), group + 2, dtype=torch.int32, device="cuda"),
            4,
            99,
            queries,
            steps,
            capacity,
            12,
            64,
            sample_from_anchor=anchor,
        )
    assert buffers.input_ids.cpu().tolist() == [200, 99, 99, 300, 99, 99] + [99] * 6
    assert buffers.positions.cpu().tolist() == [12, 13, 14, 22, 23, 24] + [0] * 6
    assert buffers.query_start_loc.cpu().tolist() == [0, 3, 6, 6, 6]
    assert buffers.seq_lens.cpu().tolist() == [15, 25, 0, 0]
    assert lookup_kv_lod(buffers.seq_lens).cpu().tolist() == [0, 15, 40, 40, 40]
    assert cpos.cpu().tolist() == [10, 11, 0, 20, 21]
    for group in range(groups):
        base = (group + 2) * 4
        assert (
            qslots[group].cpu().tolist()
            == [base + p % 4 for p in [12, 13, 14, 22, 23, 24]] + [-1] * 6
        )
        assert cslots[group].cpu().tolist() == [base + 2, base + 3, -1, base, base + 1]
    selected = [0, 1, 2, 3, 4, 5] if anchor else [1, 2, 4, 5]
    positions = [13, 14, 15, 23, 24, 25] if anchor else [13, 14, 23, 24]
    assert sidx.cpu().tolist() == selected + [0] * (2 * steps)
    assert spos.cpu().tolist() == positions + [0] * (2 * steps)
    assert smap.cpu().tolist() == [1] * steps + [0] * steps + [-1] * (2 * steps)


@pytest.mark.parametrize(
    "shape, weight_shape",
    [
        ((7, 128), (128,)),
        ((3, 7, 128), (3, 128)),
        ((3, 600, 128), (3, 128)),
        ((3, 5, 7, 128), (3, 128)),
        ((2, 7, 128), (128,)),
    ],
)
def test_rms_norm_native_and_generic_shape_contracts(shape, weight_shape):
    torch.manual_seed(31)
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    weight = torch.randn(weight_shape, device="cuda", dtype=torch.float16)
    out = torch.empty_like(x)
    rms_norm(out, x, weight, 1e-6)
    w = weight.float().cpu()
    if len(weight_shape) == 2:
        w = w.reshape(shape[0], *([1] * (len(shape) - 2)), shape[-1])
    xf = x.float().cpu()
    expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * w).half()
    torch.testing.assert_close(out.cpu(), expected, rtol=0.002, atol=0.002)


@pytest.mark.parametrize("inner", [False, True])
def test_loader_shares_target_head_and_preserves_upstream_model(monkeypatch, inner):
    monkeypatch.delenv("VLLM_KUNLUN_DSPARK_TOPK", raising=False)
    head = torch.nn.Linear(4, 8, bias=False)
    embedding = object()
    language = NS(lm_head=head, model=NS(embed_tokens=embedding))
    target = NS(get_language_model=lambda: language) if inner else language
    draft = NS(
        has_own_lm_head=False,
        lm_head=torch.nn.Linear(4, 8, bias=False),
        model=NS(embed_tokens=embedding),
    )
    monkeypatch.setattr(utils, "_original_load_dspark_model", lambda *args: draft)
    config = NS(
        model_config=NS(get_vocab_size=lambda: 8),
        speculative_config=NS(
            draft_model_config=NS(
                hf_config=NS(draft_vocab_size=8), get_vocab_size=lambda: 8
            )
        ),
    )
    assert utils.load_dspark_model(target, config) is draft
    assert draft.lm_head is head


@pytest.mark.parametrize("topk", ["0", "-1"])
def test_nonpositive_topk_is_rejected(monkeypatch, topk):
    monkeypatch.setenv("VLLM_KUNLUN_DSPARK_TOPK", topk)
    with pytest.raises(ValueError, match="positive"):
        utils.configured_topk()


def test_markov_native_sampling_matches_expected_tokens(monkeypatch):
    monkeypatch.setenv("VLLM_KUNLUN_DSPARK_TOPK", "4")
    monkeypatch.setenv("VLLM_KUNLUN_DSPARK_TP_LOCAL_TOPK", "0")
    logits = torch.zeros((6, 16), dtype=torch.float16, device="cuda")
    for row in range(6):
        logits[row, row + 2] = 5
    weight = torch.zeros((16, 256), dtype=torch.float16, device="cuda")
    model = NS(
        config=NS(draft_vocab_size=16),
        compute_draft_logits=lambda _: logits,
        draft_id_to_target_id=None,
        logits_processor=NS(scale=1.0),
        model=NS(
            markov_head=NS(_kunlun_markov_w1_full=weight, _kunlun_markov_w2_full=weight)
        ),
    )
    state = NS(
        draft_logits=None,
        num_speculative_steps=3,
        sample_indices=tensor(list(range(6)), torch.int64),
        model=model,
        input_buffers=NS(input_ids=tensor([1, 0, 0, 2, 0, 0])),
        _anchor_idx=tensor([0, 3], torch.int64),
        draft_tokens=torch.full((4, 3), -1, dtype=torch.int64, device="cuda"),
    )
    speculator._sample_sequential(state, 2, torch.zeros((6, 4), device="cuda"))
    assert state.draft_tokens.cpu().tolist() == [
        [2, 3, 4],
        [5, 6, 7],
        [-1] * 3,
        [-1] * 3,
    ]


def test_default_markov_path_remains_upstream(monkeypatch):
    monkeypatch.delenv("VLLM_KUNLUN_DSPARK_TOPK", raising=False)
    calls = []
    monkeypatch.setattr(
        speculator, "_original_sample_sequential", lambda *args: calls.append(args)
    )
    state, hidden = NS(draft_logits=None), object()
    speculator._sample_sequential(state, 2, hidden)
    assert calls == [(state, 2, hidden)]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_blocks", [9, 41])
@pytest.mark.parametrize("tail", [37, 128])
def test_only_selected_state_bytes_are_zeroed(dtype, num_blocks, tail):
    expected = torch.full((num_blocks, 8 * tail + 19), 0.375, dtype=dtype)
    raw = expected.cuda()
    a = raw.as_strided((num_blocks, 3, tail), (raw.stride(0), tail, 1), 5)
    b = raw.as_strided((num_blocks, 4, tail), (raw.stride(0), tail, 1), 3 * tail + 9)
    zeroer = MambaStateZeroer()
    for selected_a, selected_b in [
        ([0, 1, 1, num_blocks - 1], [2, 3]),
        ([4], [0, num_blocks - 1]),
        ([], []),
    ]:
        zeroer.clear([(a, selected_a), (b, selected_b), (a, selected_a)])
        for block in set(selected_a) - {0}:
            expected[block, 5 : 5 + 3 * tail] = 0
        for block in set(selected_b) - {0}:
            expected[block, 3 * tail + 9 : 7 * tail + 9] = 0
        assert torch.equal(raw.cpu(), expected)


@pytest.mark.parametrize("same_blocks", [True, False])
@pytest.mark.parametrize("padding", [0, 7])
def test_adjacent_states_merge_without_clearing_gaps(same_blocks, padding):
    expected = torch.full((17, 101), 0.375, dtype=torch.float16)
    raw = expected.cuda()
    a = raw.as_strided((17, 3, 7), (101, 7, 1), 5)
    b = raw.as_strided((17, 4, 7), (101, 7, 1), 26 + padding)
    blocks_a = [1, 3, 8, 8, 0]
    blocks_b = blocks_a if same_blocks else [2, 3, 9, 0]
    MambaStateZeroer().clear([(a, blocks_a), (b, blocks_b)])
    for block in set(blocks_a) - {0}:
        expected[block, 5:26] = 0
    for block in set(blocks_b) - {0}:
        expected[block, 26 + padding : 54 + padding] = 0
    assert torch.equal(raw.cpu(), expected)


@pytest.mark.parametrize("invalid", [-1, 9])
def test_invalid_plan_is_rejected_before_any_write(invalid):
    expected = torch.full((9, 101), 0.375, dtype=torch.float16)
    raw = expected.cuda()
    with pytest.raises(ValueError, match="out of bounds"):
        MambaStateZeroer().clear([(raw, [1]), (raw, [invalid])])
    assert torch.equal(raw.cpu(), expected)


def test_noncontiguous_inner_shape_is_rejected():
    raw = torch.ones((9, 3, 7), device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        MambaStateZeroer().clear([(raw.transpose(1, 2), [1])])
    assert torch.equal(raw.cpu(), torch.ones((9, 3, 7)))


def test_real_qwen35_state_page_preserves_padding():
    expected = torch.full((13, 524288), 0.375, dtype=torch.float16)
    raw = expected.cuda()
    conv = raw.as_strided((13, 10, 5120), (524288, 5120, 1))
    ssm = raw.as_strided((13, 24, 128, 128), (524288, 16384, 128, 1), 51200)
    zeroer = MambaStateZeroer()
    zeroer.clear([(conv, [0, 1, 7, 12]), (ssm, [0, 1, 7, 12])])
    for block in [1, 7, 12]:
        expected[block, :444416] = 0
    assert torch.equal(raw.cpu(), expected)
    assert zeroer._zero.numel() == 888832
