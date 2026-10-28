import torch

from vllm_kunlun.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    get_num_sampled_and_rejected,
    post_update_num_computed_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)


def test_prepare_prefill_inputs_updates_prompt_and_next_token():
    input_ids = torch.full((4,), -1, dtype=torch.int64)
    next_tokens = torch.full((2,), -1, dtype=torch.int64)
    prepare_prefill_inputs(
        input_ids,
        next_tokens,
        torch.tensor([1, 0]),
        torch.tensor([0, 2, 4]),
        torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
        torch.tensor([4, 3]),
        torch.tensor([1, 0]),
    )
    assert torch.equal(input_ids, torch.tensor([20, 21, 11, 12]))
    assert torch.equal(next_tokens, torch.tensor([13, 22]))


def test_prepare_pos_seq_lens_writes_positions_and_clears_padding():
    pos = torch.full((4,), -1, dtype=torch.int64)
    seq_lens = torch.full((4,), -1, dtype=torch.int32)
    prepare_pos_seq_lens(
        torch.tensor([1, 0]),
        torch.tensor([0, 2, 3]),
        torch.tensor([4, 1]),
        pos,
        seq_lens,
    )
    assert torch.equal(pos[:3], torch.tensor([1, 2, 4]))
    assert torch.equal(seq_lens, torch.tensor([3, 5, 0, 0]))


def test_combine_sampled_and_draft_tokens_splices_decode_tokens():
    input_ids = torch.tensor([7, 8, 9, 0, 0, 0], dtype=torch.int64)
    indices = combine_sampled_and_draft_tokens(
        input_ids,
        torch.tensor([0]),
        torch.tensor([42]),
        torch.tensor([0, 3]),
        torch.tensor([5]),
        torch.tensor([3]),
        torch.tensor([[51, 52]]),
        torch.tensor([0, 3]),
        num_logits=3,
    )
    assert torch.equal(indices, torch.tensor([0, 1, 2]))
    assert torch.equal(input_ids, torch.tensor([7, 51, 52, 0, 0, 0]))


def test_sampled_rejected_and_expanded_mappings():
    sampled = torch.tensor([2, 4, 9, 9], dtype=torch.int32)
    result, rejected = get_num_sampled_and_rejected(
        sampled,
        torch.tensor([5, 2, 7]),
        torch.tensor([0, 3, 5]),
        torch.tensor([1, 0]),
        torch.tensor([4, 2]),
    )
    assert torch.equal(result[:2], torch.tensor([2, 0]))
    assert torch.equal(rejected[:2], torch.tensor([1, 0]))

    expanded, local = expand_idx_mapping(
        torch.tensor([3, 1]), 5, torch.tensor([0, 2, 2]), max_expand_len=4
    )
    assert torch.equal(expanded[:2], torch.tensor([3, 3]))
    assert torch.equal(local[:2], torch.tensor([0, 1]))


def test_post_update_num_computed_tokens_uses_request_mapping():
    computed = torch.tensor([10, 20, 30], dtype=torch.int32)
    post_update_num_computed_tokens(
        torch.tensor([2, 0]), computed, torch.tensor([0, 3, 4])
    )
    assert torch.equal(computed, torch.tensor([11, 20, 33]))
