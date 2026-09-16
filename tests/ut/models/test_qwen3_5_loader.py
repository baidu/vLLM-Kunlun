# SPDX-License-Identifier: Apache-2.0

import torch

import vllm_kunlun.models.qwen3_5 as qwen3_5

GATE_UP_NAME = "model.layers.0.mlp.experts.gate_up_proj"
W13_NAME = "model.layers.0.mlp.experts.routed_experts.w13_weight"
DOWN_NAME = "model.layers.0.mlp.experts.down_proj"
W2_NAME = "model.layers.0.mlp.experts.routed_experts.w2_weight"


class FakeParameter:
    def __init__(self, accepted_experts=None):
        self.calls = []
        self.accepted_experts = accepted_experts

    def weight_loader(
        self, param, weight, name, shard_id, expert_id=None, return_success=False
    ):
        self.calls.append((weight, name, shard_id, expert_id))
        success = self.accepted_experts is None or expert_id in self.accepted_experts
        return success if return_success else None


def make_model(monkeypatch, params, mapping=()):
    model = object.__new__(qwen3_5.Qwen3_5Model)
    monkeypatch.setattr(
        qwen3_5.Qwen3_5Model,
        "named_parameters",
        lambda self: params.items(),
    )
    monkeypatch.setattr(
        qwen3_5.Qwen3_5Model,
        "get_expert_mapping",
        lambda self: list(mapping),
    )
    monkeypatch.setattr(qwen3_5, "is_pp_missing_parameter", lambda name, model: False)
    return model


def test_fused_gate_up_splits_per_expert_and_records_local_load(monkeypatch):
    parameter = FakeParameter(accepted_experts={1, 3})
    model = make_model(monkeypatch, {W13_NAME: parameter})
    weight = torch.arange(4 * 8 * 6).reshape(4, 8, 6)

    loaded = model.load_weights([(GATE_UP_NAME, weight)])

    assert W13_NAME in loaded
    assert [(call[2], call[3], tuple(call[0].shape)) for call in parameter.calls] == [
        ("w1", 0, (4, 6)),
        ("w1", 1, (4, 6)),
        ("w1", 2, (4, 6)),
        ("w1", 3, (4, 6)),
        ("w3", 0, (4, 6)),
        ("w3", 1, (4, 6)),
        ("w3", 2, (4, 6)),
        ("w3", 3, (4, 6)),
    ]
    for expert_id in range(weight.shape[0]):
        assert torch.equal(parameter.calls[expert_id][0], weight[expert_id, :4])
        assert torch.equal(
            parameter.calls[weight.shape[0] + expert_id][0], weight[expert_id, 4:]
        )


def test_fused_down_uses_w2_and_omits_all_nonlocal_parameter(monkeypatch):
    parameter = FakeParameter(accepted_experts=set())
    model = make_model(monkeypatch, {W2_NAME: parameter})
    weight = torch.randn(4, 6, 5)

    loaded = model.load_weights([(DOWN_NAME, weight)])

    assert W2_NAME not in loaded
    assert [(call[2], call[3], tuple(call[0].shape)) for call in parameter.calls] == [
        ("w2", 0, (6, 5)),
        ("w2", 1, (6, 5)),
        ("w2", 2, (6, 5)),
        ("w2", 3, (6, 5)),
    ]


def test_non_fused_expert_weight_keeps_direct_loader_behavior(monkeypatch):
    parameter = FakeParameter()
    mapping = [("experts.routed_experts.w13_weight", "experts.w1_weight", 2, "w1")]
    model = make_model(monkeypatch, {W13_NAME: parameter}, mapping)
    weight = torch.randn(6, 5)

    loaded = model.load_weights([("model.layers.0.mlp.experts.w1_weight", weight)])

    assert W13_NAME in loaded
    assert len(parameter.calls) == 1
    call = parameter.calls[0]
    assert call[2:] == ("w1", 2)
    assert call[0].shape == weight.shape
