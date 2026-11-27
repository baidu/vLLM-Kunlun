# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for overflow in the FP16 DFlash2 draft."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F


@pytest.fixture
def dflash2(load_kunlun_override, stub_module_tree):
    stub_module_tree("kunlun_ops")
    stub_module_tree("vllm", _custom_ops=SimpleNamespace(), envs=SimpleNamespace())
    stub_module_tree("vllm.compilation.backends", set_model_tag=object)
    stub_module_tree("vllm.config", CacheConfig=object, VllmConfig=object)
    stub_module_tree(
        "vllm.distributed",
        get_tensor_model_parallel_world_size=lambda: 1,
        tensor_model_parallel_all_gather=object,
    )
    stub_module_tree("vllm.forward_context", get_forward_context=object)
    stub_module_tree("vllm.model_executor.layers.linear", ReplicatedLinear=object)
    stub_module_tree(
        "vllm.model_executor.layers.quantization.base_config", QuantizationConfig=object
    )
    stub_module_tree(
        "vllm.model_executor.layers.vocab_parallel_embedding",
        UnquantizedEmbeddingMethod=type("UnquantizedEmbeddingMethod", (), {}),
    )
    stub_module_tree("vllm.model_executor.models.utils", maybe_prefix=object)
    module, _ = load_kunlun_override(
        "models/qwen3_dflash2.py",
        "vllm.model_executor.models.qwen3_dflash",
        DFlashQwen3Attention=nn.Module,
        DFlashQwen3DecoderLayer=nn.Module,
        DFlashQwen3ForCausalLM=nn.Module,
        DFlashQwen3Model=nn.Module,
    )
    return module


def test_fp16_residual_and_mlp_keep_large_values_finite(dflash2):
    class ResidualNorm(nn.RMSNorm):
        def forward(self, x, residual=None):
            assert x.dtype == torch.float32
            if residual is None:
                return super().forward(x)
            residual = residual + x
            return super().forward(residual), residual

    class IdentityConv(nn.Module):
        def prepare(self, x):
            return x, None

        def finish(self, x, coefficients):
            return x

    layer = dflash2.DFlash2Qwen3DecoderLayer.__new__(dflash2.DFlash2Qwen3DecoderLayer)
    nn.Module.__init__(layer)
    layer.model_dtype = torch.float16
    layer.norm_dtype = torch.float32
    layer.attention_conv = layer.mlp_conv = IdentityConv()
    layer.input_layernorm = ResidualNorm(2)
    layer.post_attention_layernorm = ResidualNorm(2)
    layer.self_attn = lambda **kwargs: torch.full((1, 2), 70_000.0)
    gate = torch.full((1, 2), 300.0, dtype=torch.float16)
    layer.mlp = SimpleNamespace(
        gate_up_proj=lambda x: (gate, None),
        act_fn=lambda x: x.square(),
        down_proj=lambda x: (x * 2, None),
    )
    hidden = torch.full((1, 2), 60_000.0, dtype=torch.float16)

    output, residual = layer(torch.tensor([0]), hidden, None)

    assert output.dtype == residual.dtype == torch.float32
    torch.testing.assert_close(output, torch.full((1, 2), 180_000.0))
    torch.testing.assert_close(residual, torch.full((1, 2), 130_000.0))


def test_fp16_candidate_logits_keep_values_above_fp16_range(dflash2):
    weight = torch.tensor([[2.0, 2.0], [1.0, 1.0], [-2.0, -2.0]], dtype=torch.float16)
    hidden = torch.full((1, 2), 60_000.0, dtype=torch.float16)
    assert F.linear(hidden, weight).isinf().all()
    lm = SimpleNamespace(
        lm_head=SimpleNamespace(
            weight=weight,
            quant_method=dflash2.UnquantizedEmbeddingMethod(),
            shard_indices=SimpleNamespace(
                num_org_vocab_padding=0, org_vocab_start_index=8
            ),
        ),
        model=SimpleNamespace(candidate_selector=SimpleNamespace(top_k=3)),
        output_multiplier=1.0,
        final_logit_softcapping=None,
    )

    ids, logits = dflash2.DFlash2Qwen3ForCausalLM.compute_candidates(lm, hidden)

    assert ids.dtype == torch.int64
    assert logits.dtype == torch.float32
    torch.testing.assert_close(ids, torch.tensor([[8, 9, 10]]))
    torch.testing.assert_close(
        logits, torch.tensor([[240_000.0, 120_000.0, -240_000.0]])
    )
