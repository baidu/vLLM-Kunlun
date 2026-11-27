# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU coverage for the DMTP draft embedding layout."""

import importlib.util
from pathlib import Path

import torch
from torch import nn


def test_dmtp_conditioner_only_updates_t1(stub_module):
    stub_module(
        "vllm_kunlun.models.qwen3_dflash2",
        DFlash2Qwen3Attention=nn.Module,
        DFlash2Qwen3DecoderLayer=nn.Module,
        DFlash2Qwen3ForCausalLM=nn.Module,
        DFlash2Qwen3Model=nn.Module,
    )
    model_path = (
        Path(__file__).resolve().parents[2] / "vllm_kunlun" / "models" / "qwen3_dmtp.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_kunlun.models._dmtp_under_test", model_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    conditioner = module.T1Conditioner(width=2, rank=2, eps=1e-6)
    embeddings = torch.arange(32, dtype=torch.float32).view(16, 2)
    native_hidden = torch.tensor([[2.0, 0.0], [0.0, -3.0]])
    with torch.no_grad():
        conditioner.down.weight.copy_(torch.eye(2))
        conditioner.up.weight.copy_(torch.eye(2))
        output = conditioner(embeddings, native_hidden).view(2, 8, 2)

    blocks = embeddings.view(2, 8, 2)
    normalized = (
        native_hidden / (native_hidden.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    )
    torch.testing.assert_close(output[:, 0], blocks[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(
        output[:, 1], blocks[:, 1] + nn.functional.silu(normalized)
    )
    torch.testing.assert_close(output[:, 2:], blocks[:, 2:], rtol=0, atol=0)
