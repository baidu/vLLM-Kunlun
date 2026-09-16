"""Check that KunlunCompressedTensorsConfig only intercepts MoE.

There is no compressed-tensors checkpoint on this box, so this drives
get_quant_method directly with bare layer instances: what is being verified is
the dispatch, not the schemes. The two branches the old hand-copied dispatch
had dropped (ParallelLMHead, VocabParallelEmbedding) are the point -- they used
to fall through to `return None`, silently treating a quantized lm_head or
embedding as unquantized.
"""

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
    CompressedTensorsKVCacheMethod,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_embedding import (  # noqa: E501
    CompressedTensorsEmbeddingWNA16Int,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_kunlun.quantization.compressed_tensors.compressed_tensors import (
    KunlunCompressedTensorsConfig,
)


@pytest.fixture
def cfg():
    return KunlunCompressedTensorsConfig(
        target_scheme_map={}, ignore=[], quant_format="pack-quantized"
    )


def test_registration_and_super_chain():
    assert issubclass(KunlunCompressedTensorsConfig, CompressedTensorsConfig)
    assert "get_quant_method" in KunlunCompressedTensorsConfig.__dict__
    # super() has to resolve to upstream's dispatch, not a hand-copied one.
    assert KunlunCompressedTensorsConfig.__mro__[1] is CompressedTensorsConfig
    assert (
        CompressedTensorsConfig.get_quant_method
        is KunlunCompressedTensorsConfig.__mro__[1].__dict__["get_quant_method"]
    )
    assert (
        get_quantization_config("compressed-tensors") is KunlunCompressedTensorsConfig
    )


# Bare instances: get_quant_method only needs isinstance plus, for the branches
# that look one up, a scheme -- and an empty target_scheme_map yields none.
def test_non_moe_linear_reaches_upstream(cfg):
    linear = object.__new__(LinearBase)
    got = cfg.get_quant_method(linear, "model.layers.0.mlp.gate_proj")
    assert isinstance(got, UnquantizedLinearMethod)


def test_attention_reaches_upstream_kv_cache(cfg):
    attn = object.__new__(Attention)
    got = cfg.get_quant_method(attn, "model.layers.0.self_attn.attn")
    assert isinstance(got, CompressedTensorsKVCacheMethod)


# These two are what the copied dispatch had lost: previously they fell all the
# way through to `return None`.
@pytest.mark.parametrize("cls", [ParallelLMHead, VocabParallelEmbedding])
def test_unquantized_head_and_embedding_return_none(cfg, cls):
    layer = object.__new__(cls)
    assert cfg.get_quant_method(layer, "lm_head") is None


# Reachability alone proves little when an empty scheme map makes every branch
# return None, so feed each branch a scheme and check which method comes back.
def test_quantized_lm_head_reaches_linear_method(cfg):
    lm_head = object.__new__(ParallelLMHead)
    cfg.get_scheme = lambda layer, layer_name: "sentinel-scheme"
    got = cfg.get_quant_method(lm_head, "lm_head")
    assert isinstance(got, CompressedTensorsLinearMethod)
    assert getattr(lm_head, "scheme", None) == "sentinel-scheme"


def test_quantized_embedding_reaches_wna16_int(cfg):
    embedding = object.__new__(VocabParallelEmbedding)
    wna16_int = QuantizationArgs(
        num_bits=4, type="int", strategy="group", group_size=128, symmetric=True
    )
    cfg.get_scheme_dict = lambda layer, layer_name: {"weights": wna16_int}
    got = cfg.get_quant_method(embedding, "model.embed_tokens")
    assert isinstance(got, CompressedTensorsEmbeddingWNA16Int)


def test_non_wna16_int_embedding_is_rejected(cfg):
    embedding = object.__new__(VocabParallelEmbedding)
    fp8 = QuantizationArgs(num_bits=8, type="float", strategy="tensor", symmetric=True)
    cfg.get_scheme_dict = lambda layer, layer_name: {"weights": fp8}
    with pytest.raises(ValueError):
        cfg.get_quant_method(embedding, "model.embed_tokens")


def test_unknown_layer_returns_none(cfg):
    assert cfg.get_quant_method(torch.nn.LayerNorm(4), "model.norm") is None


def test_moe_is_still_intercepted(cfg):
    experts = object.__new__(RoutedExperts)
    # The bare instance has no moe_config/quant attributes, so the Kunlun factory
    # is expected to die inside itself -- which is the proof it ran.
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        cfg.get_quant_method(experts, "model.layers.0.mlp.experts")
    frames = []
    tb = excinfo.value.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    assert "get_moe_method" in frames, frames
