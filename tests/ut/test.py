from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F


def _load_wrapper_module():
    import vllm_kunlun  # noqa: F401

    return importlib.import_module("vllm_kunlun.compilation.wrapper")


def _make_mock_config(wrapper_module):
    dynamic_shapes_config = SimpleNamespace(
        evaluate_guards=False,
        type=wrapper_module.DynamicShapesType.BACKED,
    )
    compilation_config = SimpleNamespace(
        mode=wrapper_module.CompilationMode.DYNAMO_TRACE_ONCE,
        init_backend=lambda _cfg: "eager",
        inductor_compile_config={},
        dynamic_shapes_config=dynamic_shapes_config,
        cudagraph_mode=wrapper_module.CUDAGraphMode.NONE,
    )
    observability_config = SimpleNamespace(enable_layerwise_nvtx_tracing=False)
    return SimpleNamespace(
        compilation_config=compilation_config,
        observability_config=observability_config,
        compile_debug_dump_path=lambda: None,
    )


def test_import_installs_torch251_compat_shims():
    wrapper_module = _load_wrapper_module()
    custom_graph_pass = importlib.import_module("torch._inductor.custom_graph_pass")
    graph_pickler_module = importlib.import_module("torch.fx._graph_pickler")
    dynamo_torch_module = importlib.import_module("torch._dynamo.variables.torch")
    mistral_request_module = importlib.import_module(
        "mistral_common.protocol.instruct.request"
    )
    from torch.fx import Graph
    from vllm.model_executor.parameter import BasevLLMParameter, ModelWeightParameter

    assert hasattr(custom_graph_pass, "CustomGraphPass")
    assert hasattr(graph_pickler_module, "GraphPickler")
    assert hasattr(graph_pickler_module, "Options")
    assert hasattr(mistral_request_module, "ReasoningEffort")
    assert hasattr(Graph, "output_node")
    assert hasattr(torch.compiler, "set_stance")
    assert hasattr(torch._functorch.config, "bundled_autograd_cache")
    assert hasattr(torch._functorch.config, "autograd_cache_normalize_inputs")
    assert hasattr(torch._functorch.config, "enable_remote_autograd_cache")
    assert "_cache_config_ignore_prefix" in torch._functorch.config._config
    assert "_save_config_ignore" in torch._functorch.config._config
    assert "_compile_ignored_keys" in torch._functorch.config._config
    assert isinstance(torch._functorch.config.save_config_portable(), dict)
    assert wrapper_module.TorchCompileWithNoGuardsWrapper is not None
    assert BasevLLMParameter in torch._dynamo.config.traceable_tensor_subclasses
    assert ModelWeightParameter in torch._dynamo.config.traceable_tensor_subclasses
    assert getattr(
        dynamo_torch_module.can_dispatch_torch_function,
        "_vllm_kunlun_patched",
        False,
    )

    graph = Graph()
    input_node = graph.placeholder("x")
    graph.output(input_node)
    assert graph.output_node().op == "output"


def test_import_allows_vllm_mistral_tokenizer_on_older_mistral_common():
    import vllm_kunlun  # noqa: F401

    mistral_tokenizer_module = importlib.import_module("vllm.tokenizers.mistral")

    assert hasattr(mistral_tokenizer_module, "MistralTokenizer")


def test_import_patches_v1_block_table_slot_mapping_kernel():
    import vllm_kunlun

    block_table_module = vllm_kunlun._custom_import("vllm.v1.worker.block_table")

    assert getattr(
        block_table_module._compute_slot_mapping_kernel,
        "_vllm_kunlun_patched",
        False,
    )

    query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    positions = torch.tensor([0, 1, 5], dtype=torch.int64)
    block_table = torch.tensor([[4, 7]], dtype=torch.int32)
    slot_mapping = torch.empty(5, dtype=torch.int64)

    block_table_module._compute_slot_mapping_kernel[(2,)](
        3,
        5,
        query_start_loc,
        positions,
        block_table,
        2,
        4,
        slot_mapping,
        TOTAL_CP_WORLD_SIZE=1,
        TOTAL_CP_RANK=0,
        CP_KV_CACHE_INTERLEAVE_SIZE=1,
        PAD_ID=-1,
        BLOCK_SIZE=1024,
    )

    assert slot_mapping.tolist() == [16, 17, 29, -1, -1]


def test_register_imports_python_custom_ops_when_missing(monkeypatch):
    import vllm_kunlun

    imported_modules = []
    monkeypatch.setattr(vllm_kunlun, "_has_required_python_custom_ops", lambda: False)
    monkeypatch.setattr(
        vllm_kunlun.importlib,
        "import_module",
        lambda name: imported_modules.append(name),
    )

    vllm_kunlun._ensure_python_custom_ops_registered(
        SimpleNamespace(info=lambda *args, **kwargs: None)
    )

    assert imported_modules == ["vllm_kunlun.ops._custom_ops"]


def test_register_skips_python_custom_ops_when_present(monkeypatch):
    import vllm_kunlun

    monkeypatch.setattr(vllm_kunlun, "_has_required_python_custom_ops", lambda: True)
    monkeypatch.setattr(
        vllm_kunlun.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected import: {name}"),
    )

    vllm_kunlun._ensure_python_custom_ops_registered(
        SimpleNamespace(info=lambda *args, **kwargs: None)
    )


def test_register_requires_vllm019_cache_mla_custom_op(monkeypatch):
    import vllm_kunlun

    monkeypatch.setattr(vllm_kunlun, "_has_scaled_int8_quant_op", lambda: True)
    monkeypatch.setattr(vllm_kunlun, "_has_cache_concat_mla_op", lambda: False)

    assert vllm_kunlun._has_required_python_custom_ops() is False


def test_custom_ops_registers_vllm019_cache_mla_alias():
    import vllm_kunlun

    vllm_kunlun.register()

    assert hasattr(torch.ops._C_cache_ops, "concat_and_cache_mla")


def test_scaled_int8_quant_torch_fallback_uses_per_token_absmax():
    from vllm_kunlun.ops import _custom_ops

    x = torch.tensor(
        [
            [[0.0, 1.0, -2.0], [3.0, -3.0, 1.5]],
            [[0.0, 0.0, 0.0], [1.2, -1.2, 0.6]],
        ],
        dtype=torch.float32,
    )
    x_q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((x.numel() // x.shape[-1], 1), dtype=torch.float32)

    _custom_ops._scaled_int8_quant_torch_fallback(x, x_q, scale)

    x_2d = x.reshape(-1, x.shape[-1])
    expected_scale = torch.amax(torch.abs(x_2d), dim=-1, keepdim=True)
    denom = torch.where(
        expected_scale > 0,
        expected_scale,
        torch.ones_like(expected_scale),
    )
    expected_q = torch.round(x_2d * (127.0 / denom)).clamp_(-127, 127)

    torch.testing.assert_close(scale, expected_scale)
    assert x_q.reshape(-1, x.shape[-1]).tolist() == expected_q.to(torch.int8).tolist()


def test_dynamic_scaled_int8_quant_falls_back_when_quant2d_fails(monkeypatch):
    from vllm_kunlun.ops import _custom_ops

    def fail_quant2d(**kwargs):
        raise RuntimeError("quant2d failed")

    monkeypatch.setattr(_custom_ops.kunlun_ops, "quant2d", fail_quant2d)
    x = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    x_q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((1, 1), dtype=torch.float32)

    _custom_ops._dynamic_scaled_int8_quant(x, x_q, scale)

    torch.testing.assert_close(scale, torch.tensor([[2.0]]))
    assert x_q.tolist() == [[64, -127, 32]]


def test_dynamic_scaled_int8_quant_uses_torch_after_sdnn_failure(monkeypatch):
    from vllm_kunlun.ops import _custom_ops

    calls = []

    def quant2d(**kwargs):
        calls.append(kwargs["force_sdnn"])
        raise RuntimeError("sdnn quant2d failed")

    monkeypatch.setattr(_custom_ops.kunlun_ops, "quant2d", quant2d)
    x = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    x_q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((1, 1), dtype=torch.float32)

    _custom_ops._dynamic_scaled_int8_quant(x, x_q, scale)

    assert calls == [True]
    torch.testing.assert_close(scale, torch.tensor([[2.0]]))
    assert x_q.tolist() == [[64, -127, 32]]


def test_scaled_int8_mm_torch_fallback_dequantizes_per_token_and_channel():
    import vllm_kunlun

    vllm_kunlun.register()
    from vllm_kunlun.quantization.kernels.scale_mm import (
        _torch_scaled_int8_mm_fallback,
    )

    x_q = torch.tensor([[127, -64], [0, 127]], dtype=torch.int8)
    w_q = torch.tensor([[127, 0, -64], [-127, 64, 127]], dtype=torch.int8)
    x_max = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
    w_max = torch.tensor([[8.0], [6.0], [10.0]], dtype=torch.float32)
    bias = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)

    out = _torch_scaled_int8_mm_fallback(
        x_q=x_q,
        w_q=w_q,
        x_max=x_max,
        w_max=w_max,
        out_dtype=torch.float32,
        bias=bias,
    )

    expected = torch.matmul(
        x_q.to(torch.float32) * (x_max / 127.0),
        w_q.to(torch.float32) * (w_max.reshape(1, -1) / 127.0),
    )
    expected = expected + bias
    torch.testing.assert_close(out, expected)


def test_default_unquantized_gemm_supports_torch_compile_with_vllm_parameter():
    from vllm.model_executor.layers import utils as layer_utils
    from vllm.model_executor.parameter import ModelWeightParameter

    import vllm_kunlun  # noqa: F401

    with (
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        weight = ModelWeightParameter(
            input_dim=1,
            output_dim=0,
            data=torch.full((4, 4), 2.0),
            weight_loader=lambda *args, **kwargs: None,
        )

    compiled = torch.compile(
        lambda x, param: layer_utils.default_unquantized_gemm(None, x, param, None),
        backend="inductor",
        fullgraph=True,
    )

    torch.testing.assert_close(
        compiled(torch.ones((1, 4)), weight),
        torch.full((1, 4), 8.0),
    )


def test_get_fx_graph_outputs_falls_back_to_last_node_without_output():
    from torch.fx import Graph

    import vllm_kunlun  # noqa: F401
    from vllm_kunlun.compat import _get_fx_graph_outputs

    graph = Graph()
    input_node = graph.placeholder("x")
    last_node = graph.call_function(torch.neg, (input_node,))

    assert _get_fx_graph_outputs(graph) is last_node

    graph.output(last_node)
    assert _get_fx_graph_outputs(graph) is last_node


def test_get_fx_graph_outputs_uses_mutated_args_for_inplace_custom_op():
    from torch.fx import Graph

    import vllm_kunlun  # noqa: F401
    from vllm_kunlun.compat import _get_fx_graph_outputs

    importlib.import_module("vllm.model_executor.layers.attention.attention")

    graph = Graph()
    query = graph.placeholder("query")
    key = graph.placeholder("key")
    value = graph.placeholder("value")
    output = graph.placeholder("output")
    attention_node = graph.call_function(
        torch.ops.vllm.unified_attention_with_output.default,
        (query, key, value, output, "layer_name", None, None, None),
    )

    assert attention_node.op == "call_function"
    assert _get_fx_graph_outputs(graph) is output


def test_get_fx_graph_outputs_uses_name_based_mutated_arg_fallback():
    from torch.fx import Graph

    import vllm_kunlun  # noqa: F401
    from vllm_kunlun.compat import _get_fx_graph_outputs

    def unified_attention_with_output(*args, **kwargs):
        raise AssertionError("FX test should not execute the target")

    graph = Graph()
    query = graph.placeholder("query")
    key = graph.placeholder("key")
    value = graph.placeholder("value")
    output = graph.placeholder("output")
    output_block_scale = graph.placeholder("output_block_scale")
    attention_node = graph.call_function(
        unified_attention_with_output,
        (
            query,
            key,
            value,
            output,
            "layer_name",
            None,
            output_block_scale,
            None,
        ),
    )

    assert attention_node.op == "call_function"
    assert _get_fx_graph_outputs(graph) == (output, output_block_scale)


def test_import_patches_piecewise_compile_interpreter_call_module():
    import vllm_kunlun  # noqa: F401

    backends_module = importlib.import_module("vllm.compilation.backends")

    assert getattr(
        backends_module.PiecewiseCompileInterpreter.call_module,
        "_vllm_kunlun_patched",
        False,
    )


def test_register_model_imports_vllm_on_torch251():
    from vllm import ModelRegistry

    import vllm_kunlun

    with patch.object(ModelRegistry, "register_model") as mock_register_model:
        vllm_kunlun.register_model()

    assert mock_register_model.call_count > 0
    assert any(
        call.args[0] == "Qwen3MoeForCausalLM"
        and call.args[1] == "vllm_kunlun.models.qwen3_moe:Qwen3MoeForCausalLM"
        for call in mock_register_model.call_args_list
    )
    assert any(
        call.args[0] == "DeepseekV32ForCausalLM"
        and getattr(call.args[1], "__name__", None) == "DeepseekV3ForCausalLM"
        for call in mock_register_model.call_args_list
    )


def test_kunlun_platform_allows_missing_vllm_attention_backend(monkeypatch):
    from vllm_kunlun.platforms import kunlun

    monkeypatch.delattr(kunlun.envs, "VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    assert kunlun._get_vllm_attention_backend() is None

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASHMLA")
    assert kunlun._get_vllm_attention_backend() == "FLASHMLA"
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)

    monkeypatch.setattr(
        kunlun.envs,
        "VLLM_ATTENTION_BACKEND",
        "FLASHMLA",
        raising=False,
    )
    assert kunlun._get_vllm_attention_backend() == "FLASHMLA"


def test_kunlun_platform_accepts_renamed_flashmla_support_check():
    from vllm_kunlun.platforms import kunlun

    legacy_module = SimpleNamespace(is_flashmla_supported=lambda: (True, None))
    dense_module = SimpleNamespace(is_flashmla_dense_supported=lambda: (False, "x"))

    assert kunlun._check_flashmla_supported(legacy_module) == (True, None)
    assert kunlun._check_flashmla_supported(dense_module) == (False, "x")


def test_kunlun_platform_flashmla_env_overrides_sparse_mla(monkeypatch):
    from vllm_kunlun.platforms import kunlun

    monkeypatch.setattr(kunlun, "_get_vllm_attention_backend", lambda: "FLASHMLA")

    backend_cls = kunlun.KunlunPlatform.get_attn_backend_cls(
        selected_backend=SimpleNamespace(),
        attn_selector_config=SimpleNamespace(use_mla=True, use_sparse=True),
    )

    assert backend_cls == (
        "vllm_kunlun.v1.attention.backends.mla.flashmla.FlashMLABackend"
    )


def test_kunlun_quantization_loader_skips_missing_optional_backends(monkeypatch):
    from vllm_kunlun.platforms import kunlun

    kunlun._patch_quantization_config_loader()

    import vllm.model_executor.layers.quantization as quant_module

    optional_backend = "vllm.model_executor.layers.quantization.bitblas"
    if importlib.util.find_spec(optional_backend) is None:
        assert "bitblas" not in quant_module.QUANTIZATION_METHODS

    class FakeCompressedTensorsConfig:
        pass

    monkeypatch.setitem(
        quant_module._CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
        "compressed-tensors",
        FakeCompressedTensorsConfig,
    )
    quant_config = quant_module.get_quantization_config("compressed-tensors")
    assert quant_config is FakeCompressedTensorsConfig

    class FakeModelOptMxFp8Config:
        pass

    def fake_import_module(name):
        if name == "vllm.model_executor.layers.quantization.modelopt":
            return SimpleNamespace(ModelOptMxFp8Config=FakeModelOptMxFp8Config)
        return importlib.import_module(name)

    with patch.object(
        kunlun.importlib, "import_module", side_effect=fake_import_module
    ):
        assert (
            quant_module.get_quantization_config("modelopt_mxfp8")
            is FakeModelOptMxFp8Config
        )

    weight_utils = SimpleNamespace(get_quantization_config=lambda _: None)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.weight_utils",
        weight_utils,
    )
    kunlun._patch_quantization_config_loader()
    assert weight_utils.get_quantization_config is quant_module.get_quantization_config


def test_kunlun_compressed_tensors_accepts_vllm019_attention_path():
    import vllm_kunlun

    vllm_kunlun.register()
    from vllm_kunlun.quantization.compressed_tensors.compressed_tensors import (
        KunlunCompressedTensorsConfig,
    )

    quant_config = KunlunCompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format="int-quantized",
        sparsity_scheme_map={},
        sparsity_ignore_list=[],
    )

    assert quant_config.get_quant_method(torch.nn.Module(), prefix="x") is None


def test_deepseek_rope_compat_supports_vllm019_signature():
    import vllm_kunlun

    vllm_kunlun.register()
    from vllm_kunlun.models.deepseek_v2 import _get_rope_compat

    calls = {}

    def fake_get_rope(
        head_size,
        max_position,
        is_neox_style=True,
        rope_parameters=None,
        dtype=None,
        dual_chunk_attention_config=None,
    ):
        calls.update(
            head_size=head_size,
            max_position=max_position,
            is_neox_style=is_neox_style,
            rope_parameters=rope_parameters,
        )
        return "rope"

    result = _get_rope_compat(
        64,
        rotary_dim=64,
        max_position=32768,
        base=1000000.0,
        rope_scaling={"factor": 40, "rope_type": "deepseek_yarn"},
        is_neox_style=False,
        rope_fn=fake_get_rope,
    )

    assert result == "rope"
    assert calls["head_size"] == 64
    assert calls["max_position"] == 32768
    assert calls["is_neox_style"] is False
    assert calls["rope_parameters"]["rope_theta"] == 1000000.0
    assert calls["rope_parameters"]["rope_type"] == "deepseek_yarn"


def test_deepseek_v32_mla_passes_indexer_rotary_emb(monkeypatch):
    import vllm_kunlun

    vllm_kunlun.register()
    import vllm_kunlun.models.deepseek_v2 as deepseek_module

    captured = {}

    class FakeModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class FakeIndexer(torch.nn.Module):
        topk_tokens = 1

        def __init__(self, *args, **kwargs):
            super().__init__()

    class FakeMLAWrapper(torch.nn.Module):
        def __init__(
            self,
            _hidden_size,
            _num_local_heads,
            _scaling,
            _qk_nope_head_dim,
            _qk_rope_head_dim,
            _v_head_dim,
            _q_lora_rank,
            _kv_lora_rank,
            mla_modules,
            *_args,
            **_kwargs,
        ):
            super().__init__()
            captured["mla_modules"] = mla_modules

    def fake_get_rope_compat(*_args, is_neox_style, **_kwargs):
        return f"rope-{is_neox_style}"

    monkeypatch.setattr(
        deepseek_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(deepseek_module, "_get_rope_compat", fake_get_rope_compat)
    monkeypatch.setattr(deepseek_module, "MergedColumnParallelLinear", FakeModule)
    monkeypatch.setattr(deepseek_module, "ReplicatedLinear", FakeModule)
    monkeypatch.setattr(deepseek_module, "ColumnParallelLinear", FakeModule)
    monkeypatch.setattr(deepseek_module, "RowParallelLinear", FakeModule)
    monkeypatch.setattr(deepseek_module, "RMSNorm", FakeModule)
    monkeypatch.setattr(deepseek_module, "LayerNorm", FakeModule)
    monkeypatch.setattr(deepseek_module, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        deepseek_module,
        "MultiHeadLatentAttentionWrapper",
        FakeMLAWrapper,
    )

    config = SimpleNamespace(
        rms_norm_eps=1e-6,
        index_topk=2048,
        index_n_heads=64,
        index_head_dim=128,
        qk_rope_head_dim=64,
    )

    deepseek_module.DeepseekV2MLAAttention(
        vllm_config=SimpleNamespace(),
        config=config,
        hidden_size=16,
        num_heads=1,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        q_lora_rank=4,
        kv_lora_rank=4,
        rope_scaling={"factor": 40},
        cache_config=SimpleNamespace(),
    )

    mla_modules = captured["mla_modules"]
    assert mla_modules.rotary_emb == "rope-False"
    assert mla_modules.indexer_rotary_emb == "rope-True"


def test_deepseek_v32_indexer_rope_outputs_restore_token_shapes():
    from vllm_kunlun.models.deepseek_v2 import _reshape_indexer_rope_outputs

    q_pe, k_pe = _reshape_indexer_rope_outputs(
        torch.zeros(1, 2, 64, 64),
        torch.zeros(1, 2, 1, 64),
        n_head=64,
        rope_dim=64,
    )

    assert q_pe.shape == (2, 64, 64)
    assert k_pe.shape == (2, 1, 64)


def test_deepseek_v32_indexer_passes_whole_kv_cache():
    from vllm_kunlun.models.deepseek_v2 import _get_indexer_kv_cache

    kv_cache = torch.tensor([])

    assert _get_indexer_kv_cache(SimpleNamespace(kv_cache=kv_cache)) is kv_cache


def test_flashmla_sparse_impl_uses_sparse_mla_interface():
    import inspect as py_inspect

    import vllm_kunlun

    vllm_kunlun.register()
    from vllm.v1.attention.backend import SparseMLAAttentionImpl

    from vllm_kunlun.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseImpl,
    )

    assert not py_inspect.isabstract(FlashMLASparseImpl)
    assert issubclass(FlashMLASparseImpl, SparseMLAAttentionImpl)


def test_flashmla_dense_backend_imports_vllm19_attention_symbols():
    import inspect as py_inspect

    import vllm_kunlun

    vllm_kunlun.register()
    from vllm.v1.attention.backend import AttentionCGSupport

    from vllm_kunlun.v1.attention.backends.mla.flashmla import (
        FlashMLABackend,
        FlashMLAImpl,
        FlashMLAMetadataBuilder,
    )

    assert FlashMLABackend.get_name() == "FLASHMLA"
    assert not py_inspect.isabstract(FlashMLAImpl)
    assert FlashMLAMetadataBuilder.cudagraph_support is AttentionCGSupport.UNIFORM_BATCH


def test_flashmla_decode_paged_attention_splits_q_c_and_q_r(monkeypatch):
    import vllm_kunlun

    vllm_kunlun.register()
    from vllm_kunlun.ops.attention import flashmla

    captured = {}

    def fake_paged_attention(
        out,
        x,
        k_cache,
        v_cache,
        block_tables,
        context_lens_cpu,
        context_lens_xpu,
        is_context,
        is_causal,
        vo_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        mla_scale,
        **kwargs,
    ):
        captured.update(
            out=out,
            x=x,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens_cpu=context_lens_cpu,
            context_lens_xpu=context_lens_xpu,
            is_context=is_context,
            is_causal=is_causal,
            vo_head_dim=vo_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            mla_scale=mla_scale,
            q_r=kwargs["q_r"],
        )
        return 0

    monkeypatch.setattr(flashmla.kunlun_ops, "paged_attention", fake_paged_attention)

    q = torch.arange(2 * 1 * 3 * 6, dtype=torch.float32).reshape(2, 1, 3, 6)
    k_cache = torch.zeros(4, 8, 1, 6)
    block_table = torch.zeros(2, 1, dtype=torch.int32)
    cache_lens = torch.ones(2, dtype=torch.int32)

    out, _ = flashmla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_lens,
        head_dim_v=4,
        tile_scheduler_metadata=cache_lens.cpu(),
        num_splits=cache_lens,
        softmax_scale=0.5,
        causal=True,
    )

    assert captured["out"] is out
    assert captured["x"].shape == (2, 1, 3, 4)
    assert captured["q_r"].shape == (2, 1, 3, 2)
    assert torch.equal(captured["x"], q[..., :4])
    assert torch.equal(captured["q_r"], q[..., 4:])
    assert captured["k_cache"].shape == (4, 1, 8, 6)
    assert captured["v_cache"] is None
    assert captured["kv_lora_rank"] == 4
    assert captured["qk_rope_head_dim"] == 2
    assert captured["mla_scale"] == 0.5


def test_flashmla_decode_torch_fallback_matches_reference(monkeypatch):
    import vllm_kunlun

    vllm_kunlun.register()
    from vllm_kunlun.ops.attention import flashmla

    monkeypatch.setenv("VLLM_KUNLUN_MLA_DECODE_FALLBACK", "1")

    q = torch.tensor(
        [[[[0.1, -0.2, 0.3, 0.4, 0.5, -0.6], [0.2, 0.1, -0.4, 0.3, -0.1, 0.7]]]],
        dtype=torch.float32,
    )
    k_cache = torch.tensor(
        [
            [
                [[0.2, 0.0, -0.1, 0.4, 0.1, -0.2]],
                [[-0.3, 0.2, 0.5, 0.1, -0.4, 0.3]],
            ],
            [
                [[0.6, -0.5, 0.2, 0.0, 0.2, 0.1]],
                [[9.0, 9.0, 9.0, 9.0, 9.0, 9.0]],
            ],
        ],
        dtype=torch.float32,
    )
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    cache_lens = torch.tensor([3], dtype=torch.int32)

    out, _ = flashmla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_lens,
        head_dim_v=4,
        tile_scheduler_metadata=cache_lens.cpu(),
        num_splits=cache_lens,
        softmax_scale=0.5,
        causal=True,
    )

    kv_tokens = torch.cat([k_cache[0, :, 0], k_cache[1, :1, 0]], dim=0)
    q_c = q[..., :4]
    q_r = q[..., 4:]
    kv_c = kv_tokens[:, :4]
    k_pe = kv_tokens[:, 4:]
    scores = torch.einsum("qhd,kd->qhk", q_c[0], kv_c)
    scores = scores + torch.einsum("qhr,kr->qhk", q_r[0], k_pe)
    probs = torch.softmax(scores * 0.5, dim=-1)
    expected = torch.einsum("qhk,kd->qhd", probs, kv_c).unsqueeze(0)

    assert torch.allclose(out, expected)


def test_mla_common_prefill_env_flags_default_when_missing(monkeypatch):
    import vllm_kunlun

    vllm_kunlun.register()
    from vllm_kunlun.v1.attention.backends.mla import common

    assert "flash_attn_varlen_func" in vars(common)

    monkeypatch.delattr(
        common.envs,
        "VLLM_DISABLE_FLASHINFER_PREFILL",
        raising=False,
    )
    monkeypatch.delattr(common.envs, "VLLM_USE_CUDNN_PREFILL", raising=False)
    monkeypatch.setattr(common, "flashinfer_available", True)
    monkeypatch.setattr(
        common,
        "current_platform",
        SimpleNamespace(is_device_capability=lambda capability: True),
    )
    monkeypatch.setattr(common, "has_nvidia_artifactory", lambda: True)

    assert common.use_flashinfer_prefill() is True
    assert common.use_cudnn_prefill() is False

    monkeypatch.setattr(common.envs, "VLLM_USE_CUDNN_PREFILL", True, raising=False)

    assert common.use_flashinfer_prefill() is False
    assert common.use_cudnn_prefill() is True


def test_mla_common_impl_defaults_dcp_world_size_when_group_missing(monkeypatch):
    import vllm_kunlun

    vllm_kunlun.register()
    from vllm_kunlun.v1.attention.backends.mla import common

    class FakeMLAImpl(common.MLACommonImpl):
        def _forward_decode(self, *_args, **_kwargs):
            raise NotImplementedError

    def raise_missing_dcp_group():
        raise AssertionError

    monkeypatch.setattr(common, "get_dcp_group", raise_missing_dcp_group)
    monkeypatch.setattr(common, "use_flashinfer_prefill", lambda: False)
    monkeypatch.setattr(common, "use_cudnn_prefill", lambda: False)
    monkeypatch.setattr(
        common.MLACommonMetadataBuilder,
        "determine_chunked_prefill_workspace_size",
        staticmethod(lambda _config: 1),
    )
    monkeypatch.setattr(common, "get_current_vllm_config", lambda: SimpleNamespace())

    impl = FakeMLAImpl(
        num_heads=1,
        head_size=6,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        qk_head_dim=4,
        v_head_dim=4,
        kv_b_proj=SimpleNamespace(),
    )

    assert impl.dcp_world_size == 1


def test_mla_common_torch_prefill_attention_matches_reference():
    import vllm_kunlun

    vllm_kunlun.register()
    from vllm_kunlun.v1.attention.backends.mla import common

    class FakeMLAImpl(common.MLACommonImpl):
        def _forward_decode(self, *_args, **_kwargs):
            raise NotImplementedError

    impl = object.__new__(FakeMLAImpl)
    impl._pad_v = True

    q = torch.tensor(
        [
            [[0.2, -0.1, 0.3], [0.5, 0.1, -0.4]],
            [[-0.3, 0.4, 0.2], [0.2, -0.5, 0.6]],
            [[0.1, 0.3, -0.2], [-0.2, 0.4, 0.5]],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [[0.4, -0.2, 0.1], [0.3, 0.2, -0.1]],
            [[-0.2, 0.5, 0.3], [0.1, -0.4, 0.6]],
            [[0.6, 0.1, -0.3], [-0.5, 0.2, 0.4]],
        ],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [
            [[0.7, -0.1], [0.2, 0.5]],
            [[-0.4, 0.3], [0.6, -0.2]],
            [[0.1, 0.8], [-0.3, 0.4]],
        ],
        dtype=torch.float32,
    )
    seq_lens = torch.tensor([0, 2, 3], dtype=torch.int32)

    out, lse = impl._torch_prefill_attention(
        q=q,
        k=k,
        v=v,
        context_seq_lod_xpu=None,
        context_seq_lod_cpu=seq_lens,
        return_softmax_lse=True,
        causal=True,
        softmax_scale=0.5,
    )

    padded_v = F.pad(v, [0, q.shape[-1] - v.shape[-1]], value=0)
    expected_out = torch.empty_like(q)
    expected_lse = torch.full((q.size(1), q.size(0)), float("-inf"))
    for start, end in [(0, 2), (2, 3)]:
        scores = torch.einsum("qhd,khd->hqk", q[start:end], k[start:end]) * 0.5
        mask = torch.triu(
            torch.ones(end - start, end - start, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        expected_out[start:end] = torch.einsum(
            "hqk,khd->qhd", probs, padded_v[start:end]
        )
        expected_lse[:, start:end] = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(lse, expected_lse)


def test_import_hook_installs_kunlun_fused_moe_override():
    from vllm.model_executor.custom_op import op_registry_oot

    import vllm_kunlun

    vllm_kunlun.register()
    vllm_kunlun._custom_import("vllm.model_executor.layers.fused_moe.layer")

    override_cls = op_registry_oot["UnquantizedFusedMoEMethod"]
    assert override_cls.__module__ == "vllm_kunlun.ops.fused_moe.layer"


def test_kunlun_unquantized_fused_moe_method_forces_kunlun_monolithic_entry():
    from vllm_kunlun.ops.fused_moe.layer import KunlunUnquantizedFusedMoEMethod

    with patch(
        "vllm_kunlun.ops.fused_moe.layer.UnquantizedFusedMoEMethod.__init__",
        autospec=True,
        side_effect=lambda self, moe: setattr(self, "_is_monolithic", False),
    ):
        method = KunlunUnquantizedFusedMoEMethod(SimpleNamespace())

    assert method.is_monolithic is True
    assert (
        method.apply_monolithic.__func__
        is KunlunUnquantizedFusedMoEMethod._apply_monolithic_kunlun
    )


def test_kunlun_fused_moe_native_fallback_matches_reference():
    module_name = "vllm_kunlun.ops._kunlun_ops"
    backups = {
        name: sys.modules.get(name)
        for name in [module_name, "cocopod", "xspeedgate_ops"]
    }

    sys.modules.pop(module_name, None)

    try:
        with patch.dict(
            sys.modules,
            {
                "cocopod": types.ModuleType("cocopod"),
                "xspeedgate_ops": types.ModuleType("xspeedgate_ops"),
            },
        ):
            from vllm_kunlun.ops._kunlun_ops import KunlunOps

            hidden_states = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
            w1 = torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.5, 0.0],
                        [0.0, 0.5],
                    ],
                    [
                        [0.3, 0.2],
                        [0.4, -0.1],
                        [0.2, 0.1],
                        [-0.5, 0.3],
                    ],
                ],
                dtype=torch.float32,
            )
            w2 = torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    [
                        [0.5, -0.2],
                        [0.1, 0.4],
                    ],
                ],
                dtype=torch.float32,
            )
            router_logits = torch.tensor([[3.0, 1.0]], dtype=torch.float32)

            result = KunlunOps.fused_moe(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                router_logits=router_logits,
                ep_rank=0,
                moe_top_k=2,
                renormalize=True,
            )
    finally:
        for name, module in backups.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    topk_logits, topk_ids = torch.topk(router_logits, k=2, dim=-1, sorted=False)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    expected = torch.zeros_like(hidden_states)

    for token_idx in range(hidden_states.shape[0]):
        token_hidden_state = hidden_states[token_idx : token_idx + 1]
        for route_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, route_idx])
            gate_up = token_hidden_state @ w1[expert_id].transpose(0, 1)
            half = gate_up.shape[-1] // 2
            activated = F.silu(gate_up[..., :half]) * gate_up[..., half:]
            expert_output = activated @ w2[expert_id].transpose(0, 1)
            expected[token_idx] += topk_weights[
                token_idx, route_idx
            ] * expert_output.squeeze(0)

    torch.testing.assert_close(result, expected)


def test_kunlun_fused_moe_native_fallback_maps_global_expert_ids_to_local_weights():
    module_name = "vllm_kunlun.ops._kunlun_ops"
    backups = {
        name: sys.modules.get(name)
        for name in [module_name, "cocopod", "xspeedgate_ops"]
    }

    sys.modules.pop(module_name, None)

    try:
        with patch.dict(
            sys.modules,
            {
                "cocopod": types.ModuleType("cocopod"),
                "xspeedgate_ops": types.ModuleType("xspeedgate_ops"),
            },
        ):
            from vllm_kunlun.ops._kunlun_ops import KunlunOps

            hidden_states = torch.tensor(
                [
                    [0.5, -1.0],
                    [1.5, 0.25],
                ],
                dtype=torch.float32,
            )
            w1 = torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.5, 0.0],
                        [0.0, 0.5],
                    ],
                    [
                        [0.6, -0.2],
                        [0.3, 0.4],
                        [0.2, 0.1],
                        [-0.4, 0.3],
                    ],
                ],
                dtype=torch.float32,
            )
            w2 = torch.tensor(
                [
                    [
                        [0.7, 0.0],
                        [0.0, 0.9],
                    ],
                    [
                        [0.4, -0.1],
                        [0.2, 0.5],
                    ],
                ],
                dtype=torch.float32,
            )
            router_logits = torch.tensor(
                [
                    [0.1, -0.8, 4.0, 2.0],
                    [-0.5, 0.2, 1.2, 3.8],
                ],
                dtype=torch.float32,
            )

            result = KunlunOps.fused_moe(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                router_logits=router_logits,
                ep_rank=1,
                moe_top_k=2,
                renormalize=True,
            )
    finally:
        for name, module in backups.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    topk_logits, topk_ids = torch.topk(router_logits, k=2, dim=-1, sorted=False)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    expected = torch.zeros_like(hidden_states)
    local_expert_offset = w1.shape[0]

    for token_idx in range(hidden_states.shape[0]):
        token_hidden_state = hidden_states[token_idx : token_idx + 1]
        for route_idx in range(topk_ids.shape[1]):
            local_expert_id = int(topk_ids[token_idx, route_idx]) - local_expert_offset
            gate_up = token_hidden_state @ w1[local_expert_id].transpose(0, 1)
            half = gate_up.shape[-1] // 2
            activated = F.silu(gate_up[..., :half]) * gate_up[..., half:]
            expert_output = activated @ w2[local_expert_id].transpose(0, 1)
            expected[token_idx] += topk_weights[
                token_idx, route_idx
            ] * expert_output.squeeze(0)

    torch.testing.assert_close(result, expected)


def test_import_hook_overrides_vllm_compilation_wrapper():
    import vllm_kunlun

    vllm_kunlun.import_hook()

    assert __import__ is vllm_kunlun._custom_import


def test_import_hook_maps_merge_attn_states_aliases():
    import vllm_kunlun

    target_module_name = "vllm_kunlun.ops.attention.merge_attn_states"
    alias_names = [
        "vllm.attention.ops.merge_attn_states",
        "vllm.v1.attention.ops.merge_attn_states",
    ]
    sentinel_module = types.ModuleType(target_module_name)
    module_names = [target_module_name, *alias_names]
    backups = {
        module_name: sys.modules.get(module_name) for module_name in module_names
    }

    for module_name in module_names:
        sys.modules.pop(module_name, None)

    try:
        with patch.object(
            vllm_kunlun.importlib,
            "import_module",
            return_value=sentinel_module,
        ) as mock_import_module:
            with patch.object(
                vllm_kunlun,
                "OLD_IMPORT_HOOK",
                side_effect=AssertionError("mapped imports must not fall back"),
            ):
                for alias_name in alias_names:
                    imported_module = vllm_kunlun._custom_import(
                        alias_name, fromlist=["merge_attn_states"]
                    )
                    assert imported_module is sentinel_module
                    assert sys.modules[alias_name] is sentinel_module

        assert sys.modules[target_module_name] is sentinel_module
        assert mock_import_module.call_count == len(alias_names)
    finally:
        for module_name, module in backups.items():
            if module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module


def test_import_hook_maps_attention_backend_abstract_alias():
    import vllm_kunlun

    target_module_name = "vllm.v1.attention.backend"
    alias_name = "vllm.attention.backends.abstract"
    sentinel_module = types.ModuleType(target_module_name)
    backups = {
        module_name: sys.modules.get(module_name)
        for module_name in [target_module_name, alias_name]
    }

    for module_name in [target_module_name, alias_name]:
        sys.modules.pop(module_name, None)

    try:
        with patch.object(
            vllm_kunlun.importlib,
            "import_module",
            return_value=sentinel_module,
        ) as mock_import_module:
            with patch.object(
                vllm_kunlun,
                "OLD_IMPORT_HOOK",
                side_effect=AssertionError("mapped imports must not fall back"),
            ):
                imported_module = vllm_kunlun._custom_import(
                    alias_name,
                    fromlist=["AttentionBackend"],
                )

        assert imported_module is sentinel_module
        assert sys.modules[alias_name] is sentinel_module
        assert sys.modules[target_module_name] is sentinel_module
        assert mock_import_module.call_count == 1
    finally:
        for module_name, module in backups.items():
            if module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module


def test_import_hook_maps_attention_ops_aliases():
    import vllm_kunlun

    alias_pairs = [
        ("vllm.attention.ops.common", "vllm.v1.attention.ops.common"),
        ("vllm.attention.ops.flashmla", "vllm.v1.attention.ops.flashmla"),
    ]
    module_names = [module_name for pair in alias_pairs for module_name in pair]
    backups = {
        module_name: sys.modules.get(module_name) for module_name in module_names
    }

    for module_name in module_names:
        sys.modules.pop(module_name, None)

    try:
        with patch.object(vllm_kunlun.importlib, "import_module") as mock_import_module:
            with patch.object(
                vllm_kunlun,
                "OLD_IMPORT_HOOK",
                side_effect=AssertionError("mapped imports must not fall back"),
            ):
                for alias_name, target_module_name in alias_pairs:
                    sentinel_module = types.ModuleType(target_module_name)
                    mock_import_module.return_value = sentinel_module

                    imported_module = vllm_kunlun._custom_import(
                        alias_name, fromlist=["sentinel"]
                    )

                    assert imported_module is sentinel_module
                    assert sys.modules[alias_name] is sentinel_module
                    assert sys.modules[target_module_name] is sentinel_module
                    mock_import_module.assert_called_with(target_module_name)

        assert mock_import_module.call_count == len(alias_pairs)
    finally:
        for module_name, module in backups.items():
            if module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module


def test_merge_attn_states_import_is_lazy():
    import vllm_kunlun

    module_names = [
        "kunlun_ops",
        "vllm_kunlun.ops",
        "vllm_kunlun.ops._custom_ops",
        "vllm_kunlun.ops.attention",
        "vllm_kunlun.ops.attention.merge_attn_states",
        "vllm.v1.attention.ops.merge_attn_states",
    ]

    for module_name in module_names:
        sys.modules.pop(module_name, None)

    imported_module = vllm_kunlun._custom_import(
        "vllm.v1.attention.ops.merge_attn_states",
        fromlist=["merge_attn_states"],
    )

    assert imported_module.__name__ == "vllm_kunlun.ops.attention.merge_attn_states"
    assert "kunlun_ops" not in sys.modules
    assert "vllm_kunlun.ops._custom_ops" not in sys.modules


def test_topk_topp_sampler_import_is_lazy():
    import vllm_kunlun

    module_names = [
        "kunlun_ops",
        "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
        "vllm.v1.sample.ops.topk_topp_sampler",
    ]
    backups = {
        module_name: sys.modules.get(module_name) for module_name in module_names
    }

    for module_name in module_names:
        sys.modules.pop(module_name, None)

    try:
        imported_module = vllm_kunlun._custom_import(
            "vllm.v1.sample.ops.topk_topp_sampler",
            fromlist=["TopKTopPSampler"],
        )

        assert imported_module.__name__ == "vllm_kunlun.v1.sample.ops.topk_topp_sampler"
        assert "kunlun_ops" not in sys.modules
    finally:
        for module_name, module in backups.items():
            if module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module


def test_topk_topp_sampler_falls_back_when_kunlun_ops_import_fails():
    sampler_module = importlib.import_module(
        "vllm_kunlun.v1.sample.ops.topk_topp_sampler"
    )
    sampler = sampler_module.TopKTopPSampler("raw_logprobs")
    logits = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    top_k = torch.tensor([2], dtype=torch.int32)
    expected_ids = torch.tensor([1], dtype=torch.int64)

    with patch.object(
        sampler_module.importlib,
        "import_module",
        side_effect=ImportError("kunlun ops unavailable"),
    ) as mock_import_module:
        with patch.object(
            sampler,
            "forward_native",
            return_value=(expected_ids, None),
        ) as mock_forward_native:
            token_ids, processed = sampler.forward_kunlun(logits, {}, top_k, None)
            sampler.forward_kunlun(logits, {}, top_k, None)

    assert mock_import_module.call_count == 1
    assert mock_forward_native.call_count == 2
    assert torch.equal(token_ids, expected_ids)
    assert processed is None
    assert sampler._disable_kunlun_sampling is True


def test_quantization_kernels_import_registers_oot_entries():
    from vllm.model_executor.kernels.linear import (
        _POSSIBLE_INT8_KERNELS,
        _POSSIBLE_KERNELS,
    )
    from vllm.platforms import PlatformEnum

    kernels_module = importlib.import_module("vllm_kunlun.quantization.kernels")

    assert (
        kernels_module.KunlunScaledMMLinearKernel
        in _POSSIBLE_INT8_KERNELS[PlatformEnum.OOT]
    )
    assert (
        kernels_module.KunlunExllamaLinearKernel in _POSSIBLE_KERNELS[PlatformEnum.OOT]
    )


def test_attention_compat_exports_vllm019_attention():
    compat_module = importlib.import_module("vllm_kunlun.attention_compat")
    from vllm.model_executor.layers.attention import Attention as UpstreamAttention

    assert compat_module.Attention is UpstreamAttention
    assert compat_module.check_upstream_fa_availability(torch.float32) is False
    assert isinstance(
        compat_module.check_upstream_fa_availability(torch.float16),
        bool,
    )


def test_kunlun_platform_uses_parallel_config_all2all_backend():
    from vllm.config import CUDAGraphMode

    from vllm_kunlun.platforms.kunlun import KunlunPlatform

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_cls="worker",
            data_parallel_size=2,
            all2all_backend="deepep_high_throughput",
        ),
        model_config=SimpleNamespace(use_mla=False, enforce_eager=False),
        speculative_config=None,
        cache_config=SimpleNamespace(block_size=16),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            custom_ops=[],
            pass_config=SimpleNamespace(enable_fusion=True),
            backend="inductor",
        ),
    )

    KunlunPlatform.check_and_update_config(vllm_config)

    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert vllm_config.model_config.enforce_eager is True
    assert vllm_config.compilation_config.backend == "eager"


def test_kunlun_platform_get_attn_backend_cls_accepts_num_heads():
    from vllm_kunlun.platforms.kunlun import KunlunPlatform

    attn_selector_config = SimpleNamespace(use_mla=False, use_sparse=False)

    with patch(
        "vllm_kunlun.platforms.kunlun.importlib.import_module",
        return_value=object(),
    ):
        backend_cls = KunlunPlatform.get_attn_backend_cls(
            None,
            attn_selector_config=attn_selector_config,
            num_heads=32,
        )

    assert backend_cls.endswith("KunlunAttentionBackend")


def test_kunlun_platform_get_attn_backend_cls_falls_back_to_triton():
    from vllm_kunlun.platforms.kunlun import KunlunPlatform

    attn_selector_config = SimpleNamespace(use_mla=False, use_sparse=False)

    with patch(
        "vllm_kunlun.platforms.kunlun.importlib.import_module",
        side_effect=OSError("kunlun attention backend unavailable"),
    ):
        backend_cls = KunlunPlatform.get_attn_backend_cls(
            None,
            attn_selector_config=attn_selector_config,
            num_heads=32,
        )

    assert backend_cls.endswith("TritonAttentionBackend")


def test_kunlun_platform_preregistration_skips_compressed_tensors_failures():
    import builtins

    from vllm_kunlun.platforms.kunlun import KunlunPlatform

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_kunlun.quantization.compressed_tensors":
            raise ImportError("compressed tensors unavailable in test")
        return original_import(name, globals, locals, fromlist, level)

    with patch.object(builtins, "__import__", side_effect=guarded_import):
        KunlunPlatform.pre_register_and_update()


def test_qwen35_processing_info_accepts_upstream_hf_config():
    from vllm.multimodal.processing.context import InputProcessingContext
    from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig

    from vllm_kunlun.hf_config_compat import QWEN3_5_MOE_CONFIG_TYPES

    ctx = SimpleNamespace(model_config=SimpleNamespace(hf_config=Qwen3_5MoeConfig()))
    ctx.get_hf_config = InputProcessingContext.get_hf_config.__get__(ctx, type(ctx))
    assert isinstance(ctx.get_hf_config(QWEN3_5_MOE_CONFIG_TYPES), Qwen3_5MoeConfig)


def test_wrapper_can_initialize_and_run_twice():
    wrapper_module = _load_wrapper_module()
    mock_config = _make_mock_config(wrapper_module)

    class TestWrapper(wrapper_module.TorchCompileWithNoGuardsWrapper):
        def forward(self, x):
            return x * 2

    with patch.object(
        wrapper_module, "get_current_vllm_config", return_value=mock_config
    ):
        with patch.object(wrapper_module.envs, "VLLM_USE_BYTECODE_HOOK", False):
            with patch.object(wrapper_module.envs, "VLLM_USE_AOT_COMPILE", False):
                with patch.object(
                    wrapper_module.torch,
                    "compile",
                    side_effect=lambda fn, **kwargs: fn,
                ):
                    wrapper = TestWrapper()
                    input_tensor = torch.tensor([1.0, 2.0, 3.0])

                    assert torch.allclose(wrapper(input_tensor), input_tensor * 2)
                    assert torch.allclose(wrapper(input_tensor), input_tensor * 2)
                    assert wrapper.first_compile is False


def test_bytecode_hook_ignores_unrelated_code_objects():
    wrapper_module = _load_wrapper_module()
    mock_config = _make_mock_config(wrapper_module)

    class TestWrapper(wrapper_module.TorchCompileWithNoGuardsWrapper):
        def forward(self, x):
            return x

    with patch.object(
        wrapper_module, "get_current_vllm_config", return_value=mock_config
    ):
        with patch.object(wrapper_module.envs, "VLLM_USE_BYTECODE_HOOK", True):
            with patch.object(wrapper_module.envs, "VLLM_USE_AOT_COMPILE", False):
                with patch.object(
                    wrapper_module.torch,
                    "compile",
                    side_effect=lambda fn, **kwargs: fn,
                ):
                    with patch.object(
                        wrapper_module.torch._dynamo.convert_frame,
                        "register_bytecode_hook",
                    ):
                        wrapper = TestWrapper()
                        wrong_code = (lambda: None).__code__
                        new_code = (lambda: None).__code__

                        wrapper.bytecode_hook(wrong_code, new_code)

                        assert wrapper._compiled_bytecode is None


def test_qwen3_moe_loader_compat_patch_skips_unmatched_expert_weights(
    monkeypatch,
):
    from vllm_kunlun.compat import apply_qwen3_moe_loader_compat_patch

    seen_weight_names = []

    class FakeQwen3MoeModel:
        def named_parameters(self):
            return [
                ("layers.47.mlp.experts.w2_weight", object()),
            ]

        def get_expert_mapping(self):
            return [
                (
                    "layers.47.mlp.experts.w2_weight",
                    "layers.47.mlp.experts.0.down_proj.weight",
                    0,
                    "w2",
                ),
                (
                    "layers.48.mlp.experts.w2_weight",
                    "layers.48.mlp.experts.0.down_proj.weight",
                    0,
                    "w2",
                ),
            ]

        def load_weights(self, weights):
            seen_weight_names.extend(name for name, _ in weights)
            return set(seen_weight_names)

    fake_module = type("FakeModule", (), {"Qwen3MoeModel": FakeQwen3MoeModel})
    monkeypatch.setattr("vllm_kunlun.compat.import_module", lambda _: fake_module)

    apply_qwen3_moe_loader_compat_patch()

    model = FakeQwen3MoeModel()
    loaded = model.load_weights(
        [
            ("layers.47.mlp.experts.0.down_proj.weight", torch.zeros(1)),
            ("layers.48.mlp.experts.0.down_proj.weight", torch.zeros(1)),
            ("layers.48.self_attn.q_proj.weight", torch.zeros(1)),
        ]
    )

    assert "layers.47.mlp.experts.0.down_proj.weight" in seen_weight_names
    assert "layers.48.mlp.experts.0.down_proj.weight" not in seen_weight_names
    assert "layers.48.self_attn.q_proj.weight" in seen_weight_names
    assert "layers.47.mlp.experts.0.down_proj.weight" in loaded


def test_mla_attention_weight_device_patch_runs_after_processing(monkeypatch):
    from vllm_kunlun import compat

    class FakeMLAAttention:
        def __init__(self):
            self.called = None
            self.W_UK_T = torch.zeros(1, dtype=torch.float16)

        def process_weights_after_loading(self, act_dtype):
            self.called = act_dtype
            return "processed"

        def forward_impl(self, q, marker=None):
            return ("forwarded", q, marker)

    fake_module = SimpleNamespace(MLAAttention=FakeMLAAttention)
    moved = []
    device = torch.device("cpu")

    monkeypatch.setattr(compat, "_current_cuda_device", lambda: device)
    monkeypatch.setattr(
        compat,
        "_move_mla_runtime_tensor_attrs",
        lambda instance, target_device: moved.append((instance, target_device)),
    )

    compat._patch_mla_attention_runtime_weight_device(fake_module)
    attention = FakeMLAAttention()

    assert attention.process_weights_after_loading(torch.bfloat16) == "processed"
    assert attention.called is torch.bfloat16
    assert moved == [(attention, device)]
    assert getattr(
        FakeMLAAttention.process_weights_after_loading,
        "_vllm_kunlun_patched",
        False,
    )

    q = torch.empty(1, dtype=torch.float32)
    result, runtime_q, marker = attention.forward_impl(q, marker="x")
    assert result == "forwarded"
    assert runtime_q.dtype is torch.float16
    assert marker == "x"
    assert moved[-1] == (attention, q.device)
    assert getattr(
        FakeMLAAttention.forward_impl,
        "_vllm_kunlun_patched",
        False,
    )


def test_mla_attention_weight_patch_copies_runtime_output_to_original_dtype():
    from vllm_kunlun import compat

    class FakeMLAAttention:
        def __init__(self):
            self.W_UK_T = torch.zeros(1, dtype=torch.float16)

        def forward_impl(self, q, output=None):
            assert q.dtype is torch.float16
            assert output.dtype is torch.float16
            output.fill_(2)
            return output

    fake_module = SimpleNamespace(MLAAttention=FakeMLAAttention)
    compat._patch_mla_attention_runtime_weight_device(fake_module)

    attention = FakeMLAAttention()
    q = torch.empty(1, dtype=torch.float32)
    output = torch.empty(1, dtype=torch.float32)

    result = attention.forward_impl(q, output=output)

    assert result is output
    assert output.dtype is torch.float32
    assert output.item() == 2


def test_custom_qwen3_moe_model_skips_unmatched_expert_weights(monkeypatch):
    from vllm_kunlun.models.qwen3_moe import Qwen3MoeModel

    model = object.__new__(Qwen3MoeModel)
    model.quant_config = None

    loaded_names = []

    class FakeParam:
        def weight_loader(
            self,
            _param,
            _loaded_weight,
            name_mapped,
            shard_id=None,
            expert_id=None,
            return_success=False,
        ):
            loaded_names.append((name_mapped, shard_id, expert_id))
            if return_success:
                return True
            return None

    monkeypatch.setattr(
        "vllm_kunlun.models.qwen3_moe.upstream.is_pp_missing_parameter",
        lambda *_args, **_kwargs: False,
    )

    model.named_parameters = lambda: [
        ("model.layers.47.mlp.experts.w2_weight", FakeParam()),
    ]
    model.get_expert_mapping = lambda: [
        (
            "layers.47.mlp.experts.w2_weight",
            "layers.47.mlp.experts.0.down_proj.weight",
            0,
            "w2",
        ),
        (
            "layers.48.mlp.experts.w2_weight",
            "layers.48.mlp.experts.0.down_proj.weight",
            0,
            "w2",
        ),
    ]

    loaded = Qwen3MoeModel.load_weights(
        model,
        [
            ("model.layers.47.mlp.experts.0.down_proj.weight", torch.zeros(1)),
            ("model.layers.48.mlp.experts.0.down_proj.weight", torch.zeros(1)),
        ],
    )

    assert loaded_names == [("model.layers.47.mlp.experts.w2_weight", "w2", 0)]
    assert "model.layers.47.mlp.experts.w2_weight" in loaded
    assert all("layers.48" not in name for name in loaded)


def test_fused_moe_expert_mapping_accepts_vllm019_model_arg():
    from vllm_kunlun.models.fused_moe_compat import make_expert_params_mapping

    class FakeModel(torch.nn.Module):
        pass

    class FakeNewFusedMoE:
        @classmethod
        def make_expert_params_mapping(cls, model, **kwargs):
            return [("new", model, kwargs["num_experts"])]

    model = FakeModel()

    assert make_expert_params_mapping(
        FakeNewFusedMoE,
        model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=4,
    ) == [("new", model, 4)]


def test_fused_moe_expert_mapping_keeps_legacy_signature():
    from vllm_kunlun.models.fused_moe_compat import make_expert_params_mapping

    class FakeOldFusedMoE:
        @classmethod
        def make_expert_params_mapping(cls, **kwargs):
            return [("old", kwargs["num_experts"])]

    assert make_expert_params_mapping(
        FakeOldFusedMoE,
        torch.nn.Module(),
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=4,
    ) == [("old", 4)]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
