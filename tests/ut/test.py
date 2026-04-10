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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
