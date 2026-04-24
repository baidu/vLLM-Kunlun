"""Compatibility helpers for running vLLM 0.19.x on PyTorch 2.5.1."""

from __future__ import annotations

import logging
import pickle
import sys
import types
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any, Iterable

import torch

_MISSING_OUTPUT_NODE_WARNING_EMITTED = False
_MUTATED_OUTPUT_DEBUG_EMITTED = False
_MISSING_EXAMPLE_VALUE_DEBUG_EMITTED = False


def _ensure_custom_graph_pass_module() -> None:
    """Provide torch._inductor.custom_graph_pass on older PyTorch builds."""
    module_name = "torch._inductor.custom_graph_pass"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)

    class CustomGraphPass:
        """Minimal fallback base class used by vLLM Inductor passes."""

        def __call__(self, graph: Any) -> None:
            return None

        def uuid(self) -> str:
            return self.__class__.__name__

    module.CustomGraphPass = CustomGraphPass
    sys.modules[module_name] = module


def _ensure_graph_pickler_module() -> None:
    """Backfill torch.fx._graph_pickler for PyTorch 2.5."""
    module_name = "torch.fx._graph_pickler"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)

    @dataclass
    class Options:
        ops_filter: Any = None

    class GraphPickler:
        def reducer_override(self, obj: Any) -> Any:
            return NotImplemented

        @staticmethod
        def dumps(obj: Any, options: Any | None = None) -> bytes:
            del options
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

        @staticmethod
        def loads(data: bytes, fake_mode: Any | None = None) -> Any:
            del fake_mode
            return pickle.loads(data)

    module.GraphPickler = GraphPickler
    module.Options = Options
    sys.modules[module_name] = module


def _ensure_mistral_reasoning_effort_enum() -> None:
    """Backfill mistral_common's ReasoningEffort for older releases."""
    try:
        request_module = import_module("mistral_common.protocol.instruct.request")
    except ImportError:
        return

    if hasattr(request_module, "ReasoningEffort"):
        return

    class ReasoningEffort(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    request_module.ReasoningEffort = ReasoningEffort


def _patch_v1_block_table_triton_kernel(module: types.ModuleType | None = None) -> None:
    """Replace Triton's slot-mapping kernel with a torch fallback."""
    if module is None:
        module = import_module("vllm.v1.worker.block_table")

    existing_kernel = getattr(module, "_compute_slot_mapping_kernel", None)
    if getattr(existing_kernel, "_vllm_kunlun_patched", False):
        return

    def _compute_slot_mapping_kernel_impl(
        num_tokens: int,
        max_num_tokens: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        block_table: torch.Tensor,
        block_table_stride: int,
        block_size: int,
        slot_mapping: torch.Tensor,
        TOTAL_CP_WORLD_SIZE: int,
        TOTAL_CP_RANK: int,
        CP_KV_CACHE_INTERLEAVE_SIZE: int,
        PAD_ID: int,
        BLOCK_SIZE: int,
    ) -> None:
        del block_table_stride, BLOCK_SIZE

        slot_mapping[:max_num_tokens].fill_(PAD_ID)
        if num_tokens == 0:
            return

        virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
        interleave_span = TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE
        num_reqs = query_start_loc.numel() - 1

        for req_idx in range(num_reqs):
            start_idx = int(query_start_loc[req_idx].item())
            end_idx = int(query_start_loc[req_idx + 1].item())
            if end_idx <= start_idx:
                continue

            pos = positions[start_idx:end_idx]
            block_indices = torch.div(pos, virtual_block_size, rounding_mode="floor")
            block_numbers = (
                block_table[req_idx]
                .index_select(
                    0,
                    block_indices.to(dtype=torch.long),
                )
                .to(torch.int64)
            )

            virtual_block_offsets = pos - block_indices * virtual_block_size
            is_local = (
                torch.div(
                    virtual_block_offsets,
                    CP_KV_CACHE_INTERLEAVE_SIZE,
                    rounding_mode="floor",
                )
                % TOTAL_CP_WORLD_SIZE
                == TOTAL_CP_RANK
            )
            local_block_offsets = (
                torch.div(
                    virtual_block_offsets,
                    interleave_span,
                    rounding_mode="floor",
                )
                * CP_KV_CACHE_INTERLEAVE_SIZE
                + virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
            )
            slot_ids = block_numbers * block_size + local_block_offsets
            slot_mapping[start_idx:end_idx] = torch.where(
                is_local,
                slot_ids,
                torch.full_like(slot_ids, PAD_ID),
            )

    class _ComputeSlotMappingKernelWrapper:
        _vllm_kunlun_patched = True

        def __getitem__(self, grid):
            del grid
            return _compute_slot_mapping_kernel_impl

    module._compute_slot_mapping_kernel = _ComputeSlotMappingKernelWrapper()


def _ensure_fx_graph_output_node() -> None:
    """Backfill torch.fx.Graph.output_node for PyTorch 2.5."""
    graph_cls = torch.fx.Graph
    if hasattr(graph_cls, "output_node"):
        return

    def output_node(self: torch.fx.Graph) -> torch.fx.Node:
        for node in reversed(tuple(self.nodes)):
            if node.op == "output":
                return node
        raise RuntimeError("Graph has no output node")

    output_node._vllm_kunlun_patched = True  # type: ignore[attr-defined]
    graph_cls.output_node = output_node  # type: ignore[attr-defined]


def _get_fx_graph_outputs(graph: torch.fx.Graph) -> Any:
    """Extract the output value(s) from an FX graph across PyTorch variants."""
    global _MISSING_OUTPUT_NODE_WARNING_EMITTED

    nodes = tuple(graph.nodes)
    if not nodes:
        raise RuntimeError("Graph has no nodes")

    last_node = nodes[-1]
    if last_node.op == "output":
        return last_node.args[0]

    mutated_outputs = _get_mutated_node_outputs(last_node)
    if mutated_outputs is not None:
        if not _MISSING_OUTPUT_NODE_WARNING_EMITTED:
            logging.getLogger("vllm_kunlun.compat").warning(
                "FX graph is missing an explicit output node; "
                "falling back to mutated output args of %s (%s)",
                last_node.name,
                last_node.op,
            )
            _MISSING_OUTPUT_NODE_WARNING_EMITTED = True
        return mutated_outputs

    if not _MISSING_OUTPUT_NODE_WARNING_EMITTED:
        logging.getLogger("vllm_kunlun.compat").warning(
            "FX graph is missing an explicit output node; "
            "falling back to the last node %s (%s)",
            last_node.name,
            last_node.op,
        )
        _MISSING_OUTPUT_NODE_WARNING_EMITTED = True

    return last_node


def _normalize_mutated_outputs(mutated_args: list[Any]) -> Any | None:
    if not mutated_args:
        return None
    if len(mutated_args) == 1:
        return mutated_args[0]
    return tuple(mutated_args)


def _collect_mutated_arg_candidates(
    node: torch.fx.Node,
    indices: Iterable[int],
    names: Iterable[str] = (),
) -> Any | None:
    mutated_args: list[Any] = []
    seen: set[int] = set()

    for index in indices:
        if index >= len(node.args):
            continue
        value = node.args[index]
        if value is None or id(value) in seen:
            continue
        mutated_args.append(value)
        seen.add(id(value))

    for name in names:
        value = node.kwargs.get(name)
        if value is None or id(value) in seen:
            continue
        mutated_args.append(value)
        seen.add(id(value))

    return _normalize_mutated_outputs(mutated_args)


def _describe_fx_arg(arg: Any) -> str:
    if isinstance(arg, torch.fx.Node):
        return f"{arg.name}:{arg.op}"
    return type(arg).__name__


def _maybe_log_mutated_output_debug(
    node: torch.fx.Node,
    target: Any,
    schema: Any,
) -> None:
    global _MUTATED_OUTPUT_DEBUG_EMITTED

    target_repr = str(target).lower()
    if (
        _MUTATED_OUTPUT_DEBUG_EMITTED
        or "unified_attention_with_output" not in target_repr
    ):
        return

    logging.getLogger("vllm_kunlun.compat").warning(
        "Failed to infer mutated outputs for %s (%s): target_type=%s target=%r "
        "schema=%s args=%s kwargs=%s",
        node.name,
        node.op,
        type(target).__name__,
        target,
        schema,
        [_describe_fx_arg(arg) for arg in node.args],
        sorted(node.kwargs),
    )
    _MUTATED_OUTPUT_DEBUG_EMITTED = True


def _maybe_log_missing_example_value_debug(
    target: str,
    outputs: Any,
) -> None:
    global _MISSING_EXAMPLE_VALUE_DEBUG_EMITTED

    if _MISSING_EXAMPLE_VALUE_DEBUG_EMITTED:
        return

    def describe_output(value: Any) -> str:
        if isinstance(value, torch.fx.Node):
            meta_keys = sorted(value.meta)
            return f"{value.name}:{value.op}:meta={meta_keys}"
        return repr(value)

    logging.getLogger("vllm_kunlun.compat").warning(
        "Missing example_value while materializing outputs for submodule %s: %s",
        target,
        torch.fx.map_arg(outputs, describe_output),
    )
    _MISSING_EXAMPLE_VALUE_DEBUG_EMITTED = True


def _get_mutated_node_outputs(node: torch.fx.Node) -> Any | None:
    """Infer outputs from in-place mutated args when a graph returns None."""
    target = getattr(node, "target", None)
    schema = getattr(target, "_schema", None)
    if schema is not None:
        mutated_args: list[Any] = []
        for index, arg_schema in enumerate(schema.arguments):
            alias_info = getattr(arg_schema, "alias_info", None)
            if alias_info is None or not getattr(alias_info, "is_write", False):
                continue

            if index < len(node.args):
                value = node.args[index]
            else:
                value = node.kwargs.get(arg_schema.name)

            if value is not None:
                mutated_args.append(value)

        normalized_outputs = _normalize_mutated_outputs(mutated_args)
        if normalized_outputs is not None:
            return normalized_outputs

    target_repr = str(target).lower()
    for op_name in (
        "unified_attention_with_output",
        "unified_mla_attention_with_output",
    ):
        if op_name not in target_repr:
            continue
        fallback_outputs = _collect_mutated_arg_candidates(
            node,
            indices=(3, 6),
            names=("output", "output_block_scale"),
        )
        if fallback_outputs is not None:
            return fallback_outputs

    _maybe_log_mutated_output_debug(node, target, schema)
    return None


def _ensure_compiler_set_stance() -> None:
    """Backfill torch.compiler.set_stance for PyTorch 2.5."""
    if hasattr(torch.compiler, "set_stance"):
        return

    def set_stance(_stance: str):
        return nullcontext()

    torch.compiler.set_stance = set_stance  # type: ignore[attr-defined]


def _ensure_functorch_config_keys() -> None:
    """Expose vLLM 0.19.x functorch config keys on PyTorch 2.5."""
    config_module = import_module("torch._functorch.config")
    missing_defaults = {
        "bundled_autograd_cache": False,
        "autograd_cache_normalize_inputs": False,
        "enable_remote_autograd_cache": False,
        "_cache_config_ignore_prefix": [],
        "_save_config_ignore": [],
        "_compile_ignored_keys": set(),
    }

    for key, default in missing_defaults.items():
        if key in config_module._config:
            continue
        config_module._allowed_keys.add(key)
        value = default.copy() if isinstance(default, (list, dict, set)) else default
        config_module._config[key] = value
        config_module._default[key] = (
            value.copy() if isinstance(value, (list, dict, set)) else value
        )


def _ensure_torch_accelerator_namespace() -> None:
    """Backfill torch.accelerator with a CUDA-backed proxy on PyTorch 2.5."""
    if hasattr(torch, "accelerator"):
        return

    class _CudaAcceleratorProxy:
        def is_available(self) -> bool:
            return torch.cuda.is_available()

        def device_count(self) -> int:
            return torch.cuda.device_count()

        def set_device_index(self, device: Any) -> None:
            torch.cuda.set_device(device)

        def current_device_index(self) -> int:
            return torch.cuda.current_device()

        def empty_cache(self) -> None:
            torch.cuda.empty_cache()

        def synchronize(self, device: Any | None = None) -> None:
            torch.cuda.synchronize(device)

        def memory_stats(self, device: Any | None = None) -> dict[str, Any]:
            return torch.cuda.memory_stats(device)

        def memory_reserved(self, device: Any | None = None) -> int:
            return torch.cuda.memory_reserved(device)

        def max_memory_allocated(self, device: Any | None = None) -> int:
            return torch.cuda.max_memory_allocated(device)

        def reset_peak_memory_stats(self, device: Any | None = None) -> None:
            torch.cuda.reset_peak_memory_stats(device)

        @contextmanager
        def device_index(self, device: Any) -> Any:
            with torch.cuda.device(device):
                yield

        def __getattr__(self, name: str) -> Any:
            return getattr(torch.cuda, name)

    torch.accelerator = _CudaAcceleratorProxy()  # type: ignore[attr-defined]


def _patch_traceable_vllm_parameter_subclasses(module: Any | None = None) -> None:
    """Teach torch 2.5 Dynamo to trace vLLM parameter tensor subclasses."""
    if module is None:
        module = sys.modules.get("vllm.model_executor.parameter")
        if module is None:
            try:
                module = import_module("vllm.model_executor.parameter")
            except Exception:
                return

    traceable_subclasses = getattr(
        torch._dynamo.config, "traceable_tensor_subclasses", None
    )
    if traceable_subclasses is None:
        return

    for class_name in (
        "BasevLLMParameter",
        "ModelWeightParameter",
        "_ColumnvLLMParameter",
        "RowvLLMParameter",
    ):
        parameter_cls = getattr(module, class_name, None)
        if parameter_cls is not None:
            traceable_subclasses.add(parameter_cls)


def _disable_dynamo_torch_function_dispatch() -> None:
    """Use traceable tensor subclasses instead of torch_function dispatch."""
    dynamo_torch_module = import_module("torch._dynamo.variables.torch")
    current = getattr(dynamo_torch_module, "can_dispatch_torch_function", None)
    if current is None or getattr(current, "_vllm_kunlun_patched", False):
        return

    def return_false(*args: Any, **kwargs: Any) -> bool:
        return False

    return_false._vllm_kunlun_patched = True  # type: ignore[attr-defined]
    dynamo_torch_module.can_dispatch_torch_function = return_false


def _patch_piecewise_compile_interpreter_call_module(
    module: Any | None = None,
) -> None:
    """Handle FX graphs without an explicit output node on PyTorch 2.5."""
    if module is None:
        module = sys.modules.get("vllm.compilation.backends")
        if module is None:
            try:
                module = import_module("vllm.compilation.backends")
            except Exception:
                return

    interpreter_cls = getattr(module, "PiecewiseCompileInterpreter", None)
    if interpreter_cls is None:
        return

    current = getattr(interpreter_cls, "call_module", None)
    if current is None or getattr(current, "_vllm_kunlun_patched", False):
        return

    def patched_call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)

        gm = getattr(self.module, target)
        outputs = _get_fx_graph_outputs(gm.graph)
        try:
            output = module.fx.map_arg(outputs, lambda node: node.meta["example_value"])
        except KeyError:
            if isinstance(outputs, torch.fx.Node):
                fallback_outputs = _get_mutated_node_outputs(outputs)
                if fallback_outputs is not None and fallback_outputs is not outputs:
                    outputs = fallback_outputs
                    output = module.fx.map_arg(
                        outputs, lambda node: node.meta["example_value"]
                    )
                else:
                    _maybe_log_missing_example_value_debug(target, outputs)
                    raise
            else:
                _maybe_log_missing_example_value_debug(target, outputs)
                raise

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)

            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]

            from torch._inductor.compile_fx import graph_returns_tuple

            piecewise_backend_module = import_module(
                "vllm.compilation.piecewise_backend"
            )
            piecewise_backend = piecewise_backend_module.PiecewiseBackend(
                submod,
                self.vllm_config,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                self.vllm_backend,
                graph_returns_tuple(submod),
                submod_name=target,
            )

            self.module.__dict__[target] = module.wrap_with_cudagraph_if_needed(
                piecewise_backend,
                self.vllm_config,
                self.compilation_config,
                piecewise_backend.is_first_graph,
                piecewise_backend.is_last_graph,
            )

            module.compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output

    patched_call_module._vllm_kunlun_patched = True  # type: ignore[attr-defined]
    interpreter_cls.call_module = patched_call_module


def apply_qwen3_moe_loader_compat_patch(module: Any | None = None) -> None:
    """Skip extra Qwen3-MoE expert weights that do not map to local params."""
    if module is None:
        module = import_module("vllm.model_executor.models.qwen3_moe")
    model_cls = module.Qwen3MoeModel

    if getattr(model_cls.load_weights, "_vllm_kunlun_patched", False):
        return

    logger = logging.getLogger("vllm_kunlun.compat")
    original_load_weights = model_cls.load_weights

    def patched_load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        missing_weight_names = {
            weight_name
            for param_name, weight_name, *_ in self.get_expert_mapping()
            if param_name not in params_dict
        }
        if not missing_weight_names:
            return original_load_weights(self, weights)

        filtered_weights: list[tuple[str, torch.Tensor]] = []
        skipped_count = 0
        skipped_example: str | None = None
        for name, loaded_weight in weights:
            if any(weight_name in name for weight_name in missing_weight_names):
                skipped_count += 1
                if skipped_example is None:
                    skipped_example = name
                continue
            filtered_weights.append((name, loaded_weight))

        if skipped_count:
            logger.warning(
                "Skipping %d unmatched Qwen3-MoE expert weights. Example: %s",
                skipped_count,
                skipped_example,
            )

        return original_load_weights(self, filtered_weights)

    patched_load_weights._vllm_kunlun_patched = True  # type: ignore[attr-defined]
    model_cls.load_weights = patched_load_weights


def _current_cuda_device() -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    return torch.device("cuda", torch.cuda.current_device())


def _move_mla_runtime_tensor_attrs(
    instance: Any,
    device: torch.device,
) -> None:
    for attr_name in (
        "W_UK_T",
        "W_UV",
        "W_K",
        "W_K_scale",
        "W_V",
        "W_V_scale",
        "W_UK_SCALE",
        "W_UV_SCALE",
    ):
        tensor = getattr(instance, attr_name, None)
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.device != device:
            setattr(instance, attr_name, tensor.to(device=device))


def _get_mla_runtime_compute_dtype(
    instance: Any,
    fallback: torch.dtype,
) -> torch.dtype:
    for attr_name in ("W_UK_T", "W_UV", "W_K", "W_V"):
        tensor = getattr(instance, attr_name, None)
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            return tensor.dtype
    return fallback


def _cast_mla_runtime_tensor(
    tensor: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if tensor.is_floating_point() and tensor.dtype != dtype:
        return tensor.to(dtype=dtype)
    return tensor


def _patch_mla_attention_runtime_weight_device(module: Any | None = None) -> None:
    if module is None:
        module = import_module("vllm.model_executor.layers.attention.mla_attention")

    attention_cls = getattr(module, "MLAAttention", None)
    if attention_cls is None:
        return

    current_process = getattr(attention_cls, "process_weights_after_loading", None)
    if current_process is not None and not getattr(
        current_process,
        "_vllm_kunlun_patched",
        False,
    ):

        def patched_process_weights_after_loading(
            self: Any,
            act_dtype: torch.dtype,
        ) -> Any:
            result = current_process(self, act_dtype)
            device = _current_cuda_device()
            if device is not None:
                _move_mla_runtime_tensor_attrs(self, device)
            return result

        patched_process_weights_after_loading._vllm_kunlun_patched = True  # type: ignore[attr-defined]
        attention_cls.process_weights_after_loading = (
            patched_process_weights_after_loading
        )

    current_forward = getattr(attention_cls, "forward_impl", None)
    if current_forward is not None and not getattr(
        current_forward,
        "_vllm_kunlun_patched",
        False,
    ):

        def patched_forward_impl(self: Any, q: torch.Tensor, *args: Any, **kwargs: Any):
            _move_mla_runtime_tensor_attrs(self, q.device)
            compute_dtype = _get_mla_runtime_compute_dtype(self, q.dtype)
            runtime_q = _cast_mla_runtime_tensor(q, compute_dtype)
            runtime_args = tuple(
                (
                    _cast_mla_runtime_tensor(arg, compute_dtype)
                    if index < 2 and isinstance(arg, torch.Tensor)
                    else arg
                )
                for index, arg in enumerate(args)
            )
            output = kwargs.get("output")
            if (
                isinstance(output, torch.Tensor)
                and output.is_floating_point()
                and output.dtype != compute_dtype
            ):
                runtime_kwargs = dict(kwargs)
                runtime_output = torch.empty_like(output, dtype=compute_dtype)
                runtime_kwargs["output"] = runtime_output
                current_forward(self, runtime_q, *runtime_args, **runtime_kwargs)
                output.copy_(runtime_output.to(dtype=output.dtype))
                return output

            result = current_forward(self, runtime_q, *runtime_args, **kwargs)
            if (
                isinstance(result, torch.Tensor)
                and result.is_floating_point()
                and result.dtype != q.dtype
            ):
                return result.to(dtype=q.dtype)
            return result

        patched_forward_impl._vllm_kunlun_patched = True  # type: ignore[attr-defined]
        attention_cls.forward_impl = patched_forward_impl


def apply_torch251_compat_shims() -> None:
    """Apply the compatibility shims needed by vLLM 0.19.x on torch 2.5.1."""
    _ensure_custom_graph_pass_module()
    _ensure_graph_pickler_module()
    _ensure_mistral_reasoning_effort_enum()
    _ensure_fx_graph_output_node()
    _ensure_compiler_set_stance()
    _ensure_functorch_config_keys()
    _ensure_torch_accelerator_namespace()
    _disable_dynamo_torch_function_dispatch()
    _patch_traceable_vllm_parameter_subclasses()
    _patch_piecewise_compile_interpreter_call_module()
