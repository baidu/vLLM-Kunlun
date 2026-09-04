import logging
import sys
from importlib import import_module
from types import ModuleType


def _import_kunlun_module():
    try:
        import_module("vllm")
    except ModuleNotFoundError:
        # The hook registry only needs vLLM's logger at module import time.
        # Provide the smallest compatible stub so these registry tests can run
        # in a lightweight source-check environment without a vLLM wheel.
        vllm_module = ModuleType("vllm")
        logger_module = ModuleType("vllm.logger")
        logger_module.init_logger = logging.getLogger
        sys.modules["vllm"] = vllm_module
        sys.modules["vllm.logger"] = logger_module
    return import_module("vllm_kunlun")


vllm_kunlun = _import_kunlun_module()


def _registered_hook(target):
    for registered_target, applied, apply in vllm_kunlun._POST_IMPORT_HOOKS:
        if registered_target == target:
            return applied, apply
    raise AssertionError(f"No post-import hook registered for {target}")


def test_minimax_rms_norm_hook_disables_triton():
    applied, apply = _registered_hook(
        "vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp"
    )
    module = ModuleType("minimax_rms_norm_tp")
    module.HAS_TRITON = True

    assert not applied(module)

    apply(module)

    assert module.HAS_TRITON is False
    assert applied(module)


def test_int8_moe_hook_uses_kunlun_generic_backend():
    applied, apply = _registered_hook(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
    )
    module = ModuleType("compressed_tensors_moe_w8a8_int8")
    module.select_int8_moe_backend = object()

    assert not applied(module)

    apply(module)

    assert module._kunlun_select_int8_patched is True
    assert applied(module)
    assert module.select_int8_moe_backend(
        config=object(), weight_key=object(), activation_key=object()
    ) == (None, None)


def test_int8_moe_hook_waits_for_selector_to_be_defined():
    applied, apply = _registered_hook(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
    )
    module = ModuleType("compressed_tensors_moe_w8a8_int8")

    assert not applied(module)

    apply(module)

    assert not hasattr(module, "_kunlun_select_int8_patched")
