"""Regression tests for MLA platform selection and KV-cache binding."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]


def _stub_packages(stub_module, *package_names):
    """Register package-shaped stubs so a module can import its dependencies."""
    for name in package_names:
        module = stub_module(name)
        module.__path__ = []


def _install_mla_backend_stubs(stub_module):
    """Install deterministic MLA backend modules for platform configuration."""
    _stub_packages(
        stub_module,
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.ops",
        "vllm.v1.attention.backends",
        "vllm.v1.attention.backends.mla",
        "vllm.v1.attention.backends.mla.prefill",
    )
    stub_module(
        "vllm.v1.attention.ops.flashmla",
        is_flashmla_dense_supported=lambda: (False, "unavailable"),
    )
    backend_enum = SimpleNamespace(CUSTOM=object())
    stub_module(
        "vllm.v1.attention.backends.mla.prefill.registry",
        MLAPrefillBackendEnum=backend_enum,
    )
    return backend_enum


def _load_platform(stub_module):
    """Load the platform code with only the API pieces this test exercises."""
    _stub_packages(
        stub_module,
        "vllm",
        "vllm.platforms",
        "vllm.utils",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
    )
    stub_module("psutil", virtual_memory=lambda: SimpleNamespace(total=0))
    stub_module(
        "torch",
        cuda=SimpleNamespace(),
        device=object,
        dtype=object,
        no_grad=lambda: None,
        types=SimpleNamespace(Device=object),
    )
    stub_module("vllm.envs")
    stub_module("vllm.logger", init_logger=lambda _: logging.getLogger("platform-test"))
    stub_module("vllm.config", CUDAGraphMode=SimpleNamespace(NONE=object()))
    stub_module(
        "vllm.platforms.interface",
        DeviceCapability=lambda **kwargs: kwargs,
        Platform=object,
        PlatformEnum=SimpleNamespace(
            OOT="oot",
            ROCM="rocm",
            TPU="tpu",
            HPU="hpu",
            XPU="xpu",
            CPU="cpu",
            CUDA="cuda",
        ),
    )
    stub_module("vllm.utils.argparse_utils", FlexibleArgumentParser=object)
    stub_module("vllm.v1.attention.backends.registry", AttentionBackendEnum=object)

    path = REPO_ROOT / "vllm_kunlun" / "platforms" / "kunlun.py"
    spec = importlib.util.spec_from_file_location("platform_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KunlunPlatform, sys.modules["vllm.config"].CUDAGraphMode


def _mla_config(*, use_sparse: bool, attention_config, cudagraph_mode):
    hf_config = SimpleNamespace()
    if use_sparse:
        hf_config.index_topk = 2048

    return SimpleNamespace(
        parallel_config=SimpleNamespace(worker_cls="custom", data_parallel_size=1),
        model_config=SimpleNamespace(use_mla=True, hf_config=hf_config),
        cache_config=SimpleNamespace(block_size=64),
        attention_config=attention_config,
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode.NONE),
    )


def test_dense_mla_keeps_existing_prefill_selection(stub_module):
    """The sparse-only placeholder must not be selected for dense MLA."""
    platform, cudagraph_mode = _load_platform(stub_module)
    _install_mla_backend_stubs(stub_module)
    attention_config = SimpleNamespace()
    config = _mla_config(
        use_sparse=False,
        attention_config=attention_config,
        cudagraph_mode=cudagraph_mode,
    )

    platform.check_and_update_config(config)

    assert not hasattr(attention_config, "mla_prefill_backend")


def test_sparse_mla_selects_custom_prefill_with_legacy_config(stub_module):
    """Sparse MLA supports config objects predating ``mla_prefill_backend``."""
    platform, cudagraph_mode = _load_platform(stub_module)
    backend_enum = _install_mla_backend_stubs(stub_module)
    attention_config = SimpleNamespace()
    config = _mla_config(
        use_sparse=True,
        attention_config=attention_config,
        cudagraph_mode=cudagraph_mode,
    )

    platform.check_and_update_config(config)

    assert attention_config.mla_prefill_backend is backend_enum.CUSTOM


def _load_worker_utils(stub_module):
    """Load the worker helper without importing an installed vLLM package."""
    _stub_packages(
        stub_module,
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.model_executor",
        "vllm.model_executor.models",
    )
    stub_module("vllm.v1.worker.utils", KVBlockZeroer=type("KVBlockZeroer", (), {}))
    stub_module(
        "vllm.model_executor.models.utils",
        extract_layer_index=lambda name, _: int(name.split(".")[1]),
    )

    path = REPO_ROOT / "vllm_kunlun" / "v1" / "worker" / "utils.py"
    spec = importlib.util.spec_from_file_location("worker_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bind_kv_cache_reuses_the_runner_cache_list(stub_module):
    """Rebinding clears stale cache references instead of asserting or duplicating."""
    worker_utils = _load_worker_utils(stub_module)
    cache_at_zero = object()
    cache_at_one = object()
    kv_caches = {
        "layers.1.self_attn": cache_at_one,
        "layers.0.self_attn": cache_at_zero,
    }
    forward_context = {name: SimpleNamespace(kv_cache=None) for name in kv_caches}
    runner_kv_caches = [object()]

    worker_utils.bind_kv_cache(kv_caches, forward_context, runner_kv_caches)
    worker_utils.bind_kv_cache(kv_caches, forward_context, runner_kv_caches)

    assert runner_kv_caches == [cache_at_zero, cache_at_one]
    assert [context.kv_cache for context in forward_context.values()] == list(
        kv_caches.values()
    )
