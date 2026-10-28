"""Unit tests for the plugin startup stages in ``registration/bootstrap``.

Every stage defers its torch import into the function body, so these tests
drive the real code with a stub torch and never touch a device.
"""

import logging
import sys
from types import SimpleNamespace

import pytest

from vllm_kunlun.registration import bootstrap


class TestStubVllmCudaExtensions:
    def test_registers_placeholder_for_each_extension(self, sys_modules_guard):
        for name in bootstrap._CUDA_EXTENSION_MODULES:
            sys.modules.pop(name, None)

        bootstrap.stub_vllm_cuda_extensions()

        for name in bootstrap._CUDA_EXTENSION_MODULES:
            assert sys.modules[name].__name__ == name

    def test_does_not_replace_an_already_imported_extension(self, stub_module):
        existing = stub_module(bootstrap._CUDA_EXTENSION_MODULES[0])

        bootstrap.stub_vllm_cuda_extensions()

        assert sys.modules[bootstrap._CUDA_EXTENSION_MODULES[0]] is existing


class TestRegisterWeakRefTensor:
    """vLLM's CUDA-graph capture hardcodes ``torch.ops._C.weak_ref_tensor``."""

    def test_aliases_the_xspeedgate_operator_into_the_C_namespace(
        self, weak_ref_env, logger, caplog
    ):
        with caplog.at_level(logging.INFO):
            bootstrap.register_weak_ref_tensor(logger)

        (library,) = weak_ref_env.created_libraries
        assert (library.namespace, library.kind) == ("_C", "FRAGMENT")
        assert library.defined == ["weak_ref_tensor(Tensor input) -> Tensor"]
        assert library.impls == [
            (
                "weak_ref_tensor",
                weak_ref_env.ops.xspeedgate_ops.weak_ref_tensor.default,
                "CUDA",
            )
        ]
        assert "registered _C::weak_ref_tensor" in caplog.text

    def test_keeps_a_reference_to_the_library(self, weak_ref_env, logger):
        # A garbage-collected Library deregisters its operator, so the
        # reference has to outlive the registration call.
        bootstrap.register_weak_ref_tensor(logger)

        assert bootstrap._WEAK_REF_TENSOR_LIBRARY is weak_ref_env.created_libraries[0]

    def test_second_call_does_not_register_again(self, weak_ref_env, logger):
        # register() may run more than once per process; registering the same
        # operator twice is a hard torch error.
        bootstrap.register_weak_ref_tensor(logger)
        bootstrap.register_weak_ref_tensor(logger)

        assert len(weak_ref_env.created_libraries) == 1

    def test_missing_xspeedgate_ops_is_fatal(self, fake_torch, monkeypatch, logger):
        monkeypatch.setattr(bootstrap, "_WEAK_REF_TENSOR_LIBRARY", None)
        monkeypatch.setitem(sys.modules, "xspeedgate_ops", None)

        with pytest.raises(RuntimeError, match="xspeedgate_ops>=1.5.0"):
            bootstrap.register_weak_ref_tensor(logger)

        assert fake_torch.created_libraries == []

    @pytest.mark.parametrize(
        "installed",
        ["1.4.9", "1.4.0+b1629e2.torch29", "0.0.0", "unknown"],
    )
    def test_rejects_versions_below_the_floor(
        self, weak_ref_env, monkeypatch, logger, installed
    ):
        monkeypatch.setattr(bootstrap, "version", lambda name: installed)

        with pytest.raises(RuntimeError, match="xspeedgate_ops>=1.5.0"):
            bootstrap.register_weak_ref_tensor(logger)

        assert weak_ref_env.created_libraries == []

    @pytest.mark.parametrize("installed", ["1.5.0+b1629e2.torch29", "1.6.0", "2.0.1"])
    def test_accepts_the_floor_version_and_newer(
        self, weak_ref_env, monkeypatch, logger, installed
    ):
        monkeypatch.setattr(bootstrap, "version", lambda name: installed)

        bootstrap.register_weak_ref_tensor(logger)

        assert len(weak_ref_env.created_libraries) == 1

    def test_missing_operator_is_fatal(self, weak_ref_env, monkeypatch, logger):
        # The package can be new enough yet built without the operator.
        monkeypatch.setattr(weak_ref_env, "ops", SimpleNamespace())

        with pytest.raises(RuntimeError, match="weak_ref_tensor operator"):
            bootstrap.register_weak_ref_tensor(logger)

        assert weak_ref_env.created_libraries == []


class TestResolveDeviceIndex:
    def test_integer_is_used_as_is(self, fake_torch):
        assert bootstrap._resolve_device_index(fake_torch, 4) == 4

    def test_device_index_is_extracted(self, fake_torch):
        device = fake_torch.device(2)

        assert bootstrap._resolve_device_index(fake_torch, device) == 2

    @pytest.mark.parametrize("device", [None, "cuda:2", object()])
    def test_unusable_values_fall_back_to_the_current_device(self, fake_torch, device):
        fake_torch.cuda.current_device = lambda: 7

        assert bootstrap._resolve_device_index(fake_torch, device) == 7

    def test_device_without_an_index_falls_back(self, fake_torch):
        fake_torch.cuda.current_device = lambda: 7

        assert bootstrap._resolve_device_index(fake_torch, fake_torch.device()) == 7


class TestPatchMemoryInfo:
    def test_memory_info_comes_from_mem_get_info(self, fake_torch):
        assert bootstrap._kunlun_get_memory_info(3) == (3, 1024)

    def test_patch_installs_the_kunlun_helper(self, fake_torch, logger, caplog):
        with caplog.at_level(logging.INFO):
            bootstrap.patch_memory_info(logger)

        assert (
            fake_torch.accelerator.get_memory_info is bootstrap._kunlun_get_memory_info
        )
        assert "patched torch.accelerator.get_memory_info" in caplog.text


class TestLoadSpecDecodeCompat:
    def test_available_module_is_loaded(self, stub_module, monkeypatch, logger, caplog):
        stub_module("kunlun_fake_spec_decode")
        monkeypatch.setattr(
            bootstrap,
            "_SPEC_DECODE_COMPAT_MODULES",
            ("kunlun_fake_spec_decode",),
        )

        with caplog.at_level(logging.INFO):
            bootstrap.load_spec_decode_compat(logger)

        assert "kunlun_fake_spec_decode" in caplog.text

    def test_missing_module_is_skipped_without_failing(
        self, monkeypatch, logger, caplog
    ):
        # These vLLM features do not exist in every supported version.
        monkeypatch.setattr(
            bootstrap,
            "_SPEC_DECODE_COMPAT_MODULES",
            ("vllm_kunlun.no_such_spec_decode_module",),
        )

        with caplog.at_level(logging.DEBUG):
            bootstrap.load_spec_decode_compat(logger)

        assert "unavailable" in caplog.text


class TestLoadCustomOpsModule:
    @pytest.fixture(autouse=True)
    def _clear_failure_state(self, monkeypatch):
        monkeypatch.setattr(bootstrap, "_CUSTOM_OPS_REGISTRATION_ERROR", None)

    def test_reuses_the_canonical_package_module(self, stub_module):
        # Executing the file again would repeat its torch registrations.
        sys.modules.pop(bootstrap._CUSTOM_OPS_PRIVATE_NAME, None)
        canonical = stub_module(bootstrap._CUSTOM_OPS_CANONICAL_NAME)

        assert bootstrap._load_custom_ops_module() is canonical
        assert sys.modules[bootstrap._CUSTOM_OPS_PRIVATE_NAME] is canonical

    def test_returns_the_already_loaded_private_module(self, stub_module):
        private = stub_module(bootstrap._CUSTOM_OPS_PRIVATE_NAME)

        assert bootstrap._load_custom_ops_module() is private

    def test_previous_failure_is_not_retried(self, monkeypatch):
        cause = RuntimeError("half-registered")
        monkeypatch.setattr(bootstrap, "_CUSTOM_OPS_REGISTRATION_ERROR", cause)

        with pytest.raises(
            bootstrap.CustomOpsRegistrationError, match="fresh process"
        ) as raised:
            bootstrap._load_custom_ops_module()

        assert raised.value.__cause__ is cause

    def test_register_custom_ops_reports_success(self, monkeypatch, logger, caplog):
        monkeypatch.setattr(bootstrap, "_load_custom_ops_module", lambda: None)

        with caplog.at_level(logging.INFO):
            bootstrap.register_custom_ops(logger)

        assert "custom ops registered" in caplog.text
