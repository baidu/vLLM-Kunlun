"""Unit tests for the post-import patches in ``registration/compat_patches``.

Each patch is checked against a stand-in module, which is what the dispatcher
passes in, so no vLLM module has to be importable here.
"""

import logging
import sys
from types import SimpleNamespace

import pytest

from vllm_kunlun.registration import compat_patches


class FakeVllmConfig:
    def _get_v2_model_runner_unsupported_features(self):
        return []

    def __post_init__(self):
        pass


def make_dspark_config(*, num_speculative_tokens=7, tensor_parallel_size=2):
    config = FakeVllmConfig()
    config.speculative_config = SimpleNamespace(
        method="dspark",
        rejection_sample_method="standard",
        num_speculative_tokens=num_speculative_tokens,
        draft_sample_method="greedy",
    )
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_5")
    )
    config.parallel_config = SimpleNamespace(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
    )
    return config


class TestDefaultHooks:
    def test_every_entry_is_a_usable_registration(self):
        for entry in compat_patches.DEFAULT_HOOKS:
            target, is_applied, apply_patch = entry
            assert target.startswith("vllm.")
            assert callable(is_applied) and callable(apply_patch)

    def test_targets_are_unique(self):
        targets = [target for target, _, _ in compat_patches.DEFAULT_HOOKS]

        assert len(set(targets)) == len(targets)


class TestMrv2CapabilityGate:
    @pytest.mark.parametrize("num_speculative_tokens", [1, 3, 8])
    def test_dspark_does_not_restrict_num_speculative_tokens(
        self, module_factory, num_speculative_tokens
    ):
        module = module_factory(HAS_TRITON=False, VllmConfig=FakeVllmConfig)
        compat_patches._apply_mrv2_gate(module)
        config = make_dspark_config(
            num_speculative_tokens=num_speculative_tokens,
            tensor_parallel_size=2,
        )

        assert config._get_v2_model_runner_unsupported_features() == []

    @pytest.mark.parametrize("tensor_parallel_size", [1, 4, 8])
    def test_dspark_does_not_restrict_tensor_parallel_size(
        self, module_factory, tensor_parallel_size
    ):
        module = module_factory(HAS_TRITON=False, VllmConfig=FakeVllmConfig)
        compat_patches._apply_mrv2_gate(module)
        config = make_dspark_config(
            num_speculative_tokens=7,
            tensor_parallel_size=tensor_parallel_size,
        )

        assert config._get_v2_model_runner_unsupported_features() == []


class TestSideEffectImportPatches:
    """Patches whose predicate must tolerate a vLLM version without the target."""

    @pytest.mark.parametrize(
        "is_applied",
        [
            compat_patches._kv_block_zeroer_applied,
            compat_patches._block_table_applied,
            compat_patches._memory_pool_applied,
        ],
    )
    def test_absent_attribute_counts_as_applied(self, module_factory, is_applied):
        assert is_applied(module_factory()) is True

    @pytest.mark.parametrize(
        "is_applied, attribute, marker",
        [
            (
                compat_patches._kv_block_zeroer_applied,
                "KVBlockZeroer",
                "_kunlun_patched",
            ),
            (
                compat_patches._block_table_applied,
                "BlockTable",
                "_kunlun_slot_patched",
            ),
            (
                compat_patches._memory_pool_applied,
                "Worker",
                "_kunlun_memory_pool_patched",
            ),
        ],
    )
    def test_marker_on_the_class_decides(
        self, module_factory, is_applied, attribute, marker
    ):
        unpatched = type("Target", (), {})
        assert is_applied(module_factory(**{attribute: unpatched})) is False

        patched = type("Target", (), {marker: True})
        assert is_applied(module_factory(**{attribute: patched})) is True

    def test_grammar_bitmask_needs_the_marker(self, module_factory):
        def helper():
            pass

        assert compat_patches._grammar_bitmask_applied(module_factory()) is False
        module = module_factory(apply_grammar_bitmask=helper)
        assert compat_patches._grammar_bitmask_applied(module) is False

        helper._kunlun_patched = True
        assert compat_patches._grammar_bitmask_applied(module) is True


class TestTritonPatches:
    def test_qwen3_vl_triton_is_disabled(self, module_factory, caplog):
        module = module_factory(HAS_TRITON=True)
        assert compat_patches._qwen3_vl_applied(module) is False

        with caplog.at_level(logging.INFO, logger="vllm_kunlun"):
            compat_patches._apply_qwen3_vl_patch(module)

        assert module.HAS_TRITON is False
        assert compat_patches._qwen3_vl_applied(module) is True
        assert "HAS_TRITON forced to False" in caplog.text

    def test_minimax_needs_both_gates_disabled(self, module_factory):
        module = module_factory(HAS_TRITON=False, _MINIMAX_FUSED_AR_RMS_QK=object())
        assert compat_patches._minimax_rms_norm_tp_applied(module) is False

        compat_patches._apply_minimax_rms_norm_tp_patch(module)

        assert module.HAS_TRITON is False
        assert module._MINIMAX_FUSED_AR_RMS_QK is None
        assert compat_patches._minimax_rms_norm_tp_applied(module) is True

    def test_qwen_triton_warmup_becomes_a_noop(self, module_factory, caplog):
        module = module_factory(qwen_triton_warmup=lambda *args: None)
        assert compat_patches._warmup_applied(module) is False

        compat_patches._apply_warmup_patch(module)

        assert compat_patches._warmup_applied(module) is True
        with caplog.at_level(logging.INFO, logger="vllm_kunlun"):
            module.qwen_triton_warmup("any", keyword="argument")
        assert "Skipping qwen_triton_warmup" in caplog.text


class TestOptionalDepProbePatch:
    """CUDA-only libraries must be reported absent without a trial import."""

    def test_cuda_only_modules_answer_false_without_importing(self, module_factory):
        probed = []

        def probe(module_name):
            probed.append(module_name)
            return True

        module = module_factory(_has_module=probe)
        assert compat_patches._optional_dep_probe_applied(module) is False

        compat_patches._apply_optional_dep_probe(module)

        assert compat_patches._optional_dep_probe_applied(module) is True
        for name in compat_patches._CUDA_ONLY_MODULES:
            assert module._has_module(name) is False
        assert probed == []

    def test_other_modules_still_reach_the_original_probe(self, module_factory):
        module = module_factory(_has_module=lambda module_name: True)

        compat_patches._apply_optional_dep_probe(module)

        assert module._has_module("numba") is True

    def test_old_vllm_without_the_probe_needs_nothing(self, module_factory):
        assert compat_patches._optional_dep_probe_applied(module_factory()) is True


class TestInt8MoePatch:
    def test_selector_is_replaced_and_reports_no_backend(self, module_factory):
        module = module_factory(select_int8_moe_backend=lambda config: ("cuda", "impl"))
        assert compat_patches._int8_moe_applied(module) is False

        compat_patches._apply_int8_moe_patch(module)

        assert compat_patches._int8_moe_applied(module) is True
        assert module.select_int8_moe_backend(object()) == (None, None)

    def test_absent_selector_is_left_alone(self, module_factory):
        module = module_factory()

        compat_patches._apply_int8_moe_patch(module)

        assert not hasattr(module, "select_int8_moe_backend")


class TestOotRegistrations:
    def test_old_vllm_without_the_registration_api_needs_nothing(self, module_factory):
        module = module_factory(CustomOp=object())

        assert compat_patches._oot_registrations_applied(module) is True

    def test_pending_when_the_ops_package_has_not_registered(
        self, module_factory, sys_modules_guard
    ):
        sys.modules.pop("vllm_kunlun.ops", None)
        module = module_factory(CustomOp=object(), PluggableLayer=object())

        assert compat_patches._oot_registrations_applied(module) is False

    def test_done_once_the_ops_package_marks_itself(self, module_factory, stub_module):
        stub_module("vllm_kunlun.ops", _KUNLUN_OOT_REGISTRATIONS_LOADED=True)
        module = module_factory(CustomOp=object(), PluggableLayer=object())

        assert compat_patches._oot_registrations_applied(module) is True


class TestRmsNormImportOrder:
    def test_patch_defers_kernel_import_until_first_call(
        self, module_factory, monkeypatch
    ):
        import builtins

        original_import = builtins.__import__
        attempts = []
        calls = []
        kernel_module = SimpleNamespace(rms_norm=lambda *args: calls.append(args))

        def guarded_import(name, *args, **kwargs):
            if name == "vllm_kunlun.ops.layernorm":
                attempts.append(name)
                return kernel_module
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        module = module_factory(rms_norm=lambda *args: None)
        compat_patches._apply_rms_norm_abi(module)
        assert attempts == []
        assert compat_patches._rms_norm_abi_applied(module)
        patched = module.rms_norm
        compat_patches._apply_rms_norm_abi(module)
        assert module.rms_norm is patched
        module.rms_norm("out", "input", "weight", 1e-6)
        module.rms_norm("out", "input", "weight", 1e-6)
        assert len(attempts) == 1
        assert calls == [("out", "input", "weight", 1e-6)] * 2
