"""Unit tests for the post-import patches in ``registration/compat_patches``.

Each patch is checked against a stand-in module, which is what the dispatcher
passes in, so no vLLM module has to be importable here.
"""

import logging
import sys

import pytest

from vllm_kunlun.registration import compat_patches


class TestDefaultHooks:
    def test_every_entry_is_a_usable_registration(self):
        for entry in compat_patches.DEFAULT_HOOKS:
            target, is_applied, apply_patch = entry
            assert target.startswith("vllm.")
            assert callable(is_applied) and callable(apply_patch)

    def test_targets_are_unique(self):
        targets = [target for target, _, _ in compat_patches.DEFAULT_HOOKS]

        assert len(set(targets)) == len(targets)


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
