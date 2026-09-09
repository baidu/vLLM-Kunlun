"""Unit tests for the wholesale module swaps in ``registration/module_redirects``."""

import sys

import pytest

from vllm_kunlun.registration import module_redirects


class TestMappingTable:
    def test_maps_upstream_names_to_kunlun_modules(self):
        for upstream, replacement in module_redirects.MODULE_MAPPINGS.items():
            assert upstream.startswith("vllm.")
            assert replacement.startswith("vllm_kunlun.")

    def test_no_replacement_is_reused(self):
        replacements = list(module_redirects.MODULE_MAPPINGS.values())

        assert len(set(replacements)) == len(replacements)


class TestPreloadMapped:
    @pytest.fixture
    def mapping(self, monkeypatch, stub_module):
        """One redirect whose replacement is already importable."""
        replacement = stub_module("kunlun_fake_replacement")
        monkeypatch.setitem(
            module_redirects.MODULE_MAPPINGS,
            "vllm.fake_target",
            replacement.__name__,
        )
        return replacement

    def test_registers_the_replacement_under_the_upstream_name(self, mapping):
        module_redirects.preload_mapped("vllm.fake_target")

        assert sys.modules["vllm.fake_target"] is mapping
        assert sys.modules["kunlun_fake_replacement"] is mapping

    def test_leaves_an_already_imported_upstream_module_alone(
        self, mapping, stub_module
    ):
        # Swapping a module out from under an importer that already holds it
        # would be worse than not redirecting at all.
        upstream = stub_module("vllm.fake_target")

        module_redirects.preload_mapped("vllm.fake_target")

        assert sys.modules["vllm.fake_target"] is upstream

    def test_module_import_triggers_the_redirect(self, mapping):
        module_redirects.preload_import_mappings("vllm.fake_target", None)

        assert sys.modules["vllm.fake_target"] is mapping

    def test_from_import_triggers_the_redirect_for_a_mapped_submodule(self, mapping):
        # ``from vllm import fake_target`` names the parent package only.
        module_redirects.preload_import_mappings("vllm", ["fake_target", "other"])

        assert sys.modules["vllm.fake_target"] is mapping

    def test_unmapped_imports_are_untouched(self, mapping):
        module_redirects.preload_import_mappings("vllm.other", ["thing"])

        assert "vllm.other" not in sys.modules
        assert "vllm.other.thing" not in sys.modules
