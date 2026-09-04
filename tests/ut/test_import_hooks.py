"""Unit tests for the import dispatcher in ``registration/import_hooks``."""

import builtins
import logging
import sys

import pytest

from vllm_kunlun.registration import import_hooks


@pytest.fixture
def empty_registry(monkeypatch):
    """Run against an empty hook registry instead of the built-in table."""
    monkeypatch.setattr(import_hooks, "_HOOKS", [])
    monkeypatch.setattr(import_hooks, "_HOOK_TARGETS", set())
    monkeypatch.setattr(import_hooks, "_DISPATCHING", False)
    return import_hooks


class Recorder:
    """Hook pair that records how often it was asked and applied."""

    def __init__(self, applied: bool = False) -> None:
        self.applied = applied
        self.checks = 0
        self.modules: list = []

    def is_applied(self, module) -> bool:
        self.checks += 1
        return self.applied

    def apply(self, module) -> None:
        self.modules.append(module)
        self.applied = True


class TestRegisterHook:
    def test_registers_in_order(self, empty_registry):
        first, second = Recorder(), Recorder()

        import_hooks.register_hook("vllm.a", first.is_applied, first.apply)
        import_hooks.register_hook("vllm.b", second.is_applied, second.apply)

        assert [hook.target for hook in import_hooks._HOOKS] == ["vllm.a", "vllm.b"]

    def test_duplicate_target_is_rejected(self, empty_registry):
        # Two patches for one module would make the order non-deterministic.
        recorder = Recorder()
        import_hooks.register_hook("vllm.a", recorder.is_applied, recorder.apply)

        with pytest.raises(ValueError, match="Duplicate import hook target"):
            import_hooks.register_hook("vllm.a", recorder.is_applied, recorder.apply)

    def test_default_hooks_are_registered_at_import_time(self):
        from vllm_kunlun.registration import compat_patches

        assert len(import_hooks._HOOKS) == len(compat_patches.DEFAULT_HOOKS)
        assert import_hooks._HOOK_TARGETS == {
            target for target, _, _ in compat_patches.DEFAULT_HOOKS
        }


class TestDispatchHooks:
    def test_patches_a_loaded_module(self, empty_registry, stub_module):
        module = stub_module("vllm.loaded")
        recorder = Recorder()
        import_hooks.register_hook("vllm.loaded", recorder.is_applied, recorder.apply)

        import_hooks.dispatch_hooks()

        assert recorder.modules == [module]

    def test_skips_a_module_that_is_not_imported_yet(self, empty_registry):
        recorder = Recorder()
        import_hooks.register_hook(
            "vllm.not_imported", recorder.is_applied, recorder.apply
        )

        import_hooks.dispatch_hooks()

        assert recorder.checks == 0
        assert recorder.modules == []

    def test_is_idempotent(self, empty_registry, stub_module):
        stub_module("vllm.loaded")
        recorder = Recorder()
        import_hooks.register_hook("vllm.loaded", recorder.is_applied, recorder.apply)

        import_hooks.dispatch_hooks()
        import_hooks.dispatch_hooks()

        assert len(recorder.modules) == 1

    def test_skips_a_patch_that_is_already_in_effect(self, empty_registry, stub_module):
        stub_module("vllm.loaded")
        recorder = Recorder(applied=True)
        import_hooks.register_hook("vllm.loaded", recorder.is_applied, recorder.apply)

        import_hooks.dispatch_hooks()

        assert recorder.modules == []

    def test_a_failing_patch_is_isolated(self, empty_registry, stub_module, caplog):
        # A broken patch must not abort vLLM's import flow.
        stub_module("vllm.broken")
        stub_module("vllm.working")
        working = Recorder()

        def explode(module):
            raise RuntimeError("patch is broken")

        import_hooks.register_hook("vllm.broken", lambda module: False, explode)
        import_hooks.register_hook("vllm.working", working.is_applied, working.apply)

        with caplog.at_level(logging.ERROR, logger="vllm_kunlun"):
            import_hooks.dispatch_hooks()

        assert "post-import hook failed for target=vllm.broken" in caplog.text
        assert len(working.modules) == 1

    def test_reentrant_dispatch_is_ignored(self, empty_registry, stub_module):
        # Applying a patch imports modules, which re-enters the dispatcher.
        stub_module("vllm.loaded")
        recorder = Recorder()

        def apply(module):
            import_hooks.dispatch_hooks()
            recorder.apply(module)

        import_hooks.register_hook("vllm.loaded", recorder.is_applied, apply)

        import_hooks.dispatch_hooks()

        assert len(recorder.modules) == 1


class TestCustomImport:
    @pytest.fixture
    def imports(self, monkeypatch):
        """Replace the real ``__import__`` and redirect preload with spies."""
        calls = {"imported": [], "preloaded": []}
        sentinel = object()

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            calls["imported"].append((name, fromlist, level))
            return sentinel

        def fake_preload(name, fromlist):
            calls["preloaded"].append((name, fromlist))

        monkeypatch.setattr(import_hooks, "_OLD_IMPORT", fake_import)
        monkeypatch.setattr(import_hooks, "preload_import_mappings", fake_preload)
        calls["sentinel"] = sentinel
        return calls

    def test_absolute_import_is_checked_for_redirects(self, imports):
        result = import_hooks._custom_import("vllm.x", None, None, ("y",), 0)

        assert imports["preloaded"] == [("vllm.x", ("y",))]
        assert imports["imported"] == [("vllm.x", ("y",), 0)]
        assert result is imports["sentinel"]

    def test_relative_import_skips_the_redirects(self, imports):
        # A relative import can never name an upstream vLLM module.
        import_hooks._custom_import("x", None, None, (), 1)

        assert imports["preloaded"] == []
        assert imports["imported"] == [("x", (), 1)]

    def test_a_failing_preload_does_not_break_the_import(self, imports, monkeypatch):
        def explode(name, fromlist):
            raise RuntimeError("redirect is broken")

        monkeypatch.setattr(import_hooks, "preload_import_mappings", explode)

        assert import_hooks._custom_import("vllm.x") is imports["sentinel"]

    def test_dispatch_runs_after_every_import(self, imports, monkeypatch):
        dispatched = []
        monkeypatch.setattr(
            import_hooks, "dispatch_hooks", lambda: dispatched.append(True)
        )

        import_hooks._custom_import("vllm.x")

        assert dispatched == [True]

    def test_install_replaces_the_builtin_import(self, monkeypatch):
        monkeypatch.setattr(builtins, "__import__", builtins.__import__)

        import_hooks.install_import_hook()

        assert builtins.__import__ is import_hooks._custom_import
        assert sys.modules["json"] is __import__("json")
