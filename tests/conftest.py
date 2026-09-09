"""Shared fixtures for the vllm_kunlun unit tests.

The suite is deliberately hardware-free: it runs on a plain CPython 3.10 with
only pytest installed, without torch, vllm, torch_xmlir, xspeedgate_ops or a
Kunlun device.  That is possible because the modules under test either use
the standard library only, or defer their vendor imports into function
bodies, which lets a test install a stub in ``sys.modules`` first.

Fixtures provided here:

``sys_modules_guard``  Undo ``sys.modules`` edits made during a test.
``stub_module``        Create an empty module and register it.
``module_factory``     Create a module object without registering it.
``fake_torch``         Minimal torch stand-in installed as ``sys.modules``.
``weak_ref_env``       ``fake_torch`` plus everything ``register_weak_ref_tensor``
                       needs to succeed.
``envs_module``        ``vllm_kunlun/platforms/envs.py`` loaded standalone.
``logger``             Plain logger for the helpers that take one.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vllm_kunlun.registration import bootstrap

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeLibrary:
    """Record ``torch.library.Library`` calls instead of touching torch.

    Registering an operator for real is global process state that cannot be
    undone reliably, so the tests assert on what would have been registered.
    """

    def __init__(self, namespace: str, kind: str) -> None:
        self.namespace = namespace
        self.kind = kind
        self.defined: list[str] = []
        self.impls: list[tuple] = []

    def define(self, schema: str) -> None:
        self.defined.append(schema)

    def impl(self, name: str, func, dispatch_key: str) -> None:
        self.impls.append((name, func, dispatch_key))


class FakeDevice:
    """Stand-in for ``torch.device``; only ``index`` is ever read."""

    def __init__(self, index=None) -> None:
        self.index = index


@pytest.fixture(autouse=True)
def vllm_kunlun_logger_visible_to_caplog():
    """Undo vllm's logger configuration for the duration of a test.

    ``vllm.logger.init_logger`` sets ``propagate = False`` and pins a level on
    the loggers it configures, and merely collecting this suite is enough to
    import vllm. After that, records emitted under ``vllm_kunlun`` never reach
    the root handler ``caplog`` installs, and ``caplog.at_level(DEBUG)`` -- which
    only raises the *root* level -- cannot re-enable the debug records either.
    Both make ``caplog.text`` come back empty, which is why those assertions
    pass file by file and fail only when the whole suite runs in one process.
    """
    log = logging.getLogger("vllm_kunlun")
    saved = (log.propagate, log.level)
    log.propagate = True
    log.setLevel(logging.NOTSET)
    yield
    log.propagate, log.level = saved


@pytest.fixture
def sys_modules_guard():
    """Restore ``sys.modules`` after a test that installs stub modules.

    Only hand-built stubs are evicted on teardown, recognised by having neither
    ``__spec__`` nor ``__file__``. Dropping *every* entry that appeared during
    the test also evicts modules pytest imports lazily; because
    ``_pytest.fixtures`` and ``_pytest.python`` import each other, re-importing
    one of them mid-session yields a half-initialised module, and pytest then
    dies with ``INTERNALERROR ... cannot import name 'FixtureLookupError'`` the
    next time it has to render a failure.
    """
    saved = dict(sys.modules)
    yield sys.modules
    for name in set(sys.modules) - set(saved):
        module = sys.modules[name]
        if (
            getattr(module, "__spec__", None) is None
            and getattr(module, "__file__", None) is None
        ):
            del sys.modules[name]
    for name, module in saved.items():
        if sys.modules.get(name) is not module:
            sys.modules[name] = module


@pytest.fixture
def module_factory():
    """Return a factory for standalone module objects (not importable)."""

    def make(name: str = "fake_module", **attributes) -> ModuleType:
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        return module

    return make


@pytest.fixture
def stub_module(sys_modules_guard, module_factory):
    """Return a factory that also registers the module in ``sys.modules``."""

    def make(name: str, **attributes) -> ModuleType:
        module = module_factory(name, **attributes)
        sys.modules[name] = module
        return module

    return make


@pytest.fixture
def fake_torch(sys_modules_guard):
    """Install a minimal torch stub for the deferred ``import torch`` calls.

    ``created_libraries`` collects every ``torch.library.Library`` the code
    under test asked for, in creation order.
    """
    created: list[FakeLibrary] = []

    def make_library(namespace: str, kind: str) -> FakeLibrary:
        library = FakeLibrary(namespace, kind)
        created.append(library)
        return library

    torch = ModuleType("torch")
    torch.created_libraries = created
    torch.library = SimpleNamespace(Library=make_library)
    torch.ops = SimpleNamespace(
        xspeedgate_ops=SimpleNamespace(
            weak_ref_tensor=SimpleNamespace(default="xspeedgate_ops::weak_ref_tensor")
        )
    )
    torch.device = FakeDevice
    torch.cuda = SimpleNamespace(
        current_device=lambda: 0,
        mem_get_info=lambda index: (index, 1024),
    )
    torch.accelerator = SimpleNamespace()
    sys.modules["torch"] = torch
    return torch


@pytest.fixture
def weak_ref_env(fake_torch, stub_module, monkeypatch):
    """Set up a working environment for ``register_weak_ref_tensor``.

    ``monkeypatch`` also resets the module-level Library reference, so the
    idempotency guard starts from a clean state and is restored afterwards.
    """
    stub_module("xspeedgate_ops")
    monkeypatch.setattr(bootstrap, "version", lambda name: "1.5.1+87067b3.torch29")
    monkeypatch.setattr(bootstrap, "_WEAK_REF_TENSOR_LIBRARY", None)
    return fake_torch


@pytest.fixture
def envs_module():
    """Load ``platforms/envs.py`` directly from its path.

    ``vllm_kunlun.platforms.__init__`` imports the platform class, which needs
    torch and vllm.  ``envs.py`` has no intra-package imports, so loading the
    file on its own keeps these tests hardware-free.
    """
    path = REPO_ROOT / "vllm_kunlun" / "platforms" / "envs.py"
    spec = importlib.util.spec_from_file_location("vllm_kunlun_envs_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def logger():
    """Logger for helpers that take one; assert on records via ``caplog``."""
    return logging.getLogger("vllm_kunlun.tests")
