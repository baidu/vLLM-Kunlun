# Tests

## `tests/ut` — unit tests

Hardware-free unit tests for the plugin's Python logic.  They run on a plain
CPython 3.10 with only `pytest` installed: no torch, vllm, torch_xmlir,
xspeedgate_ops or Kunlun device, which is what lets them run in CI on every
pull request.

```bash
pip install pytest
pytest                 # configuration lives in pyproject.toml
pytest tests/ut/test_bootstrap.py -v
```

## Adding a test

Keep the suite hardware-free.  Two patterns make that possible:

- Modules under test import torch (or any vendor package) inside function
  bodies, so a test can install a stub in `sys.modules` first.  Use the
  `fake_torch` / `stub_module` fixtures from `tests/conftest.py`.
- Patch functions receive the target module as an argument, so a plain
  `ModuleType` stand-in is enough.  Use `module_factory`.

If a change can only be verified on a device, it belongs in the CI accuracy or
performance jobs under `ci/scripts/tests/`, not here.
