# Tests

## `tests/ut` — unit tests

Unit tests cover Python logic and CPU tensor adaptation. They do not require
Kunlun devices, torch_xmlir, xspeedgate_ops or kunlun_ops. Tensor tests use CPU
PyTorch and NumPy; upstream vLLM modules and device operators are stubbed.

```bash
pip install pytest numpy
pip install "torch==2.9.0" --index-url https://download.pytorch.org/whl/cpu
pytest
```

## Adding a test

Use the shared fixtures in `tests/conftest.py`:

- `fake_torch` and `module_factory` for pure Python patches.
- `stub_module_tree` and `load_kunlun_override` to load a real override against
  isolated upstream modules.
- `native_ops` to replace device operators without global schema registration.

Check argument forwarding, dtype/shape conversion, output writeback, empty
inputs and registration where applicable. Do not require an operator package
to be installed before a mock can be created.

Device accuracy, Graph replay and performance checks belong in dedicated
jobs under `ci/scripts/tests/`. Temporary experiments belong outside the repo.
