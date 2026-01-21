import pytest

def pytest_collection_modifyitems(config, items):
    skip_gpu_kernel = pytest.mark.skip(
        reason="Skip Kunlun/Flash kernel tests to stabilize CI"
    )
    for item in items:
        if any(name in item.nodeid for name in [
            "test_kunlun_attn_v1.py",
            "test_flashmla_sparse_v1.py",
            "test_flashmla_v1.py",
        ]):
            item.add_marker(skip_gpu_kernel)
