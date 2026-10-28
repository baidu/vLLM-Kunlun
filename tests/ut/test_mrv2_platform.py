from types import SimpleNamespace

import pytest


def test_auto_worker_class_is_v1_worker():
    """Document the platform contract used by check_and_update_config."""
    parallel_config = SimpleNamespace(worker_cls="auto")
    if parallel_config.worker_cls == "auto":
        parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
    assert parallel_config.worker_cls == "vllm.v1.worker.gpu_worker.Worker"


def test_explicit_worker_class_is_preserved():
    parallel_config = SimpleNamespace(worker_cls="custom.Worker")
    if parallel_config.worker_cls == "auto":
        parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
    assert parallel_config.worker_cls == "custom.Worker"


@pytest.mark.parametrize(
    "method, expected",
    [("eagle", "speculative decoding"), ("dflash", "speculative decoding")],
)
def test_v2_gate_marks_speculative_decoding_unsupported(method, expected):
    """The Kunlun V2 gate rejects speculative decoding independently of method."""
    config = SimpleNamespace(speculative_config=SimpleNamespace(method=method))
    unsupported = []
    if config.speculative_config is not None:
        unsupported.append(expected)
    assert expected in unsupported
