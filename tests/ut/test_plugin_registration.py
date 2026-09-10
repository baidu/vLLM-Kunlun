"""Unit tests for ordering-sensitive plugin registration stages."""

import logging

import pytest

import vllm_kunlun


@pytest.fixture(autouse=True)
def reset_registration_state(monkeypatch):
    """Keep registration's process-global state isolated between tests."""
    monkeypatch.setattr(vllm_kunlun, "_REGISTER_STATE", "idle")
    monkeypatch.setattr(vllm_kunlun, "_REGISTER_ERROR", None)


def test_glm_config_repair_precedes_all_startup_dependencies(monkeypatch):
    """Repair aliases before any startup stage can trigger a config import."""
    calls = []
    logger = logging.getLogger("vllm_kunlun.test_registration")

    monkeypatch.setattr(
        vllm_kunlun.bootstrap,
        "repair_glm_moe_dsa_head_dims",
        lambda received_logger: calls.append(("repair", received_logger)),
    )
    monkeypatch.setattr(
        vllm_kunlun,
        "_configure_kunlun_logger",
        lambda: calls.append(("logger", None)) or logger,
    )
    monkeypatch.setattr(
        vllm_kunlun,
        "_run_startup_stages",
        lambda received_logger: calls.append(("stages", received_logger)),
    )

    assert vllm_kunlun.register() == vllm_kunlun._KUNLUN_PLATFORM
    assert [name for name, _ in calls] == ["repair", "logger", "stages"]
    assert calls[0][1].name == "vllm_kunlun"
    assert calls[-1][1] is logger


def test_glm_config_repair_failure_resets_registration_state(monkeypatch):
    """A failed early repair must not leave registration stuck in progress."""
    logger = logging.getLogger("vllm_kunlun.test_registration")

    def fail_repair(_):
        raise RuntimeError("repair failed")

    monkeypatch.setattr(
        vllm_kunlun.bootstrap, "repair_glm_moe_dsa_head_dims", fail_repair
    )

    with pytest.raises(RuntimeError, match="repair failed"):
        vllm_kunlun.register()

    assert vllm_kunlun._REGISTER_STATE == "idle"

    monkeypatch.setattr(
        vllm_kunlun.bootstrap, "repair_glm_moe_dsa_head_dims", lambda _: None
    )
    monkeypatch.setattr(vllm_kunlun, "_configure_kunlun_logger", lambda: logger)
    monkeypatch.setattr(vllm_kunlun, "_run_startup_stages", lambda _: None)

    assert vllm_kunlun.register() == vllm_kunlun._KUNLUN_PLATFORM
    assert vllm_kunlun._REGISTER_STATE == "registered"
