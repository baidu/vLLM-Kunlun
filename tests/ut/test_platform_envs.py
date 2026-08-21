# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from vllm_kunlun.platforms import envs

BOOLEAN_ENV_VARS = [
    "ENABLE_VLLM_MULTI_LOG",
    "ENABLE_VLLM_INFER_HOOK",
    "ENABLE_VLLM_OPS_HOOK",
    "ENABLE_VLLM_MODULE_HOOK",
    "ENABLE_VLLM_MOE_FC_SORTED",
    "ENABLE_CUSTOM_DPSK_SCALING_ROPE",
    "ENABLE_VLLM_FUSED_QKV_SPLIT_NORM_ROPE",
    "VLLM_KUNLUN_ENABLE_INT8_BMM",
]


@pytest.mark.parametrize("name", BOOLEAN_ENV_VARS)
def test_boolean_env_vars_default_to_false(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)

    assert getattr(envs, name) is False
    assert envs.is_set(name) is False


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1"])
@pytest.mark.parametrize("name", BOOLEAN_ENV_VARS)
def test_boolean_env_vars_accept_enabled_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    assert getattr(envs, name) is True
    assert envs.is_set(name) is True


@pytest.mark.parametrize("value", ["false", "0", "yes", ""])
@pytest.mark.parametrize("name", BOOLEAN_ENV_VARS)
def test_boolean_env_vars_reject_non_enabled_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    assert getattr(envs, name) is False


def test_vllm_multi_logpath_default_and_override(monkeypatch):
    monkeypatch.delenv("VLLM_MULTI_LOGPATH", raising=False)
    assert envs.VLLM_MULTI_LOGPATH == "./logs"

    monkeypatch.setenv("VLLM_MULTI_LOGPATH", "/tmp/vllm-kunlun-logs")
    assert envs.VLLM_MULTI_LOGPATH == "/tmp/vllm-kunlun-logs"


def test_maybe_convert_int():
    assert envs.maybe_convert_int(None) is None
    assert envs.maybe_convert_int("7") == 7

    with pytest.raises(ValueError):
        envs.maybe_convert_int("invalid")


def test_unknown_env_var_raises_attribute_error():
    with pytest.raises(AttributeError):
        getattr(envs, "UNKNOWN_VLLM_KUNLUN_ENV")

    with pytest.raises(AttributeError):
        envs.is_set("UNKNOWN_VLLM_KUNLUN_ENV")


def test_documented_env_vars_match_runtime_env_vars():
    doc_path = (
        Path(__file__).resolve().parents[2]
        / "docs/source/user_guide/configuration/env_vars.md"
    )
    doc = doc_path.read_text()

    for name in envs.__dir__():
        assert f"`{name}`" in doc

    assert "`FUSED_QK_ROPE_OP`" not in doc
