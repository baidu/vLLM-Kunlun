"""Smoke tests for the adapter runtime utilities.

These tests do not require a GPU or any native Kunlun libraries.
"""
import logging
import os
from unittest.mock import patch

import torch

# exercise the package under construction directly from source tree
import vllm_kunlun.runtime_utils as ru


def _env(monkeypatch):
    monkeypatch.delitems(os.environ.keys(), raising=False)
    return monkeypatch


def test_env_bool_canonical():
    with patch.dict(os.environ, {"KUNLUN_DSV4_X": "yes"}, clear=True):
        assert ru.env_bool("KUNLUN_DSV4_X", default=True) is True
        assert ru.env_bool("KUNLUN_DSV4_Y", default=True) is True
        assert ru.env_bool("KUNLUN_DSV4_Z", default=False) is False


def test_alias_deprecation_warning_emitted_once(caplog):
    caplog.set_level(logging.WARNING)
    ru.WarningOnce.clear()
    # Use old name and canonical absence: should emit exactly one warning.
    with caplog.at_level(logging.WARNING), \
         patch.dict(os.environ, {"KUNLUN_V4_HASH_TOPK_FUSED": "0"}, clear=True):
        _ = ru.env_bool(
            "KUNLUN_DSV4_HASH_TOPK_FUSED",
            default=True,
            aliases=("KUNLUN_V4_HASH_TOPK_FUSED",),
        )
        # second resolution should not re-emit thanks to WarningOnce latch
        _ = ru.env_bool(
            "KUNLUN_DSV4_HASH_TOPK_FUSED",
            default=True,
            aliases=("KUNLUN_V4_HASH_TOPK_FUSED",),
        )
    alias_warnings = [
        rec for rec in caplog.records if "deprecated" in str(rec.getMessage())
    ]
    assert len(alias_warnings) == 1, f"expected one warning got {len(alias_warnings)}"


def test_invalid_value_warns_once_and_defaults():
    logger_name = "vllm_kunlun.runtime_utils"
    ru.WarningOnce.clear()
    handler = logging.Handler()
    emitted = []
    class Collect(logging.Handler):
        def emit(self, record): emitted.append(record)
    h = Collect()
    h.setLevel(logging.WARNING)
    logging.getLogger(logger_name).addHandler(h)
    try:
        old = os.environ.pop("TEST_K_BOOL", None)
        try:
            with patch.dict(os.environ, {"TEST_K_BOOL": "maybe"}, clear=False):
                val = ru.env_bool("TEST_K_BOOL", default=False)
            assert val is False
            val2 = ru.env_bool("TEST_K_BOOL", default=False)
            assert val2 is False
            invalid_msgs = [r for r in emitted if r.levelno == logging.WARNING]
            assert len(invalid_msgs) == 1
        finally:
            if old is not None:
                os.environ["TEST_K_BOOL"] = old
    finally:
        logging.getLogger(logger_name).removeHandler(h)


def test_static_cpu_tensor_deterministic_shape_dtype():
    buf = ru.make_static_cpu_tensor("tag_a", (3, 5), torch.float32, fill_value=7)
    assert buf.shape == torch.Size([3, 5])
    assert buf.dtype == torch.float32
    assert (buf == 7).all()
