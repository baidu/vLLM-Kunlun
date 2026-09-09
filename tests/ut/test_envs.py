"""Unit tests for the Kunlun environment-variable table in ``platforms/envs``."""

import pytest


def boolean_names(envs_module):
    """Every entry except the one string-valued variable."""
    return [
        name
        for name in envs_module.xvllm_environment_variables
        if name != "VLLM_MULTI_LOGPATH"
    ]


@pytest.fixture
def clean_env(envs_module, monkeypatch):
    """Unset every Kunlun variable so defaults can be asserted."""
    for name in envs_module.xvllm_environment_variables:
        monkeypatch.delenv(name, raising=False)
    return envs_module


class TestDefaults:
    def test_log_path_default(self, clean_env):
        assert clean_env.VLLM_MULTI_LOGPATH == "./logs"

    def test_every_flag_defaults_to_off(self, clean_env):
        for name in boolean_names(clean_env):
            assert getattr(clean_env, name) is False, name


class TestBooleanParsing:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("0", False),
            ("false", False),
            ("yes", False),
            ("on", False),
            ("", False),
        ],
    )
    def test_only_true_and_1_enable_a_flag(
        self, clean_env, monkeypatch, value, expected
    ):
        for name in boolean_names(clean_env):
            monkeypatch.setenv(name, value)
            assert getattr(clean_env, name) is expected, name


class TestLookup:
    def test_values_are_read_on_every_access(self, clean_env, monkeypatch):
        # The table holds callables so a variable set after import still counts.
        monkeypatch.setenv("VLLM_MULTI_LOGPATH", "/tmp/first")
        assert clean_env.VLLM_MULTI_LOGPATH == "/tmp/first"

        monkeypatch.setenv("VLLM_MULTI_LOGPATH", "/tmp/second")
        assert clean_env.VLLM_MULTI_LOGPATH == "/tmp/second"

    def test_unknown_name_raises_attribute_error(self, envs_module):
        with pytest.raises(AttributeError, match="NO_SUCH_KUNLUN_VARIABLE"):
            getattr(envs_module, "NO_SUCH_KUNLUN_VARIABLE")

    def test_dir_lists_the_whole_table(self, envs_module):
        assert dir(envs_module) == sorted(envs_module.xvllm_environment_variables)


class TestIsSet:
    def test_reports_whether_the_variable_is_present(self, clean_env, monkeypatch):
        assert clean_env.is_set("ENABLE_VLLM_OPS_HOOK") is False

        # An explicit "False" is still explicitly set.
        monkeypatch.setenv("ENABLE_VLLM_OPS_HOOK", "False")
        assert clean_env.is_set("ENABLE_VLLM_OPS_HOOK") is True

    def test_unknown_name_raises_attribute_error(self, envs_module):
        with pytest.raises(AttributeError, match="NO_SUCH_KUNLUN_VARIABLE"):
            envs_module.is_set("NO_SUCH_KUNLUN_VARIABLE")


class TestMaybeConvertInt:
    @pytest.mark.parametrize("value, expected", [(None, None), ("5", 5), ("-1", -1)])
    def test_converts_unless_none(self, envs_module, value, expected):
        assert envs_module.maybe_convert_int(value) == expected

    def test_rejects_a_non_numeric_value(self, envs_module):
        with pytest.raises(ValueError):
            envs_module.maybe_convert_int("not-a-number")
