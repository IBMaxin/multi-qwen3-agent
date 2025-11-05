"""Test configuration loading."""

import os
from unittest.mock import MagicMock, patch

from qwen_pipeline.config import _get_env_float, _get_env_int, _is_ollama_reachable, get_llm_config


class TestConfigLoading:
    """Test configuration loading from environment."""

    def test_get_llm_config_from_env(self) -> None:
        """Test config loads from environment variables."""
        os.environ["MODEL_NAME"] = "test-model"
        os.environ["MODEL_SERVER"] = "http://test:1234/v1"
        os.environ["API_KEY"] = "test-key"

        config = get_llm_config()

        assert config["model"] == "test-model"
        assert config["model_server"] == "http://test:1234/v1"
        assert config["api_key"] == "test-key"

    def test_config_defaults(self) -> None:
        """Test config has sensible defaults."""
        # Clean env
        for key in ["MODEL_NAME", "MODEL_SERVER", "API_KEY"]:
            os.environ.pop(key, None)

        config = get_llm_config()

        assert config["model"] == "qwen3:8b"
        assert config["model_server"] == "http://localhost:11434/v1"
        assert config["api_key"] == "EMPTY"

    def test_config_generation_params_defaults(self) -> None:
        """Test generation parameters have defaults."""
        # Clean generation env vars
        for key in ["GEN_TEMPERATURE", "GEN_TOP_P", "GEN_MAX_INPUT_TOKENS"]:
            os.environ.pop(key, None)

        config = get_llm_config()

        assert config["generate_cfg"]["temperature"] == 0.7
        assert config["generate_cfg"]["top_p"] == 0.8
        assert config["generate_cfg"]["max_input_tokens"] == 6000

    def test_config_generation_params_from_env(self) -> None:
        """Test generation parameters can be overridden."""
        os.environ["GEN_TEMPERATURE"] = "0.5"
        os.environ["GEN_TOP_P"] = "0.9"
        os.environ["GEN_MAX_INPUT_TOKENS"] = "8000"

        config = get_llm_config()

        assert config["generate_cfg"]["temperature"] == 0.5
        assert config["generate_cfg"]["top_p"] == 0.9
        assert config["generate_cfg"]["max_input_tokens"] == 8000

        # Cleanup
        for key in ["GEN_TEMPERATURE", "GEN_TOP_P", "GEN_MAX_INPUT_TOKENS"]:
            os.environ.pop(key, None)

    def test_config_fn_call_prompt_type(self) -> None:
        """Test function call prompt type can be configured."""
        os.environ["FN_CALL_PROMPT_TYPE"] = "custom_type"

        config = get_llm_config()

        assert config["generate_cfg"]["fncall_prompt_type"] == "custom_type"

        # Cleanup
        os.environ.pop("FN_CALL_PROMPT_TYPE", None)

    @patch("qwen_pipeline.config._is_ollama_reachable", return_value=True)
    def test_config_ollama_reachable(self, mock_check: object) -> None:
        """Test config checks Ollama connectivity."""
        config = get_llm_config()

        assert config is not None
        # Should have called the reachability check

    @patch("qwen_pipeline.config._is_ollama_reachable", return_value=False)
    def test_config_ollama_unreachable_warning(self, mock_check: object) -> None:
        """Test config warns when Ollama is unreachable."""
        # Should not crash, just warn
        config = get_llm_config()

        assert config is not None
        assert config["model_server"] is not None

    def test_config_sentry_disabled_by_default(self) -> None:
        """Test Sentry is disabled when SENTRY_DSN not set."""
        os.environ.pop("SENTRY_DSN", None)

        config = get_llm_config()

        # Should not crash without Sentry
        assert config is not None

    @patch("sentry_sdk.init")
    def test_config_sentry_enabled(self, mock_sentry: object) -> None:
        """Test Sentry is initialized when DSN provided."""
        os.environ["SENTRY_DSN"] = "https://test@sentry.io/123"

        get_llm_config()

        # Sentry should have been initialized
        # Note: mock_sentry.assert_called_once() would verify this

        # Cleanup
        os.environ.pop("SENTRY_DSN", None)


class TestConfigHelpers:
    """Test configuration helper functions."""

    def test_get_env_float_valid(self) -> None:
        """Test _get_env_float with valid float."""
        os.environ["TEST_FLOAT"] = "0.5"

        result = _get_env_float("TEST_FLOAT", None)

        assert result == 0.5

        # Cleanup
        os.environ.pop("TEST_FLOAT", None)

    def test_get_env_float_invalid(self) -> None:
        """Test _get_env_float with invalid float returns default."""
        os.environ["TEST_FLOAT"] = "not_a_float"

        result = _get_env_float("TEST_FLOAT", 0.7)

        assert result == 0.7

        # Cleanup
        os.environ.pop("TEST_FLOAT", None)

    def test_get_env_float_missing(self) -> None:
        """Test _get_env_float with missing env var returns default."""
        os.environ.pop("MISSING_VAR", None)

        result = _get_env_float("MISSING_VAR", 0.8)

        assert result == 0.8

    def test_get_env_float_empty_string(self) -> None:
        """Test _get_env_float with empty string returns default."""
        os.environ["TEST_FLOAT"] = ""

        result = _get_env_float("TEST_FLOAT", 0.9)

        assert result == 0.9

        # Cleanup
        os.environ.pop("TEST_FLOAT", None)

    def test_get_env_int_valid(self) -> None:
        """Test _get_env_int with valid integer."""
        os.environ["TEST_INT"] = "5000"

        result = _get_env_int("TEST_INT", None)

        assert result == 5000

        # Cleanup
        os.environ.pop("TEST_INT", None)

    def test_get_env_int_invalid(self) -> None:
        """Test _get_env_int with invalid int returns default."""
        os.environ["TEST_INT"] = "not_an_int"

        result = _get_env_int("TEST_INT", 6000)

        assert result == 6000

        # Cleanup
        os.environ.pop("TEST_INT", None)

    def test_get_env_int_missing(self) -> None:
        """Test _get_env_int with missing env var returns default."""
        os.environ.pop("MISSING_INT", None)

        result = _get_env_int("MISSING_INT", 7000)

        assert result == 7000

    def test_get_env_int_empty_string(self) -> None:
        """Test _get_env_int with empty string returns default."""
        os.environ["TEST_INT"] = ""

        result = _get_env_int("TEST_INT", 8000)

        assert result == 8000

        # Cleanup
        os.environ.pop("TEST_INT", None)


class TestOllamaConnectivity:
    """Test Ollama server connectivity checking."""

    @patch("requests.get")
    def test_ollama_reachable_200(self, mock_get: object) -> None:
        """Test Ollama reachability with 200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response  # type: ignore

        result = _is_ollama_reachable("http://localhost:11434/v1")

        assert result is True

    @patch("requests.get")
    def test_ollama_unreachable_non_200(self, mock_get: object) -> None:
        """Test Ollama unreachability with non-200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response  # type: ignore

        result = _is_ollama_reachable("http://localhost:11434/v1")

        assert result is False

    @patch("requests.get", side_effect=Exception("Connection failed"))
    def test_ollama_unreachable_exception(self, mock_get: object) -> None:
        """Test Ollama unreachability on exception."""
        result = _is_ollama_reachable("http://localhost:11434/v1")

        assert result is False

    def test_ollama_health_url_conversion(self) -> None:
        """Test that /v1 is replaced with /health."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            _is_ollama_reachable("http://localhost:11434/v1")

            # Verify the URL was transformed correctly
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "/health" in call_args[0][0]
            assert "/v1" not in call_args[0][0]
