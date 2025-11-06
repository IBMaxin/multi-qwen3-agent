"""Configuration management for Qwen-Agent LLM settings.

Centralized configuration loading from environment variables with
validation and health checks for Ollama connectivity. Provides
get_llm_config() as the single source of truth for LLM parameters.

Copyright: Based on Qwen-Agent patterns from QwenLM/Qwen-Agent
License: Apache License 2.0
"""

import logging
import os
from typing import Any

import requests
import sentry_sdk
import structlog
from dotenv import find_dotenv, load_dotenv

logging.basicConfig(level=logging.INFO)
structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()

_HTTP_OK = 200


def _is_ollama_reachable(model_server: str) -> bool:
    """Check if Ollama server is reachable (non-blocking).

    Args:
        model_server: The Ollama server URL.

    Returns:
        True if server is reachable, False otherwise.
    """
    try:
        health_url = model_server.replace("/v1", "/health")
        response: Any = requests.get(health_url, timeout=2)
    except Exception:
        return False
    else:
        return bool(response.status_code == _HTTP_OK)


# Load variables from .env if present. Does not override explicit environment values.
_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(dotenv_path=_dotenv_path, override=False)
    logger.info("dotenv_loaded", path=_dotenv_path)
else:
    logger.info("dotenv_not_found", using="environment variables only")


def _get_env_float(name: str, default: float | None) -> float | None:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning("invalid_float_env_var", name=name, value=val, default=default)
        return default


def _get_env_int(name: str, default: int | None) -> int | None:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("invalid_int_env_var", name=name, value=val, default=default)
        return default


def get_llm_config() -> dict[str, Any]:
    """Get LLM config from env vars for Ollama with validation.

    Returns:
        Config dict with Ollama model configuration.
    """
    model_server: str = os.getenv("MODEL_SERVER", "http://localhost:11434/v1")
    api_key: str = os.getenv("API_KEY", "EMPTY")
    model_name: str = os.getenv("MODEL_NAME", "qwen3:8b")

    # Validate Ollama connectivity (non-blocking warning)
    # Allow disabling via env: OLLAMA_HEALTHCHECK=0|false|no
    hc_env = os.getenv("OLLAMA_HEALTHCHECK", "1").strip().lower()
    hc_enabled = hc_env not in {"0", "false", "no", "off"}
    if hc_enabled and not _is_ollama_reachable(model_server):
        logger.warning(
            "ollama_unreachable",
            model_server=model_server,
            tip="Ensure 'ollama serve' is running",
        )

    # Base generation defaults (can be overridden via env below)
    generate_cfg: dict[str, Any] = {
        "top_p": 0.8,
        "temperature": 0.7,
        "max_input_tokens": 6000,
    }

    # Optional overrides from env
    temp = _get_env_float("GEN_TEMPERATURE", None)
    if temp is not None:
        generate_cfg["temperature"] = temp
    top_p = _get_env_float("GEN_TOP_P", None)
    if top_p is not None:
        generate_cfg["top_p"] = top_p
    max_in = _get_env_int("GEN_MAX_INPUT_TOKENS", None)
    if max_in is not None:
        generate_cfg["max_input_tokens"] = max_in
    # Limit output length to speed up responses (supported by OpenAI-compatible backends)
    max_out = _get_env_int("GEN_MAX_TOKENS", None)
    if max_out is not None:
        generate_cfg["max_tokens"] = max_out
    fn_type = os.getenv("FN_CALL_PROMPT_TYPE")
    if fn_type:
        generate_cfg["fncall_prompt_type"] = fn_type

    config: dict[str, Any] = {
        "model": model_name,
        "model_server": model_server,
        "api_key": api_key,
        "generate_cfg": generate_cfg,
    }

    logger.info(
        "llm_config_initialized",
        model=model_name,
        server=model_server,
    )

    sentry_dsn: str | None = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0)
        logger.info("sentry_enabled")

    return config
