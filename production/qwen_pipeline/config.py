import logging
import os
from typing import Any

import requests  # type: ignore[import-untyped]
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
    logger.info("✓ .env file loaded", path=_dotenv_path)
else:
    logger.info(".env not found - using environment variables only")


def _get_env_float(name: str, default: float | None) -> float | None:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning("Invalid float for %s: %s", name, val)
        return default


def _get_env_int(name: str, default: int | None) -> int | None:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("Invalid int for %s: %s", name, val)
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
    if not _is_ollama_reachable(model_server):
        logger.warning(
            "⚠ Ollama server may not be reachable",
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
        "✓ LLM config initialized",
        model=model_name,
        server=model_server,
    )

    sentry_dsn: str | None = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0)
        logger.info("✓ Sentry monitoring enabled")

    return config
