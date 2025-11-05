import logging
import os
from typing import Any

import sentry_sdk
import structlog

logging.basicConfig(level=logging.INFO)
structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()


def get_llm_config() -> dict[str, Any]:
    """Get LLM config from env vars for Ollama.

    Returns:
        Config dict with Ollama model configuration.
    """
    model_server: str = os.getenv("MODEL_SERVER", "http://localhost:11434/v1")
    api_key: str = os.getenv("API_KEY", "EMPTY")
    model_name: str = os.getenv("MODEL_NAME", "qwen3:8b")

    config: dict[str, Any] = {
        "model": model_name,
        "model_server": model_server,
        "api_key": api_key,
        "generate_cfg": {
            "top_p": 0.8,
            "temperature": 0.7,
            "max_input_tokens": 6000,
        },
    }

    logger.info(
        "LLM config initialized",
        model=model_name,
        server=model_server,
    )

    sentry_dsn: str | None = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0)
        logger.info("Sentry initialized for monitoring.")

    return config
