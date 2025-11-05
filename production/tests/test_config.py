import os

from qwen_pipeline.config import get_llm_config


def test_get_llm_config() -> None:
    os.environ["MODEL_NAME"] = "qwen2.5:7b"
    config = get_llm_config()
    assert config["model"] == "qwen2.5:7b"
    assert "model_server" in config
    assert config["model_server"] == "http://localhost:11434/v1"
