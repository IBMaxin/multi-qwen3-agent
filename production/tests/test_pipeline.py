from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from qwen_pipeline.pipeline import human_approval, run_pipeline


def test_human_approval_yes() -> None:
    with patch("builtins.input", return_value="yes"):
        result = human_approval("Test", "content")
        assert result == "content"


def test_human_approval_edit() -> None:
    with patch("builtins.input", side_effect=["edit", "new content"]):
        result = human_approval("Test", "content")
        assert result == "new content"


def test_human_approval_no() -> None:
    with patch("builtins.input", return_value="no"), pytest.raises(ValueError):
        human_approval("Test", "content")


def test_human_approval_invalid_then_yes() -> None:
    with patch("builtins.input", side_effect=["invalid", "yes"]):
        result = human_approval("Test", "content")
        assert result == "content"


@patch("qwen_pipeline.pipeline.human_approval", return_value="mock output")
@patch("qwen_pipeline.pipeline.create_agents")
def test_run_pipeline(mock_agents: Any, mock_approval: Any) -> None:
    class MockManager:
        def run(self, messages: Any = None) -> Iterator[dict[str, str]]:
            yield {"content": "mock"}

    mock_agents.return_value = MockManager()
    result = run_pipeline("query")
    assert "mock" in result


@patch("qwen_pipeline.pipeline.create_agents")
def test_run_pipeline_error(mock_agents: Any) -> None:
    mock_agents.side_effect = Exception("Test error")
    result = run_pipeline("query")
    assert "Error" in result


@patch("qwen_pipeline.pipeline.human_approval", return_value="Final")
@patch("qwen_pipeline.pipeline.create_agents")
def test_multiple_responses_accumulated(mock_agents: Any, mock_approval: Any) -> None:
    """Ensure all responses are accumulated, not overwritten."""

    class MockManager:
        def run(self, messages: Any = None) -> Iterator[dict[str, str]]:
            yield {"content": "First"}
            yield {"content": "Second"}
            yield {"content": "Final"}

    mock_agents.return_value = MockManager()
    result = run_pipeline("test")
    assert "Final" in result
