from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from qwen_pipeline.pipeline import PipelineTimeoutError, run_pipeline_streaming


class _MockManager:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def run(self, messages: Any = None) -> Iterator[Any]:
        yield from self._items


@patch("qwen_pipeline.pipeline.create_agents")
@patch("qwen_pipeline.pipeline.human_approval", return_value="Approved")
def test_streaming_yields_messages(mock_approval: Any, mock_create: Any) -> None:
    manager = _MockManager([
        {"content": "Step 1"},
        {"content": "Step 2"},
        {"content": "Final"},
    ])
    mock_create.return_value = manager

    chunks = list(run_pipeline_streaming("hello"))
    assert chunks == ["Step 1", "Step 2", "Final"]
    mock_approval.assert_called_once()


@patch("qwen_pipeline.pipeline.create_agents")
@patch("qwen_pipeline.pipeline.human_approval", return_value="Approved")
def test_streaming_non_dict_responses(mock_approval: Any, mock_create: Any) -> None:
    manager = _MockManager(["a", "b", "c"])  # non-dict
    mock_create.return_value = manager

    chunks = list(run_pipeline_streaming("hello"))
    assert chunks == ["a", "b", "c"]


@patch("qwen_pipeline.pipeline.create_agents")
@patch("qwen_pipeline.pipeline.human_approval", return_value="Approved")
def test_streaming_empty_raises(mock_approval: Any, mock_create: Any) -> None:
    manager = _MockManager([])
    mock_create.return_value = manager

    with pytest.raises(ValueError, match="No response from agent"):
        list(run_pipeline_streaming("hello"))


def test_streaming_timeout_immediate_raises() -> None:
    class Manager:
        def run(self, messages=None):
            # Would yield, but timeout should trigger before first iteration
            yield {"content": "hello"}

    with patch("qwen_pipeline.pipeline.create_agents", return_value=Manager()):
        gen = run_pipeline_streaming("test", timeout_seconds=0)
        with pytest.raises(PipelineTimeoutError):
            next(gen)
