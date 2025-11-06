import itertools
import sys
from typing import Any
from unittest.mock import patch

import pytest

from qwen_pipeline.cli import main


@patch("builtins.input", side_effect=itertools.cycle(["exit"]))
@patch("sys.exit")
def test_main_exit(mock_exit: Any, mock_input: Any) -> None:
    main()
    mock_exit.assert_called_with(0)


@patch("builtins.input", side_effect=itertools.cycle(["query", "exit"]))
@patch("qwen_pipeline.cli.run_pipeline", return_value="result")
@patch("sys.exit")
def test_main_query(mock_exit: Any, mock_run: Any, mock_input: Any) -> None:
    main()
    mock_run.assert_called_with("query")


@patch("builtins.input", side_effect=KeyboardInterrupt)
@patch("sys.exit")
def test_cli_keyboard_interrupt(mock_exit: Any, mock_input: Any) -> None:
    """Test that CLI handles Ctrl+C gracefully."""
    main()
    mock_exit.assert_called_with(0)


@patch("builtins.input", side_effect=["bad_query", "another_bad", "exit"])
@patch(
    "qwen_pipeline.cli.run_pipeline",
    side_effect=[ValueError("test"), ValueError("test2"), "ok"],
)
@patch("sys.exit")
def test_cli_error_recovery(mock_exit: Any, mock_run: Any, mock_input: Any) -> None:
    """Test that CLI recovers from pipeline errors."""
    main()
    # Should not exit on first 2 errors, only on 'exit' command
    mock_exit.assert_called_with(0)


@patch("builtins.input", side_effect=["err1", "err2", "err3", "err4"])
@patch("qwen_pipeline.cli.run_pipeline", side_effect=Exception("test"))
@patch("sys.exit")
def test_cli_max_errors(mock_exit: Any, mock_run: Any, mock_input: Any) -> None:
    """Test that CLI exits after max errors threshold."""
    main()
    # Should exit with code 1 after 3 errors
    mock_exit.assert_called_with(1)


@patch("builtins.input", side_effect=EOFError)
@patch("sys.exit")
def test_cli_eof_handling(mock_exit: Any, mock_input: Any) -> None:
    """Test that CLI handles EOF gracefully."""
    main()
    mock_exit.assert_called_with(0)


def test_cli_metrics_command(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate `qwen-pipeline metrics`
    monkeypatch.setattr(sys, "argv", ["prog", "metrics"])
    with (
        patch("qwen_pipeline.cli._get_metrics_json", return_value="{}"),
        patch("sys.exit") as mock_exit,
    ):
        main()
        mock_exit.assert_called_with(0)


@patch("builtins.input", side_effect=["q", "exit"])
@patch("qwen_pipeline.cli.run_pipeline_streaming", return_value=iter(["a", "b"]))
@patch("sys.exit")
def test_cli_stream_flag(mock_exit: Any, mock_stream: Any, mock_input: Any) -> None:
    with patch.object(sys, "argv", ["prog", "--stream", "--timeout", "5"]):
        main()
    mock_stream.assert_called_with("q", timeout_seconds=5)
