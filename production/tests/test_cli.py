import itertools
from typing import Any
from unittest.mock import patch

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
