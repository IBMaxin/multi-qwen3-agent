import json
from typing import Any

from qwen_pipeline.tools import SafeCalculatorTool


def _raise_zerodiv(_: str) -> Any:
    raise ZeroDivisionError("division by zero")


def _raise_value_error(_: str) -> Any:
    raise ValueError("bad expression")


def _raise_runtime(_: str) -> Any:
    raise RuntimeError("boom")


def test_calc_handles_zerodivision() -> None:
    calc = SafeCalculatorTool()
    calc.aeval = _raise_zerodiv  # type: ignore[assignment]
    result = json.loads(calc.call('{"expression": "1/0"}'))
    assert result["error"].lower().startswith("division by zero")


def test_calc_handles_value_error() -> None:
    calc = SafeCalculatorTool()
    calc.aeval = _raise_value_error  # type: ignore[assignment]
    out = json.loads(calc.call('{"expression": "oops"}'))
    assert "invalid expression" in out["error"].lower()


def test_calc_handles_generic_exception() -> None:
    calc = SafeCalculatorTool()
    calc.aeval = _raise_runtime  # type: ignore[assignment]
    out = json.loads(calc.call('{"expression": "any"}'))
    assert out["error"].startswith("Calculation failed")
