import json

import pytest

from qwen_pipeline.tools import SafeCalculatorTool


@pytest.fixture
def calc_tool() -> SafeCalculatorTool:
    return SafeCalculatorTool()


def test_calc_success(calc_tool: SafeCalculatorTool) -> None:
    params = '{"expression": "2 + 2"}'
    result = calc_tool.call(params)
    assert '"result": 4' in result


def test_calc_error(calc_tool: SafeCalculatorTool) -> None:
    params = '{"expression": "invalid"}'
    result = calc_tool.call(params)
    assert "error" in result


def test_calc_kwargs(calc_tool: SafeCalculatorTool) -> None:
    params = '{"expression": "3 * 3"}'
    result = calc_tool.call(params, extra="ignored")
    assert '"result": 9' in result


def test_calc_invalid_json(calc_tool: SafeCalculatorTool) -> None:
    """Test calculator with invalid JSON input."""
    params = '{"invalid json}'
    result = calc_tool.call(params)
    assert "Invalid JSON" in result


def test_calc_missing_expression(calc_tool: SafeCalculatorTool) -> None:
    """Test calculator with missing expression parameter."""
    params = '{"other_field": "value"}'
    result = calc_tool.call(params)
    assert "Missing" in result
    assert "expression" in result


def test_calc_empty_expression(calc_tool: SafeCalculatorTool) -> None:
    """Test calculator with empty expression."""
    params = '{"expression": ""}'
    result = calc_tool.call(params)
    assert "empty" in result.lower()


def test_calc_division_by_zero(calc_tool: SafeCalculatorTool) -> None:
    """Test calculator handles division by zero."""
    params = '{"expression": "1/0"}'
    result = calc_tool.call(params)
    # asteval returns None for division by zero, not ZeroDivisionError
    assert "Expression returned no value" in result or "error" in result.lower()


def test_calc_expression_too_long(calc_tool: SafeCalculatorTool) -> None:
    """Test calculator rejects expressions that are too long."""
    long_expr = "1 + 2 " * 200  # Creates expression > 500 chars
    params = f'{{"expression": "{long_expr}"}}'
    result = calc_tool.call(params)
    assert "too long" in result.lower()


def test_calc_invalid_expression(calc_tool: SafeCalculatorTool) -> None:
    """Test calculator handles invalid mathematical expressions."""
    params = '{"expression": "invalid_math_expr"}'
    result = calc_tool.call(params)
    assert "error" in result.lower()


def test_calc_numpy_type_conversion(calc_tool: SafeCalculatorTool) -> None:
    """Test calculator converts numpy types to native Python types."""
    # abs() returns numpy int64 which needs conversion
    params = '{"expression": "abs(-42)"}'
    result_str = calc_tool.call(params)
    result = json.loads(result_str)
    assert result["result"] == 42
    assert isinstance(result["result"], int)  # Native Python int, not numpy.int64
