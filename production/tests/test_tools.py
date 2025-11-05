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
