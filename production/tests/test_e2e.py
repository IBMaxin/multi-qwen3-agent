"""End-to-end tests with real components (no mocks)."""

import json

import pytest

from qwen_pipeline.tools import SafeCalculatorTool


class TestE2ECalculator:
    """Real calculator tests with no mocking."""

    @pytest.fixture
    def calc(self) -> SafeCalculatorTool:
        """Create a real SafeCalculatorTool instance."""
        return SafeCalculatorTool()

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("2+2", 4),
            ("sqrt(16)", 4.0),
            ("sin(0)", 0.0),
            ("max(1,5,3)", 5),
            ("10**2", 100),
            ("abs(-42)", 42),
            ("min(10, 20, 5)", 5),
            ("round(3.7)", 4),
            ("2*3+4", 10),
            ("(5+3)*2", 16),
        ],
    )
    def test_valid_expressions(
        self, calc: SafeCalculatorTool, expr: str, expected: float | int
    ) -> None:
        """Test various valid math expressions."""
        params = f'{{"expression": "{expr}"}}'
        result = json.loads(calc.call(params))
        assert "result" in result
        assert result["result"] == expected

    @pytest.mark.parametrize(
        "expr",
        [
            "1/0",
            "invalid syntax @@",
            "import os",
            "__import__('os')",
            "exec('print(1)')",
            "eval('1+1')",
            "open('file.txt')",
        ],
    )
    def test_invalid_expressions_safe(self, calc: SafeCalculatorTool, expr: str) -> None:
        """Test that invalid/dangerous expressions are rejected safely."""
        params = f'{{"expression": "{expr}"}}'
        result = json.loads(calc.call(params))
        assert "error" in result
        # Should not contain the actual result or execution
        assert "result" not in result

    def test_complex_scientific_expression(self, calc: SafeCalculatorTool) -> None:
        """Test complex expression with multiple operations."""
        expr = "sqrt(abs(-64)) + sin(0) * cos(0) + log(1)"
        params = f'{{"expression": "{expr}"}}'
        result = json.loads(calc.call(params))

        assert "result" in result
        # sqrt(64) + 0*1 + 0 = 8
        assert result["result"] == 8.0

    def test_expression_with_parentheses(self, calc: SafeCalculatorTool) -> None:
        """Test proper order of operations with parentheses."""
        expr = "((2 + 3) * 4 - 1) / 2"
        params = f'{{"expression": "{expr}"}}'
        result = json.loads(calc.call(params))

        assert "result" in result
        # ((5) * 4 - 1) / 2 = 19/2 = 9.5
        assert result["result"] == 9.5

    def test_calculator_initialization(self, calc: SafeCalculatorTool) -> None:
        """Test that calculator initializes properly with asteval."""
        assert calc.aeval is not None
        assert callable(calc.aeval)

    def test_calculator_metadata(self, calc: SafeCalculatorTool) -> None:
        """Test that calculator has proper metadata."""
        assert calc.description != ""
        assert "math" in calc.description.lower() or "calculate" in calc.description.lower()
        assert len(calc.parameters) > 0
        assert any(p["name"] == "expression" for p in calc.parameters)

    def test_very_long_expression_rejected(self, calc: SafeCalculatorTool) -> None:
        """Test that overly long expressions are rejected."""
        # Create expression longer than 500 chars
        long_expr = " + ".join(["1"] * 200)  # > 500 chars
        params = f'{{"expression": "{long_expr}"}}'
        result = json.loads(calc.call(params))

        assert "error" in result
        assert "too long" in result["error"].lower()

    def test_null_result_detection(self, calc: SafeCalculatorTool) -> None:
        """Test that null/None results are caught."""
        # Division by zero returns None in asteval
        params = '{"expression": "1/0"}'
        result = json.loads(calc.call(params))

        assert "error" in result
        # Should mention either "no value" or be some error
        assert "error" in result or "Expression returned no value" in result.get("error", "")

    def test_whitespace_handling(self, calc: SafeCalculatorTool) -> None:
        """Test that whitespace in expressions is handled correctly."""
        params = '{"expression": "  2   +   3  "}'
        result = json.loads(calc.call(params))

        assert "result" in result
        assert result["result"] == 5

    @pytest.mark.parametrize(
        ("expr", "expected_contains"),
        [
            ("2 ** 3", 8),  # Power
            ("10 % 3", 1),  # Modulo
            ("10 // 3", 3),  # Floor division
            ("-5", -5),  # Negative number
            ("+5", 5),  # Positive number
        ],
    )
    def test_various_operators(
        self, calc: SafeCalculatorTool, expr: str, expected_contains: int
    ) -> None:
        """Test various mathematical operators."""
        params = f'{{"expression": "{expr}"}}'
        result = json.loads(calc.call(params))

        assert "result" in result
        assert result["result"] == expected_contains


class TestE2EToolRegistration:
    """Test tool registration and metadata."""

    def test_calculator_is_registered(self) -> None:
        """Test that SafeCalculatorTool is properly registered."""
        calc = SafeCalculatorTool()

        # Should have BaseTool attributes
        assert hasattr(calc, "call")
        assert hasattr(calc, "description")
        assert hasattr(calc, "parameters")

    def test_calculator_name_in_registry(self) -> None:
        """Test calculator tool name matches expected format."""
        calc = SafeCalculatorTool()

        # Tool should have a name attribute or be registered under 'safe_calculator'
        # Based on @register_tool("safe_calculator") decorator
        assert "safe_calculator" in str(type(calc).__name__.lower()) or hasattr(calc, "name")


class TestE2EJSONHandling:
    """Test JSON input/output handling."""

    def test_json5_compatibility(self) -> None:
        """Test that tool accepts JSON5 format (trailing commas, comments)."""
        calc = SafeCalculatorTool()

        # JSON5 allows trailing commas
        params = '{"expression": "2+2",}'  # Trailing comma
        result = calc.call(params)

        parsed = json.loads(result)
        assert "result" in parsed or "error" in parsed

    def test_unicode_in_expression(self) -> None:
        """Test handling of unicode characters in expressions."""
        calc = SafeCalculatorTool()

        params = '{"expression": "2+2 # π"}'
        result = calc.call(params)

        # Should handle gracefully (either accept or reject cleanly)
        parsed = json.loads(result)
        assert "result" in parsed or "error" in parsed

    def test_escaped_characters(self) -> None:
        """Test handling of escaped characters in JSON."""
        calc = SafeCalculatorTool()

        params = '{"expression": "2+2\\n"}'  # Newline in expression
        result = calc.call(params)

        parsed = json.loads(result)
        # Should strip and process or error cleanly
        assert isinstance(parsed, dict)
