"""Integration tests - test full pipeline flow."""

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

from qwen_pipeline.agent import create_agents
from qwen_pipeline.pipeline import run_pipeline
from qwen_pipeline.tools import SafeCalculatorTool


class TestPipelineIntegration:
    """Full pipeline integration tests."""

    @patch("qwen_pipeline.pipeline.create_agents")
    @patch("qwen_pipeline.pipeline.human_approval", return_value="Final answer: 42")
    def test_full_pipeline_flow(self, mock_approval: Any, mock_create: Any) -> None:
        """Test complete pipeline: input → agent → approval → output."""

        class MockManager:
            def run(self, messages: Any = None) -> Iterator[dict[str, str]]:
                yield {"content": "Step 1"}
                yield {"content": "Step 2"}
                yield {"content": "Final: 42"}

        manager = MockManager()
        mock_create.return_value = manager

        result = run_pipeline("Calculate 2+2")

        assert "42" in result
        mock_create.assert_called_once()
        mock_approval.assert_called_once()

    @patch("qwen_pipeline.pipeline.create_agents")
    @patch(
        "qwen_pipeline.pipeline.human_approval",
        side_effect=ValueError("User rejected"),
    )
    def test_pipeline_hitl_rejection(self, mock_approval: Any, mock_create: Any) -> None:
        """Test HITL rejection stops pipeline."""

        class MockManager:
            def run(self, messages: Any = None) -> Iterator[dict[str, str]]:
                yield {"content": "Result"}

        manager = MockManager()
        mock_create.return_value = manager

        result = run_pipeline("test")

        assert "Error" in result
        assert "User rejected" in result

    @patch("qwen_pipeline.pipeline.create_agents", side_effect=Exception("Agent error"))
    def test_pipeline_agent_error(self, mock_create: Any) -> None:
        """Test graceful handling of agent initialization errors."""
        result = run_pipeline("test")

        assert "Error" in result
        assert "Agent error" in result

    @patch("qwen_pipeline.pipeline.create_agents")
    @patch("qwen_pipeline.pipeline.human_approval", return_value="Empty response")
    def test_pipeline_empty_responses(self, mock_approval: Any, mock_create: Any) -> None:
        """Test pipeline handles empty response list."""

        class MockManager:
            def run(self, messages: Any = None) -> Iterator[dict[str, str]]:
                return iter([])  # Empty iterator

        manager = MockManager()
        mock_create.return_value = manager

        result = run_pipeline("test")

        assert "Error" in result
        assert "No response" in result

    @patch("qwen_pipeline.pipeline.create_agents")
    @patch("qwen_pipeline.pipeline.human_approval", return_value="No content")
    def test_pipeline_empty_content(self, mock_approval: Any, mock_create: Any) -> None:
        """Test pipeline handles responses with empty content."""

        class MockManager:
            def run(self, messages: Any = None) -> Iterator[dict[str, str]]:
                yield {"content": ""}  # Empty content

        manager = MockManager()
        mock_create.return_value = manager

        result = run_pipeline("test")

        # Should handle empty content gracefully
        assert result is not None


class TestToolsIntegration:
    """Tool integration tests."""

    def test_calculator_with_qwen_format(self) -> None:
        """Test calculator accepts Qwen tool format."""
        calc = SafeCalculatorTool()

        # Qwen format: JSON string with parameters
        params = '{"expression": "sqrt(16) + 2*3"}'
        result = calc.call(params)

        # Should return JSON string
        parsed = json.loads(result)
        assert "result" in parsed
        assert parsed["result"] == 10.0

    def test_calculator_error_format(self) -> None:
        """Test calculator error format is consistent."""
        calc = SafeCalculatorTool()

        params = '{"expression": "1/0"}'
        result = calc.call(params)

        parsed = json.loads(result)
        assert "error" in parsed
        assert isinstance(parsed["error"], str)

    def test_calculator_with_complex_expression(self) -> None:
        """Test calculator handles complex nested expressions."""
        calc = SafeCalculatorTool()

        params = '{"expression": "(sqrt(25) + 3) * 2 - 1"}'
        result = calc.call(params)

        parsed = json.loads(result)
        assert "result" in parsed
        # (5 + 3) * 2 - 1 = 15
        assert parsed["result"] == 15.0

    def test_calculator_with_scientific_functions(self) -> None:
        """Test calculator supports scientific functions."""
        calc = SafeCalculatorTool()

        params = '{"expression": "sin(0) + cos(0) + abs(-5)"}'
        result = calc.call(params)

        parsed = json.loads(result)
        assert "result" in parsed
        # sin(0)=0, cos(0)=1, abs(-5)=5 → 0+1+5=6
        assert parsed["result"] == 6.0


class TestAgentCreation:
    """Agent creation tests."""

    def test_create_agents_with_custom_tools(self) -> None:
        """Test agents can be created with custom tool list."""
        calc_tool = SafeCalculatorTool()
        tools = ["code_interpreter", calc_tool]

        manager = create_agents(tools)

        assert manager is not None
        # GroupChat should have agents list
        assert hasattr(manager, "agents")
        assert len(manager.agents) == 3  # Planner, Coder, Reviewer

    def test_create_agents_with_code_interpreter_only(self) -> None:
        """Test agents with only code_interpreter."""
        tools = ["code_interpreter"]

        manager = create_agents(tools)

        assert manager is not None
        assert hasattr(manager, "agents")
        assert len(manager.agents) == 3

    def test_create_agents_with_calculator_only(self) -> None:
        """Test agents with only calculator tool."""
        calc_tool = SafeCalculatorTool()
        tools = [calc_tool]

        manager = create_agents(tools)

        assert manager is not None
        assert hasattr(manager, "agents")
        # Should still create all 3 agents
        assert len(manager.agents) == 3


class TestMessageFormatting:
    """Test message format compatibility with Qwen-Agent."""

    @patch("qwen_pipeline.pipeline.create_agents")
    @patch("qwen_pipeline.pipeline.human_approval", return_value="OK")
    def test_message_format_structure(self, mock_approval: Any, mock_create: Any) -> None:
        """Test that messages follow official Qwen-Agent format."""

        class MockManager:
            def __init__(self) -> None:
                self.received_messages: list[dict[str, str]] | None = None

            def run(self, messages: Any = None) -> Iterator[dict[str, str]]:
                self.received_messages = messages
                yield {"content": "Response"}

        manager = MockManager()
        mock_create.return_value = manager

        run_pipeline("Test query")

        # Verify message format
        assert manager.received_messages is not None
        assert len(manager.received_messages) == 1
        assert manager.received_messages[0]["role"] == "user"
        assert manager.received_messages[0]["content"] == "Test query"


class TestErrorRecovery:
    """Test error recovery scenarios."""

    @patch("qwen_pipeline.pipeline.create_agents")
    @patch("qwen_pipeline.pipeline.human_approval", return_value="Recovered")
    def test_pipeline_recovers_from_non_dict_response(
        self, mock_approval: Any, mock_create: Any
    ) -> None:
        """Test pipeline handles non-dict responses gracefully."""

        class MockManager:
            def run(self, messages: Any = None) -> Iterator[Any]:
                yield "String response"  # Not a dict!
                yield {"content": "Dict response"}

        manager = MockManager()
        mock_create.return_value = manager

        result = run_pipeline("test")

        # Should convert string to dict and continue
        assert "Recovered" in result or "Dict response" in result
