"""Custom tools for the Qwen-Agent system.

This module defines custom tools that extend the capabilities of the Qwen-Agent.
Tools defined here follow the `qwen_agent.tools.base.BaseTool` pattern and
are registered for use by agents.

- SafeCalculatorTool: A secure calculator that uses `asteval` to prevent
  unsafe code execution.
"""

import json
from typing import Any, ClassVar

import json5
import structlog
from asteval import Interpreter
from qwen_agent.tools.base import BaseTool, register_tool

structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()


@register_tool("safe_calculator")
class SafeCalculatorTool(BaseTool):
    """Safe calculator tool using asteval.

    This tool evaluates math expressions safely without exec or eval.
    """

    description: ClassVar[str] = "Safely calculate math like sqrt(16) or sin(3.14)."
    parameters: ClassVar[list[dict[str, Any]]] = [
        {"name": "expression", "type": "string", "required": True}
    ]

    def __init__(self, _cfg: dict | None = None) -> None:
        """Initialize the safe interpreter.

        _cfg: Optional tool configuration passed by qwen-agent registry; ignored.
        """
        super().__init__()
        self.aeval: Interpreter = Interpreter()
        logger.info("safe_calculator_initialized")

    def call(self, params: str, **_kwargs: Any) -> str:  # noqa: PLR0911
        """Call the calculator with robust error handling.

        Args:
            params: JSON string with expression.
            **kwargs: Optional extra args.

        Returns:
            JSON string with result or error.
        """
        logger.info({"event": "calculator_call", "params": params})

        # Validate JSON structure
        try:
            params_dict: dict[str, str] = json5.loads(params)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "calculator_invalid_json",
                raw_params=params[:100],  # Truncate for logging
                error=str(e),
            )
            return json.dumps({"error": "Invalid JSON parameters"})

        # Validate required field
        if "expression" not in params_dict:
            logger.warning("calculator_missing_expression", keys=list(params_dict.keys()))
            return json.dumps(
                {
                    "error": "Missing 'expression' parameter",
                    "expected_keys": ["expression"],
                }
            )

        expression: str = str(params_dict["expression"]).strip()

        # Validate non-empty
        if not expression:
            logger.warning("calculator_empty_expression")
            return json.dumps({"error": "Expression cannot be empty"})

        # Limit expression length (prevent abuse)
        max_expr_len = 500
        if len(expression) > max_expr_len:
            logger.warning(
                "calculator_expr_too_long",
                length=len(expression),
                max=max_expr_len,
            )
            return json.dumps({"error": f"Expression too long (max {max_expr_len} chars)"})

        try:
            result: Any = self.aeval(expression)

            if result is None:
                logger.warning("calculator_null_result", expression=expression)
                return json.dumps(
                    {
                        "error": "Expression returned no value",
                        "expression": expression,
                    }
                )

            # Convert numpy types to native Python types for JSON serialization
            if hasattr(result, "item"):  # numpy scalar
                result = result.item()
            elif isinstance(result, (list, tuple)):
                result = [x.item() if hasattr(x, "item") else x for x in result]

            logger.info(
                "calculator_success",
                expression=expression,
                result=result,
            )
            return json.dumps({"result": result})

        except ZeroDivisionError:
            logger.warning("calculator_division_by_zero", expression=expression)
            return json.dumps(
                {
                    "error": "Division by zero",
                    "expression": expression,
                }
            )

        except (ValueError, SyntaxError) as e:
            logger.warning(
                "calculator_invalid_expression",
                expression=expression,
                error_type=type(e).__name__,
            )
            return json.dumps(
                {
                    "error": f"Invalid expression: {str(e)[:100]}",
                    "expression": expression,
                }
            )

        except Exception:
            logger.exception(
                "calculator_unexpected_error",
                expression=expression,
            )
            # Don't leak internal errors to user
            return json.dumps({"error": "Calculation failed (internal error)"})
