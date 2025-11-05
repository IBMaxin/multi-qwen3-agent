import json
from typing import Any

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

    description: str = "Safely calculate math like sqrt(16) or sin(3.14)."
    parameters: list[dict[str, Any]] = [{"name": "expression", "type": "string", "required": True}]

    def __init__(self) -> None:
        """Initialize the safe interpreter."""
        super().__init__()
        self.aeval: Interpreter = Interpreter()
        logger.info("SafeCalculatorTool initialized.")

    def call(self, params: str, **kwargs: Any) -> str:
        """Call the calculator.

        Args:
            params: JSON string with expression.
            **kwargs: Optional extra args.

        Returns:
            JSON string with result or error.
        """
        logger.info({"event": "calculator_call", "params": params})
        params_dict: dict[str, str] = json5.loads(params)
        expression: str = params_dict["expression"]
        try:
            result: Any = self.aeval(expression)
            if result is None:
                logger.error(
                    {"event": "calculator_error", "error": "Invalid expression or no result."}
                )
                return json.dumps({"error": "Invalid expression or no result."})
            logger.info({"event": "calculator_success", "result": result})
            return json.dumps({"result": result})
        except Exception as e:
            logger.error({"event": "calculator_error", "error": str(e)})
            return json.dumps({"error": str(e)})
