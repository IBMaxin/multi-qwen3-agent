from typing import Any

import structlog

from .agent import create_agents
from .tools import SafeCalculatorTool

structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()


def human_approval(step_name: str, content: str) -> str:
    """Get human approval for a step.

    Args:
        step_name: Name of the step requiring approval.
        content: Content to be approved.

    Returns:
        Approved or edited content.

    Raises:
        ValueError: If human rejects the step.
    """
    logger.info({"event": "hitl_start", "step": step_name})
    print(f"\nHITL Check - {step_name}:")
    print(content)
    while True:
        user_choice: str = input("Approve? (yes/no/edit): ").strip().lower()
        if user_choice == "yes":
            logger.info({"event": "hitl_approved", "step": step_name})
            return content
        if user_choice == "no":
            logger.warning({"event": "hitl_rejected", "step": step_name})
            raise ValueError("Human stopped the pipeline.")
        if user_choice == "edit":
            new_content: str = input("Enter your edit: ")
            logger.info({"event": "hitl_edited", "step": step_name})
            return new_content
        print("Type yes, no, or edit.")


def run_pipeline(query: str) -> str:
    """Run the full pipeline for a query.

    Args:
        query: User query string.

    Returns:
        Final output or error message.
    """
    logger.info({"event": "pipeline_start", "query": query})
    tools: list[str | SafeCalculatorTool] = ["code_interpreter", SafeCalculatorTool()]
    messages: list[dict[str, str]] = [{"role": "user", "content": query}]

    try:
        manager = create_agents(tools)
        responses: list[Any] = []
        for response in manager.run(messages=messages):
            # Accumulate streaming responses; each item is a dict with at least 'content'.
            responses.append(response)
        output: str = responses[-1]["content"]
        output = human_approval("Final Output", output)
        logger.info({"event": "pipeline_complete"})
    except Exception as e:
        logger.exception({"event": "pipeline_error"})
        return f"Error: {e!s}"
    else:
        return output
