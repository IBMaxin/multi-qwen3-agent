from collections.abc import Iterator
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

    def _ensure_nonempty(resps: list[Any]) -> None:
        if not resps:
            raise ValueError("No response from agent")

    try:
        manager = create_agents(tools)
        responses: list[Any] = []
        for response in manager.run(messages=messages):
            # ✅ CORRECT: Accumulate all responses
            if isinstance(response, dict):
                responses.append(response)
            else:
                responses.append({"content": str(response)})

        _ensure_nonempty(responses)

        output: str = responses[-1].get("content", "")
        if not output:
            logger.warning("Agent returned empty content")
            output = "No response generated"

        output = human_approval("Final Output", output)
        logger.info({"event": "pipeline_complete"})
    except Exception as e:
        logger.exception({"event": "pipeline_error"})
        return f"Error: {e!s}"
    else:
        return output


def run_pipeline_streaming(
    query: str,
    *,
    require_approval: bool = True,
) -> Iterator[str]:
    """Stream pipeline responses as they are produced.

    Yields incremental content strings from the agent. On completion, optionally
    applies human approval to the final content when require_approval is True.
    """
    logger.info({"event": "pipeline_start_streaming", "query": query})
    tools: list[str | SafeCalculatorTool] = ["code_interpreter", SafeCalculatorTool()]
    messages: list[dict[str, str]] = [{"role": "user", "content": query}]

    manager = create_agents(tools)
    last: str | None = None
    for response in manager.run(messages=messages):
        content = response["content"] if isinstance(response, dict) else str(response)
        last = content
        yield content

    if last is None:
        raise ValueError("No response from agent")

    if require_approval:
        last = human_approval("Final Output", last)
    logger.info({"event": "pipeline_complete_streaming"})
