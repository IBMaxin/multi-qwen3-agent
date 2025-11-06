"""Core pipeline execution logic for the Qwen-Agent system.

Provides functions to run the multi-agent pipeline in different modes:
- `run_pipeline`: Standard execution, returns final result after HITL.
- `run_pipeline_streaming`: Yields results incrementally.
- `run_pipeline_structured`: Returns a detailed result object with metadata.

Includes Human-In-The-Loop (HITL) for approval and a timeout mechanism.
"""

import time
import types
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any, Literal

import structlog

from .agent import create_agents
from .metrics import get_metrics
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


class PipelineTimeoutError(Exception):
    """Raised when pipeline exceeds the configured timeout."""


class PipelineTimeout:
    """Lightweight timeout guard using periodic checks.

    Windows-safe alternative to signal.alarm; call check() in loops.
    """

    def __init__(self, seconds: int = 60):
        self.seconds = max(0, int(seconds))
        self._start: float = 0.0

    def __enter__(self) -> "PipelineTimeout":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> Literal[False]:
        # Don't suppress exceptions
        return False

    def check(self) -> None:
        if self.seconds == 0:
            # Immediate timeout if enabled and set to 0
            msg = f"Pipeline exceeded {self.seconds}s timeout"
            raise PipelineTimeoutError(msg)
        if self.seconds > 0:
            elapsed = time.perf_counter() - self._start
            if elapsed >= self.seconds:
                msg = f"Pipeline exceeded {self.seconds}s timeout"
                raise PipelineTimeoutError(msg)


def run_pipeline(query: str, timeout_seconds: int = 60) -> str:
    """Run the full pipeline for a query.

    Args:
        query: User query string.
        timeout_seconds: Max execution time (0 = immediate timeout, <0 disables checks).

    Returns:
        Final output or error message.
    """
    logger.info({"event": "pipeline_start", "query": query})
    metrics = get_metrics()
    metrics.record_query_start()
    tools: list[str | SafeCalculatorTool] = ["code_interpreter", SafeCalculatorTool()]
    messages: list[dict[str, str]] = [{"role": "user", "content": query}]

    def _ensure_nonempty(resps: list[Any]) -> None:
        if not resps:
            raise ValueError("No response from agent")

    try:
        manager = create_agents(tools)
        responses: list[Any] = []
        with PipelineTimeout(timeout_seconds) as pt:
            # Initial check (handles 0s immediate timeout deterministically)
            pt.check()
            for response in manager.run(messages=messages):
                # ✅ CORRECT: Accumulate all responses
                if isinstance(response, dict):
                    responses.append(response)
                else:
                    responses.append({"content": str(response)})
                # Periodic timeout check per-chunk
                pt.check()

        _ensure_nonempty(responses)

        output: str = responses[-1].get("content", "")
        if not output:
            logger.warning("agent_empty_content", last_response=responses[-1])
            output = "No response generated"

        output = human_approval("Final Output", output)
        logger.info({"event": "pipeline_complete"})
        metrics.record_query_end(success=True)
    except PipelineTimeoutError as e:
        logger.exception({"event": "pipeline_timeout"})
        metrics.record_query_end(success=False, error_type=type(e).__name__)
        return f"Error: Pipeline timeout - {e!s}"
    except Exception as e:
        logger.exception({"event": "pipeline_error"})
        metrics.record_query_end(success=False, error_type=type(e).__name__)
        return f"Error: {e!s}"
    else:
        return output


def run_pipeline_streaming(
    query: str,
    *,
    require_approval: bool = True,
    timeout_seconds: int = -1,
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
    # If timeout_seconds < 0, disable timeout checks for streaming
    if timeout_seconds < 0:
        for response in manager.run(messages=messages):
            content = response["content"] if isinstance(response, dict) else str(response)
            last = content
            yield content
    else:
        with PipelineTimeout(timeout_seconds) as pt:
            pt.check()  # early/immediate timeout support
            for response in manager.run(messages=messages):
                content = response["content"] if isinstance(response, dict) else str(response)
                last = content
                yield content
                pt.check()

    if last is None:
        raise ValueError("No response from agent")

    if require_approval:
        last = human_approval("Final Output", last)
    logger.info({"event": "pipeline_complete_streaming"})


# Structured output mode
@dataclass
class PipelineResult:
    response: str
    query: str
    duration_seconds: float
    success: bool
    error: str | None = None
    agents_used: list[str] | None = None
    tools_available: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_pipeline_structured(query: str, timeout_seconds: int = 60) -> PipelineResult:
    """Run pipeline and return structured result with metadata.

    Does not alter existing run_pipeline behavior; provided as an additive API.
    """
    start = time.perf_counter()
    agents_used = ["planner", "coder", "reviewer"]
    tools: list[str | SafeCalculatorTool] = ["code_interpreter", SafeCalculatorTool()]
    tools_available = len(tools)
    try:
        output = run_pipeline(query, timeout_seconds=timeout_seconds)
        success = not output.startswith("Error:")
        err: str | None = None if success else output
        return PipelineResult(
            response=output if success else "",
            query=query,
            duration_seconds=time.perf_counter() - start,
            success=success,
            error=err,
            agents_used=agents_used,
            tools_available=tools_available,
        )
    except Exception as e:  # pragma: no cover - defensive guard
        return PipelineResult(
            response="",
            query=query,
            duration_seconds=time.perf_counter() - start,
            success=False,
            error=str(e),
            agents_used=agents_used,
            tools_available=tools_available,
        )
