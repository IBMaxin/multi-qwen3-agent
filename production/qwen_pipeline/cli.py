"""Command-line interface for the Qwen-Agent pipeline.

Provides an interactive chat CLI for querying the agent pipeline, with support
for streaming responses and other configurations. Also includes commands for
launching a Gradio WebUI and inspecting performance metrics.

Features:
- Interactive chat loop
- Streaming mode (`--stream`)
- Timeout configuration (`--timeout`)
- Metrics reporting (`metrics` command)
- Gradio GUI launcher (`gui` command)
"""

import argparse
import sys
from collections.abc import Callable, Iterator
from typing import cast

import structlog


# Lightweight indirection to avoid importing heavy pipeline/qwen_agent at module import time.
def run_pipeline(query: str) -> str:  # pragma: no cover - thin wrapper
    from .pipeline import run_pipeline as _run  # noqa: PLC0415 - intentional lazy import

    return _run(query)


def run_pipeline_streaming(
    query: str, *, timeout_seconds: int = -1
) -> Iterator[str]:  # pragma: no cover - wrapper
    from .pipeline import run_pipeline_streaming as _run_stream  # noqa: PLC0415

    return _run_stream(query, timeout_seconds=timeout_seconds)


def _get_metrics_json() -> str:  # pragma: no cover - thin wrapper
    from .metrics import get_metrics  # noqa: PLC0415

    return get_metrics().to_json()


structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pipeline",
        description="Qwen pipeline CLI",
        add_help=True,
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("metrics", help="Print current metrics and exit")

    gui_parser = sub.add_parser("gui", help="Launch Gradio WebUI for the Qwen Agent")
    gui_parser.add_argument("--host", default="127.0.0.1", help="Host/IP to bind the GUI server")
    gui_parser.add_argument("--port", type=int, default=7860, help="Port for the GUI server")
    gui_parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio share (tunnels public URL). Use only if you understand the risks.",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode (prints chunks as they arrive)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for pipeline (use -1 to disable, 0 for immediate)",
    )
    return parser.parse_args(argv)


def _run_interactive(args: argparse.Namespace) -> None:  # noqa: PLR0912, PLR0915
    """Main CLI entry point with robust error handling."""
    print("Qwen Pipeline ready. Type query or 'exit'.")

    error_count = 0
    max_errors = 3

    try:
        while True:
            try:
                user_input: str = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "bye"):
                    logger.info("cli_exit_graceful")
                    print("Goodbye!")
                    exit_fn = cast("Callable[[int], None]", sys.exit)
                    exit_fn(0)
                    break

                if args.stream:
                    # Stream chunks inline
                    try:
                        for chunk in run_pipeline_streaming(
                            user_input, timeout_seconds=int(args.timeout)
                        ):
                            print(f"Agent: {chunk}")
                        print("")
                    finally:
                        error_count = 0
                else:
                    result: str = run_pipeline(user_input)
                    print(f"Agent: {result}\n")
                    error_count = 0  # Reset on success

            except (EOFError, KeyboardInterrupt):
                # Re-raise to outer handler
                raise

            except ValueError as e:
                # HITL rejection or validation error
                logger.warning("cli_user_rejection_or_validation_error", error=str(e))
                print(f"Operation cancelled: {e}\n")
                error_count += 1

            except Exception:
                # Unexpected error but continue
                logger.exception("cli_unexpected_pipeline_error")
                print("An unexpected error occurred. Trying again...\n")
                error_count += 1

                if error_count >= max_errors:
                    logger.exception("cli_max_errors_reached", max_errors=max_errors)
                    print(f"Too many errors ({max_errors}). Exiting.")
                    exit_fn = cast("Callable[[int], None]", sys.exit)
                    exit_fn(1)
                    break

    except KeyboardInterrupt:
        logger.info("cli_interrupt_user")
        print("\n\nGoodbye!")
        sys.exit(0)
    except EOFError:
        # Handle EOF (e.g., when piping input)
        logger.info("cli_eof_received")
        sys.exit(0)


def main() -> None:
    """CLI entry point; parses args and dispatches commands."""
    logger.info("cli_started")
    args = _parse_args(sys.argv[1:])
    if args.command == "metrics":
        print(_get_metrics_json())
        exit_fn = cast("Callable[[int], None]", sys.exit)
        exit_fn(0)
        return
    if args.command == "gui":
        # Lazy imports to keep CLI startup fast
        from qwen_agent.gui import WebUI  # noqa: PLC0415

        from .agent import create_agents  # noqa: PLC0415

        tools: list[str] = [
            "code_interpreter",
            "safe_calculator",  # provided by our SafeCalculatorTool registration
        ]
        agent = create_agents(tools)

        chatbot_config = {
            "user.name": "You",
            "input.placeholder": "Ask me anything...",
        }
        web_ui = WebUI(agent=agent, chatbot_config=chatbot_config)
        logger.info(
            "gui_starting",
            host=str(getattr(args, "host", "127.0.0.1")),
            port=int(getattr(args, "port", 7860)),
            share=bool(getattr(args, "share", False)),
        )
        web_ui.run(
            server_name=str(getattr(args, "host", "127.0.0.1")),
            server_port=int(getattr(args, "port", 7860)),
            share=bool(getattr(args, "share", False)),
            concurrency_limit=10,
        )
        return
    _run_interactive(args)
