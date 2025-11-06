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
                    logger.info("CLI exiting gracefully.")
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
                logger.warning("User rejected or validation failed", error=str(e))
                print(f"Operation cancelled: {e}\n")
                error_count += 1

            except Exception as e:
                # Unexpected error but continue
                logger.exception("Unexpected error in pipeline")
                print(f"Error: {e}\nTrying again...\n")
                error_count += 1

                if error_count >= max_errors:
                    logger.exception("Too many errors, exiting")
                    print(f"Too many errors ({max_errors}). Exiting.")
                    exit_fn = cast("Callable[[int], None]", sys.exit)
                    exit_fn(1)
                    break

    except KeyboardInterrupt:
        logger.info("CLI interrupted by user (Ctrl+C).")
        print("\n\nGoodbye!")
        sys.exit(0)
    except EOFError:
        # Handle EOF (e.g., when piping input)
        logger.info("CLI reached EOF.")
        sys.exit(0)


def main() -> None:
    """CLI entry point; parses args and dispatches commands."""
    logger.info("CLI started.")
    args = _parse_args(sys.argv[1:])
    if args.command == "metrics":
        print(_get_metrics_json())
        exit_fn = cast("Callable[[int], None]", sys.exit)
        exit_fn(0)
        return
    _run_interactive(args)
