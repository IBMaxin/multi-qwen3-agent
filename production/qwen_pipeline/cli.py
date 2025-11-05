import sys

import structlog


# Lightweight indirection to avoid importing heavy pipeline/qwen_agent at module import time.
def run_pipeline(query: str) -> str:  # pragma: no cover - thin wrapper
    from .pipeline import run_pipeline as _run  # noqa: PLC0415 - intentional lazy import

    return _run(query)


structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()


def main() -> None:
    """Main CLI entry point with robust error handling."""
    logger.info("CLI started.")
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
                    sys.exit(0)
                    # In tests, sys.exit is patched and won't raise; ensure we exit the loop.
                    break  # type: ignore[unreachable]

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
                    sys.exit(1)
                    break  # type: ignore[unreachable]

    except KeyboardInterrupt:
        logger.info("CLI interrupted by user (Ctrl+C).")
        print("\n\nGoodbye!")
        sys.exit(0)
    except EOFError:
        # Handle EOF (e.g., when piping input)
        logger.info("CLI reached EOF.")
        sys.exit(0)
