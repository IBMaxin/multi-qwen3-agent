import sys

import structlog


# Lightweight indirection to avoid importing heavy pipeline/qwen_agent at module import time.
def run_pipeline(query: str) -> str:  # pragma: no cover - thin wrapper
    from .pipeline import run_pipeline as _run  # noqa: PLC0415 - intentional lazy import

    return _run(query)


structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()


def main() -> None:
    """Main CLI entry point."""
    logger.info("CLI started.")
    print("Qwen Pipeline ready. Type query or 'exit'.")
    while True:
        user_input: str = input("You: ").strip()
        if user_input.lower() == "exit":
            logger.info("CLI exiting.")
            print("Goodbye!")
            sys.exit(0)
            # In tests, sys.exit is patched and won't raise; ensure we exit the loop.
            break  # type: ignore[unreachable]
        result: str = run_pipeline(user_input)
        print(f"Agent: {result}\n")
