import sys

import structlog

from .pipeline import run_pipeline

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
        result: str = run_pipeline(user_input)
        print(f"Agent: {result}\n")
