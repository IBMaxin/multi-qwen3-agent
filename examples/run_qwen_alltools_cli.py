#!/usr/bin/env python
"""
Run a Qwen-Agent chat with all available tools (excluding those requiring API keys
unless set). Interactive CLI with a simple prompt loop.

Use --fast to run a single ReAct agent (no GroupChat) for lower latency.

Prereqs:
- Ollama running and the chat model specified in your .env file pulled.
- .env configured with MODEL_SERVER, MODEL_NAME, and API_KEY.
- production package installed: from repo root -> cd production && pip install -e .
"""

import argparse
import sys

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

from qwen_pipeline.agent import (
    create_agents_all_tools_no_keys,
    create_fast_agent_all_tools_no_keys,
)


def _extract_last_assistant_message(resp: object, *, require_name: bool) -> dict | None:
    """Return the last assistant message dict (must include a name) from a response item.

    GroupChat typically yields intermediate dict chunks and ends with a list of
    message dicts. We must keep the assistant message dict (with `name`) to feed
    back into history; otherwise GroupChat will assert.
    """
    # Case 1: single message dict
    if isinstance(resp, dict):  # type: ignore[redundant-cast]
        if resp.get("role") == "assistant":
            if not require_name or isinstance(resp.get("name"), str):
                return resp
        return None
    # Case 2: final list of messages
    if isinstance(resp, list):
        for item in reversed(resp):
            if isinstance(item, dict) and item.get("role") == "assistant":
                if not require_name or isinstance(item.get("name"), str):
                    return item
        return None
    # Ignore string chunks for history purposes; they lack required metadata
    return None


def _clean_display(text: str) -> str:
    """Reduce chain-of-thought noise in displayed outputs.

    If the model emits markers like "Thought:" and "Final Answer:", keep the final
    answer section when present; otherwise, drop leading 'Thought:' lines.
    """
    if not isinstance(text, str) or not text:
        return text
    # Prefer content after the last 'Final Answer:' if present.
    marker = "Final Answer:"
    if marker in text:
        return text.split(marker)[-1].strip()
    # Otherwise, remove lines starting with 'Thought:'
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("Thought:")]
    cleaned = "\n".join(lines).strip()
    return cleaned or text


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-Agent All-Tools CLI")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a single ReAct agent (no GroupChat) for faster responses",
    )
    args = parser.parse_args()

    if args.fast:
        agent = create_fast_agent_all_tools_no_keys(enable_vl=False, enable_mcp=False)
        require_name = False
    else:
        agent = create_agents_all_tools_no_keys(enable_vl=False, enable_mcp=False)
        require_name = True
    print("Qwen All-Tools CLI. Type 'exit' to quit.\n")
    history: list[dict[str, str]] = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if not user:
            continue
        if user.lower() in {"exit", "quit", "/bye", "bye"}:
            print("Bye!")
            return
        history.append({"role": "user", "content": user})
        # Run the agent and capture the last message content
        last_msg: dict | None = None
        # Stream partial assistant content for better perceived latency
        printed_len = 0
        for resp in agent.run(messages=history):
            # Print incremental assistant content when available
            if isinstance(resp, dict) and resp.get("role") == "assistant":
                content_chunk = resp.get("content")
                if isinstance(content_chunk, str) and content_chunk:
                    delta = content_chunk[printed_len:]
                    if delta:
                        # Print without newline to stream
                        print(delta, end="", flush=True)
                        printed_len = len(content_chunk)
            candidate = _extract_last_assistant_message(resp, require_name=require_name)
            if candidate:
                last_msg = candidate
        # Finalize this turn
        if last_msg is None:
            print("(no response)\n")
            continue
        # Normalize content for printing/appending
        content = last_msg.get("content")
        printable = content if isinstance(content, str) else str(content)
        # If no streaming happened, print the full content once prefixed with Agent:
        if printed_len == 0:
            print(f"Agent: {_clean_display(printable)}\n")
        else:
            # Finish the streaming line with a newline for cleanliness
            print()
        # Append assistant message back to history.
        # For GroupChat we must include the `name`; for fast mode we only need role/content.
        if require_name and isinstance(last_msg.get("name"), str):
            history.append(last_msg)
        else:
            history.append({"role": "assistant", "content": _clean_display(printable)})


if __name__ == "__main__":
    sys.exit(main())
