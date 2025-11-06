#!/usr/bin/env python
"""
Run a Qwen-Agent GroupChat with all available tools (excluding those requiring API keys
unless set). Interactive CLI with a simple prompt loop.

Prereqs:
- Ollama running and qwen3:8b pulled (or compatible model)
- .env configured with MODEL_SERVER=http://localhost:11434/v1, MODEL_NAME=qwen3:8b, API_KEY=EMPTY
- production package installed: from repo root -> cd production && pip install -e .
"""

import sys

from qwen_pipeline.agent import create_agents_all_tools_no_keys


def main() -> None:
    agent = create_agents_all_tools_no_keys(enable_vl=False, enable_mcp=False)
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
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            return
        history.append({"role": "user", "content": user})
        # Run the agent and capture the last message content
        last_content = None
        for resp in agent.run(messages=history):
            if isinstance(resp, dict):
                last_content = resp.get("content", None)
            else:
                last_content = str(resp)
        if last_content is None:
            print("(no response)\n")
            continue
        history.append({"role": "assistant", "content": last_content})
        print(f"Agent: {last_content}\n")


if __name__ == "__main__":
    sys.exit(main())
