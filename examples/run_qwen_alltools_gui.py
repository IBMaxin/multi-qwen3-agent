#!/usr/bin/env python
"""
Launch Qwen-Agent Gradio WebUI with a GroupChat using all available tools (excluding
those requiring API keys unless configured).

Prereqs:
- Ollama running and qwen3:8b pulled
- .env configured with MODEL_SERVER=http://localhost:11434/v1, MODEL_NAME=qwen3:8b, API_KEY=EMPTY
- production package installed: from repo root -> cd production && pip install -e .
"""

from qwen_agent.gui import WebUI
from qwen_pipeline.agent import create_agents_all_tools_no_keys


def main() -> None:
    agent = create_agents_all_tools_no_keys(enable_vl=False, enable_mcp=False)
    chatbot_config = {
        "user.name": "Developer",
        "input.placeholder": "Ask me anything...",
        "prompt.suggestions": [
            "Summarize a web page",
            "Analyze this code for issues",
            "Extract key terms from this document",
        ],
    }
    ui = WebUI(agent=agent, chatbot_config=chatbot_config)
    print("Starting Qwen All-Tools GUI at http://127.0.0.1:7860")
    ui.run(server_name="127.0.0.1", server_port=7860, share=False, concurrency_limit=10)


if __name__ == "__main__":
    main()
