#!/usr/bin/env python
"""
Official Qwen Agent GUI with Qwen3:8B + Ollama Integration.

This launches the official Qwen Agent web UI (Gradio-based) with your
qwen3-8b model via Ollama, integrated with MCP tools.
"""

import os
import sys
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print(">>> Initializing Qwen Agent GUI...")

# Import Qwen agent components
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

# Configure the LLM to use Ollama with Qwen3:8B
llm_config = {
    # Model name - using qwen3:8b locally via Ollama
    "model": "qwen3-8b",
    # Use Ollama as the model server (OpenAI-compatible API)
    "model_server": "http://localhost:11434/v1",
    "api_key": "EMPTY",  # Ollama doesn't require an API key
    # Generation parameters
    "generate_cfg": {
        "top_p": 0.9,
        "temperature": 0.7,
        "max_tokens": 2048,
    },
}

# Create system instruction for code analysis assistant
system_instruction = """You are a professional Python code analysis and optimization assistant powered by Qwen3:8B.

Your expertise includes:
1. **Code Analysis**: Identifying errors, type issues, and code quality problems
2. **Performance Optimization**: Suggesting performance improvements and best practices
3. **Security Review**: Checking for security vulnerabilities and recommending fixes
4. **Code Explanation**: Explaining code patterns and providing educational insights
5. **Refactoring**: Suggesting code improvements for readability and maintainability

When analyzing code:
- Be thorough and specific in your findings
- Provide actionable recommendations
- Explain the reasoning behind your suggestions
- Consider performance, security, and maintainability
- Suggest best practices and patterns

Format your responses clearly with:
- **Issues Found**: List any problems
- **Recommendations**: Specific improvements
- **Code Examples**: When helpful, provide improved code snippets
- **Best Practices**: Relevant guidelines

Always be helpful and constructive in your feedback."""

print(">>> Creating Assistant agent...")

# Create the assistant agent with your configuration
# Use no built-in tools to avoid dependency issues
agent = Assistant(
    llm=llm_config,
    system_message=system_instruction,
    name="PylanceCodeAssistant",
    description="AI-powered Python code analysis and optimization assistant",
    function_list=[],  # No built-in tools, keep it simple
)

print(">>> Creating Gradio UI...")

# Optional: Configure the chatbot UI
chatbot_config = {
    "user.name": "Developer",
    "input.placeholder": "Ask me to analyze or optimize your Python code...",
    "prompt.suggestions": [
        "Analyze this file for errors and issues",
        "Suggest optimizations for performance",
        "Review for security vulnerabilities",
        "Explain this code pattern",
        "Help me refactor this function",
    ],
}

# Create and launch the web UI
web_ui = WebUI(
    agent=agent,
    chatbot_config=chatbot_config,
)

print("\n" + "=" * 60)
print("SUCCESS: Qwen Agent GUI is starting...")
print("=" * 60)
print("\nAccess the UI at: http://127.0.0.1:7860")
print("Model: Qwen3:8B (via Ollama)")
print("Server: http://localhost:11434/v1")
print("\nWARNING: Make sure Ollama is running: ollama serve")
print("Have code to analyze? Paste it in the chat!")
print("=" * 60 + "\n")

# Launch the GUI
if __name__ == "__main__":
    web_ui.run(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        concurrency_limit=10,
    )
