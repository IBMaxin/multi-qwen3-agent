#!/usr/bin/env python3
"""
QWEN-AGENT COMPLETE IMPLEMENTATION (UPGRADED)
=============================================
A fully-featured agentic assistant using Qwen-Agent framework
with all pre-made tools. Configured for Ollama with Qwen3:8b.

This UPGRADED version includes:
1. Official web_search tool (using Serper.dev API)
2. A SECURE CalculatorTool (using asteval)

Prerequisites:
1. Install Ollama: https://ollama.com/download
2. Pull Qwen3: ollama pull qwen3:8b
3. Install dependencies:
   pip install -U "qwen-agent[gui,rag,code_interpreter,mcp]" asteval
4. Configure .env with SERPER_API_KEY (from https://serper.dev)
5. Start Ollama: ollama serve (runs on http://localhost:11434)
"""

import json

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import os
import urllib.parse

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import json5
from asteval import Interpreter  # <-- UPGRADE 1: Secure math evaluator
from qwen_agent.agents import Assistant, ReActChat
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_CONFIG = {
    "model": "qwen3:8b",  # Options: qwen3:4b, qwen3:8b, qwen3:14b
    "model_server": "http://localhost:11434/v1",
    "api_key": "EMPTY",
    "generate_cfg": {
        "top_p": 0.8,
        "temperature": 0.7,
        "max_input_tokens": 6000,
        "fncall_prompt_type": "nous",
    },
}


# ============================================================================
# CUSTOM TOOLS (UPGRADED)
# ============================================================================


class MyImageGen(BaseTool):
    """Custom image generation using free Pollinations API"""

    name = "my_image_gen"
    description = "AI painting service. Input text description, " "return image URL."
    parameters = [
        {
            "name": "prompt",
            "type": "string",
            "description": "Detailed image description in English",
            "required": True,
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        prompt = json5.loads(params)["prompt"]
        prompt_encoded = urllib.parse.quote(prompt)
        image_url = f"https://image.pollinations.ai/prompt/{prompt_encoded}"
        return json.dumps({"image_url": image_url, "prompt": prompt}, ensure_ascii=False)


class CalculatorTool(BaseTool):
    """
    [UPGRADED] Advanced calculator with math functions, using a safe evaluator.
    """

    name = "calculator"
    description = (
        "Perform mathematical calculations with Python math functions. "
        "Uses a safe evaluator (asteval)."
    )
    parameters = [
        {
            "name": "expression",
            "type": "string",
            "description": 'Math expression (e.g., "sqrt(16)", "sin(pi/2)")',
            "required": True,
        }
    ]

    def __init__(self, *args, **kwargs):
        """Initialize the safe asteval interpreter."""
        super().__init__(*args, **kwargs)
        # Create a persistent, safe interpreter
        self.aeval = Interpreter()
        # asteval includes 'sqrt', 'sin', 'cos', 'tan', 'pi', 'e' by default

    def call(self, params: str, **kwargs) -> str:
        expression = json5.loads(params)["expression"]
        try:
            # Use the safe aeval.eval() instead of Python's eval()
            result = self.aeval.eval(expression)

            # Clear potential errors from the interpreter for the next run
            self.aeval.err_text = ""
            self.aeval.error_msg = ""

            return json.dumps({"result": result, "expression": expression}, ensure_ascii=False)
        except Exception as e:
            # Catch any evaluation errors
            return json.dumps(
                {"error": str(e), "asteval_error": self.aeval.err_text}, ensure_ascii=False
            )


class FileSystemTool(BaseTool):
    """Safe file system operations"""

    name = "filesystem"
    description = "List directory contents or read text files (read-only)."
    parameters = [
        {
            "name": "operation",
            "type": "string",
            "description": 'Operation: "list" or "read"',
            "required": True,
        },
        {
            "name": "path",
            "type": "string",
            "description": "Directory or file path",
            "required": True,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        params_dict = json5.loads(params)
        operation = params_dict["operation"]
        path = params_dict["path"]

        try:
            if operation == "list":
                items = os.listdir(path)
                return json.dumps({"path": path, "items": items}, ensure_ascii=False)
            elif operation == "read":
                with open(path, encoding="utf-8") as f:
                    content = f.read(5000)  # Limit to 5000 chars
                return json.dumps({"path": path, "content": content}, ensure_ascii=False)
            else:
                return json.dumps({"error": "Invalid operation"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)


# ============================================================================
# MCP CONFIGURATION (Optional - requires Node.js and uv)
# ============================================================================

MCP_CONFIG = {
    "mcpServers": {
        "time": {"command": "uvx", "args": ["mcp-server-time", "--local-timezone=UTC"]},
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
    }
}


# ============================================================================
# AGENT FACTORY FUNCTIONS
# ============================================================================


def create_assistant_agent(
    tools: list | None = None,
    files: list | None = None,
    system_message: str | None = None,
    use_mcp: bool = False,
) -> Assistant:
    """
    Create full-featured Assistant agent.

    Args:
        tools: Tool list (defaults to all custom + built-in tools)
        files: File paths for document access
        system_message: Custom system prompt
        use_mcp: Include MCP server tools
    """

    if system_message is None:
        system_message = """You are a helpful AI assistant powered by Qwen3.

You have access to:
- Code execution (Python)
- Image generation
- REAL Web search
- SECURE Mathematical calculations
- File system access

When helping users:
1. Understand their needs
2. Use appropriate tools
3. Provide clear explanations
4. Show your reasoning

Be helpful, accurate, and friendly."""

    if tools is None:
        tools = [
            "code_interpreter",  # Built-in: Python execution
            "web_search",  # Built-in: Web search via Serper.dev
            MyImageGen(),  # Custom: Image generation
            CalculatorTool(),  # Custom: Calculator (SECURE with asteval)
            FileSystemTool(),  # Custom: File operations
        ]

        if use_mcp:
            tools.insert(0, MCP_CONFIG)

    return Assistant(
        llm=OLLAMA_CONFIG,
        system_message=system_message,
        function_list=tools,
        files=files or [],
    )


def create_react_agent(tools: list | None = None, system_message: str | None = None) -> ReActChat:
    """Create ReAct agent for step-by-step reasoning."""

    if tools is None:
        tools = ["code_interpreter", "web_search", MyImageGen(), CalculatorTool()]

    if system_message is None:
        system_message = """You use step-by-step reasoning to solve problems.

Process:
1. Break down the task
2. Think through each step
3. Use tools as needed
4. Verify results
5. Provide clear answers"""

    return ReActChat(
        llm=OLLAMA_CONFIG,
        system_message=system_message,
        function_list=tools,
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


def example_basic_chat():
    """Basic chat with tool usage"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Basic Chat with Tools")
    print("=" * 60 + "\n")

    agent = create_assistant_agent()
    messages = []

    # Test new calculator
    query_1 = "What is (sqrt(144) + 50) / 2?"
    print(f"User: {query_1}")
    messages.append({"role": "user", "content": query_1})

    responses = []
    for response in agent.run(messages=messages):
        responses = response
    messages.extend(responses)
    print(f"Agent: {responses[-1]['content']}\n")

    # Test new web search
    query_2 = "What's the latest news about the Qwen model?"
    print(f"User: {query_2}")
    messages.append({"role": "user", "content": query_2})

    responses = []
    for response in agent.run(messages=messages):
        responses = response
    messages.extend(responses)
    print(f"Agent: {responses[-1]['content']}\n")


def example_code_execution():
    """Code interpreter example"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Code Execution")
    print("=" * 60 + "\n")

    agent = create_assistant_agent()

    query = """Create a bar chart showing programming languages:
Python: 30%, JavaScript: 25%, Java: 20%, C++: 15%, Other: 10%
Save as 'languages.png'"""

    print(f"User: {query}")
    messages = [{"role": "user", "content": query}]

    responses = []
    for response in agent.run(messages=messages):
        responses = response
    print(f"Agent: {responses[-1]['content']}\n")


def example_interactive():
    """Interactive chat loop"""
    print("\n" + "=" * 60)
    print("INTERACTIVE CHAT - Type 'quit' to exit")
    print("=" * 60 + "\n")

    agent = create_assistant_agent()
    messages = []

    print(
        "Agent: Hello! I'm your upgraded AI assistant. "
        "I can now search the web and calculate safely!\n"
    )

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "bye", "q"]:
                print("\nAgent: Goodbye! Have a great day!")
                break

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            responses = []
            print("\nAgent: ...thinking...")
            for response in agent.run(messages=messages):
                responses = response

            messages.extend(responses)
            print(f"\nAgent: {responses[-1]['content']}\n")

        except KeyboardInterrupt:
            print("\n\nAgent: Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def launch_web_ui():
    """Launch Gradio web interface"""
    print("\n" + "=" * 60)
    print("LAUNCHING WEB UI on http://localhost:7860")
    print("=" * 60 + "\n")

    agent = create_assistant_agent()
    ui = WebUI(agent)
    ui.run(server_name="0.0.0.0", server_port=7860)


# ============================================================================
# MAIN
# ============================================================================


def main():
    print(
        """
    ╔══════════════════════════════════════════════════════════╗
    ║   QWEN-AGENT (UPGRADED)                                  ║
    ║   Powered by Ollama + Qwen3 8B                           ║
    ║   - SECURE Calculator                                    ║
    ║   - REAL Web Search                                      ║
    ╚══════════════════════════════════════════════════════════╝
    """
    )

    print("Available modes:")
    print("1. Interactive Chat (recommended)")
    print("2. Basic Example (runs calc and web search)")
    print("3. Code Execution Example")
    print("4. Launch Web UI")
    print("5. Exit")

    while True:
        choice = input("\nSelect mode (1-5): ").strip()

        if choice == "1":
            example_interactive()
            break
        elif choice == "2":
            example_basic_chat()
            break
        elif choice == "3":
            example_code_execution()
            break
        elif choice == "4":
            launch_web_ui()
            break
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()
