#!/usr/bin/env python3
"""
Qwen-Agent RAG Implementation
==============================

Production-ready RAG (Retrieval-Augmented Generation) system using official
Qwen-Agent patterns. Combines web search with document memory for persistent
knowledge retrieval.

Architecture:
    - Assistant agent with RAG capabilities
    - Web search for current information
    - Document indexing and retrieval
    - Conversation history management

Prerequisites:
    pip install -U "qwen-agent[rag]" chromadb
    ollama serve && ollama pull qwen3:8b
    Set SERPER_API_KEY in .env

Reference:
    https://qwen-agent.readthedocs.io/en/latest/
"""

# Standard library imports
import json
import sys
from pathlib import Path
from typing import Any

# Third-party imports
import structlog
from dotenv import load_dotenv

# Load environment variables BEFORE importing qwen_agent
load_dotenv()

# Qwen-Agent imports (after dotenv to ensure env vars are loaded)
from qwen_agent.agents import Assistant

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "production"))
from qwen_pipeline.config import get_llm_config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Note: Configuration is centralized in production/qwen_pipeline/config.py
# LLM config loaded via get_llm_config() function


# ============================================================================
# RAG AGENT
# ============================================================================


class QwenRAGAgent:
    """
    RAG-enabled agent with web search and document memory.

    Implements official Qwen-Agent RAG patterns for:
        - Web search integration
        - Document knowledge base
        - Context-aware responses
        - Citation tracking

    Attributes:
        agent: Underlying Qwen Assistant instance
        messages: Conversation history
        files: Indexed document paths
    """

    def __init__(
        self,
        files: list[str] | None = None,
        system_message: str | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        """
        Initialize RAG agent.

        Args:
            files: Document paths for knowledge base (.txt, .pdf, .md)
            system_message: Custom system prompt
            tools: Additional tools beyond web_search
        """
        self.files = self._validate_files(files or [])
        self.messages: list[dict[str, str]] = []

        if system_message is None:
            system_message = self._get_default_system_message()

        if tools is None:
            tools = ["web_search"]

        logger.info("rag_agent_init", num_files=len(self.files), tools=tools)

        llm_cfg = get_llm_config()

        self.agent = Assistant(
            llm=llm_cfg,
            system_message=system_message,
            function_list=tools,
            files=self.files,
        )

        logger.info("rag_agent_ready")

    @staticmethod
    def _validate_files(file_paths: list[str]) -> list[str]:
        """
        Validate and filter existing file paths.

        Args:
            file_paths: List of file paths to validate

        Returns:
            List of valid file paths
        """
        valid_paths = []
        for path in file_paths:
            path_obj = Path(path)
            if path_obj.exists():
                valid_paths.append(path)
                logger.debug("Valid file: %s", path)
            else:
                logger.warning("File not found: %s", path)

        return valid_paths

    @staticmethod
    def _get_default_system_message() -> str:
        """
        Get default system prompt for RAG agent.

        Returns:
            System message string
        """
        return """You are an AI research assistant with access to web search and document knowledge base.

Capabilities:
    1. Web search for current/factual information
    2. Document retrieval from knowledge base
    3. Context-aware conversation
    4. Source citation

Guidelines:
    - Use web_search for real-time data
    - Reference documents when available
    - Cite sources explicitly
    - Maintain conversation context
    - Acknowledge knowledge gaps
    - Provide structured responses"""

    def add_documents(self, file_paths: list[str]) -> None:
        """
        Add documents to knowledge base.

        Args:
            file_paths: List of document paths to add
        """
        new_files = self._validate_files(file_paths)
        self.files.extend(new_files)

        logger.info("adding_documents", count=len(new_files))

        # Recreate agent with updated files
        llm_cfg = get_llm_config()
        self.agent = Assistant(
            llm=llm_cfg,
            system_message=self.agent.system_message,
            function_list=list(self.agent.function_map.keys()),
            files=self.files,
        )

        logger.info("knowledge_base_updated", total_docs=len(self.files))

    def chat(self, query: str, stream: bool = False) -> str:
        """
        Process query with RAG and web search.

        Args:
            query: User query string
            stream: Enable streaming responses

        Returns:
            Agent response string
        """
        logger.info("Processing query: %s", query[:100])

        self.messages.append({"role": "user", "content": query})

        responses = []
        for response in self.agent.run(messages=self.messages):
            if stream and isinstance(response, list) and response:
                # Stream intermediate responses
                print(response[-1].get("content", ""), end="", flush=True)
            responses = response

        self.messages.extend(responses)

        final_response = responses[-1]["content"] if responses else ""

        logger.info("Response generated (%d chars)", len(final_response))

        return final_response

    def get_history(self) -> list[dict[str, str]]:
        """
        Retrieve conversation history.

        Returns:
            List of message dictionaries
        """
        return self.messages.copy()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        logger.info("Conversation history cleared")

    def save_history(self, output_path: str) -> None:
        """
        Save conversation history to JSON file.

        Args:
            output_path: Output file path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, indent=2, ensure_ascii=False)

        logger.info("History saved to %s", output_path)

    def load_history(self, input_path: str) -> None:
        """
        Load conversation history from JSON file.

        Args:
            input_path: Input file path
        """
        with open(input_path, encoding="utf-8") as f:
            self.messages = json.load(f)

        logger.info("History loaded from %s (%d messages)", input_path, len(self.messages))


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


def example_basic_rag() -> None:
    """Demonstrate basic RAG workflow."""
    logger.info("Starting basic RAG example")

    agent = QwenRAGAgent()

    # Query 1: Web search for current info
    print("\n" + "=" * 60)
    print("Query 1: Web Search")
    print("=" * 60)

    query1 = "What are the latest features in Qwen-Agent framework?"
    print(f"\nUser: {query1}")
    response1 = agent.chat(query1)
    print(f"\nAgent: {response1}")

    # Query 2: Follow-up using context
    print("\n" + "=" * 60)
    print("Query 2: Context Retrieval")
    print("=" * 60)

    query2 = "Can you summarize the key points from your previous answer?"
    print(f"\nUser: {query2}")
    response2 = agent.chat(query2)
    print(f"\nAgent: {response2}")

    logger.info("Basic RAG example completed")


def example_document_rag() -> None:
    """Demonstrate RAG with document knowledge base."""
    logger.info("Starting document RAG example")

    # Create sample documents
    docs_dir = Path("./rag_docs")
    docs_dir.mkdir(exist_ok=True)

    sample_doc = docs_dir / "qwen_overview.txt"
    sample_doc.write_text(
        """Qwen-Agent Framework Overview

Qwen-Agent is an official agent framework from Alibaba's Qwen team.

Key Features:
- Multi-agent orchestration via GroupChat
- Built-in tools: web_search, code_interpreter
- RAG capabilities with document retrieval
- MCP (Model Context Protocol) support
- Gradio WebUI integration

Architecture:
- Assistant: Basic agent with tool access
- ReActChat: Agent with step-by-step reasoning
- GroupChat: Multi-agent coordination

Official Repository: https://github.com/QwenLM/Qwen-Agent
"""
    )

    logger.info("Sample document created: %s", sample_doc)

    # Create agent with document
    agent = QwenRAGAgent(files=[str(sample_doc)])

    print("\n" + "=" * 60)
    print("Query: Document + Web Search")
    print("=" * 60)

    query = "Based on the Qwen-Agent documentation, what agent types are available?"
    print(f"\nUser: {query}")
    response = agent.chat(query)
    print(f"\nAgent: {response}")

    logger.info("Document RAG example completed")


def example_interactive_rag() -> None:
    """Interactive RAG session."""
    logger.info("Starting interactive RAG session")

    agent = QwenRAGAgent()

    print("\n" + "=" * 60)
    print("Interactive RAG Chat")
    print("=" * 60)
    print("\nCommands:")
    print("  quit/exit - End session")
    print("  history   - Show conversation history")
    print("  clear     - Clear conversation history")
    print("\n")

    while True:
        try:
            user_input = input("User: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                logger.info("Ending interactive session")
                break

            if user_input.lower() == "history":
                history = agent.get_history()
                print(f"\nConversation history ({len(history)} messages):")
                for i, msg in enumerate(history, 1):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")[:100]
                    print(f"  {i}. {role}: {content}...")
                print()
                continue

            if user_input.lower() == "clear":
                agent.clear_history()
                print("History cleared.\n")
                continue

            if not user_input:
                continue

            response = agent.chat(user_input)
            print(f"\nAgent: {response}\n")

        except KeyboardInterrupt:
            logger.info("Interactive session interrupted")
            break
        except Exception as e:
            logger.error("Error in interactive session: %s", e, exc_info=True)
            print(f"\nError: {e}\n")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Main entry point."""
    print(
        """
╔══════════════════════════════════════════════════════════╗
║   QWEN-AGENT RAG SYSTEM                                  ║
║   Web Search + Document Memory                           ║
╚══════════════════════════════════════════════════════════╝

Select example:
  1. Basic RAG (web search + context)
  2. Document RAG (knowledge base)
  3. Interactive RAG session

"""
    )

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        example_basic_rag()
    elif choice == "2":
        example_document_rag()
    elif choice == "3":
        example_interactive_rag()
    else:
        print("Invalid choice. Running basic RAG example.")
        example_basic_rag()


if __name__ == "__main__":
    main()
