"""Example: RAG Agent with Local Vector Search using Qwen3 Embeddings.

This example demonstrates how to use the LocalVectorSearch tool for 100% local
semantic search over documents using Ollama qwen3-embedding model.

Additions in this revision:
- init_agent_service(): builds a ReActChat agent with LocalVectorSearch tool
- demo_script(): demonstrates the multi-turn pattern with messages.extend
- app_tui(): tiny terminal UI for interactive queries

Requirements:
- Python 3.10
- Ollama running locally; OpenAI-compatible endpoint at http://localhost:11434/v1
- Install production package: `cd production && pip install -e .`
"""

from __future__ import annotations

from typing import List, Dict, Any
import os
import subprocess

from qwen_agent.agents import Assistant, ReActChat
from qwen_pipeline.config import get_llm_config
from qwen_pipeline.tools_custom import LocalVectorSearch  # noqa: F401
from dotenv import load_dotenv


def example_basic_rag() -> None:
    """Basic RAG with local vector search only (Assistant agent with rag_cfg)."""
    print("\n=== Example 1: Basic Local Vector RAG ===\n")

    llm_cfg = get_llm_config()
    model_name = os.getenv("MODEL_NAME", "qwen3:4b-instruct-2507-q4_K_M")
    print(f"Using chat model: {model_name}")

    bot = Assistant(
        llm=llm_cfg,
        name="Local RAG Assistant",
        description="100% local RAG with Ollama embeddings",
        files=[],
        rag_cfg={
            "max_ref_token": 4000,
            "parser_page_size": 500,
            "rag_searchers": ["local_vector_search"],
        },
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"text": "What is this document about?"},
                {"file": "https://arxiv.org/pdf/2310.08560.pdf"},
            ],
        }
    ]

    for response in bot.run(messages):
        if response[-1]["role"] == "assistant":
            print(f"Assistant: {response[-1]['content']}\n")


def example_hybrid_rag() -> None:
    """Hybrid RAG combining vector search + keyword search."""
    print("\n=== Example 2: Hybrid RAG (Vector + Keyword) ===\n")

    llm_cfg = get_llm_config()

    bot = Assistant(
        llm=llm_cfg,
        name="Hybrid RAG Assistant",
        files=[],
        rag_cfg={
            "max_ref_token": 4000,
            "parser_page_size": 500,
            "rag_searchers": [
                "local_vector_search",
                "keyword_search",
            ],
        },
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Explain the attention mechanism"},
                {"file": "https://arxiv.org/pdf/1706.03762.pdf"},
            ],
        }
    ]

    for response in bot.run(messages):
        if response[-1]["role"] == "assistant":
            print(f"Assistant: {response[-1]['content']}\n")


def init_agent_service() -> ReActChat:
    """Initialize a ReActChat agent configured for local vector RAG.

    Returns:
        ReActChat: Agent wired with LocalVectorSearch tool.
    """
    llm_cfg = get_llm_config()

    rag_tool = LocalVectorSearch(
        cfg={
            # Default local embeddings model; override via EMBEDDING_MODEL env if desired
            "embedding_model": os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b"),
            "base_url": "http://localhost:11434",  # Ollama base URL (no /v1 suffix)
            "top_k": 3,
            "chunk_size": 500,
        }
    )

    agent = ReActChat(
        llm=llm_cfg,
        function_list=[rag_tool],
        system_message=(
            "You are a helpful assistant. If information is needed, use the "
            "local_vector_search tool to retrieve relevant passages and cite sources."
        ),
        name="rag_coder",
    )
    return agent


def run_once(agent: ReActChat, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run the agent once and return streamed responses as a list of messages."""
    responses: List[Dict[str, Any]] = []
    for chunk in agent.run(messages=messages):
        if isinstance(chunk, list):
            responses.extend(chunk)
    return responses


def demo_script() -> None:
    """Scripted multi-turn demo showing the required messages.extend pattern."""
    agent = init_agent_service()

    messages: List[Dict[str, Any]] = []

    # Turn 1
    messages.append({"role": "user", "content": "Search for: Python design patterns"})
    responses = run_once(agent, messages)
    messages.extend(responses)  # CRITICAL: extend to preserve history

    # Turn 2
    messages.append({"role": "user", "content": "Summarize the top 3 results"})
    responses = run_once(agent, messages)
    messages.extend(responses)

    # Print last assistant message if present
    if messages and messages[-1].get("role") == "assistant":
        print("\nFinal Response:\n", messages[-1].get("content", ""))


def app_tui() -> None:
    """Tiny TUI for interactive local RAG.

    Type 'exit' or 'quit' to end the session.
    """
    agent = init_agent_service()
    messages: List[Dict[str, Any]] = []

    print("Local Vector RAG TUI (type 'exit' to quit)\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        messages.append({"role": "user", "content": user})

        # Stream and append
        responses: List[Dict[str, Any]] = []
        for chunk in agent.run(messages=messages):
            if isinstance(chunk, list):
                for msg in chunk:
                    if msg.get("role") == "assistant":
                        # Print as it streams (best effort)
                        print(msg.get("content", ""), end="", flush=True)
                responses.extend(chunk)
        print()  # newline after streaming
        messages.extend(responses)


def example_multi_document() -> None:
    """Query across multiple documents with persistent memory (Assistant)."""
    llm_cfg = get_llm_config()

    # Pre-load multiple documents
    bot = Assistant(
        llm=llm_cfg,
        name="Multi-Doc Assistant",
        files=[
            "https://arxiv.org/pdf/2310.08560.pdf",  # Qwen report
            "https://arxiv.org/pdf/1706.03762.pdf",  # Attention paper
        ],
        rag_cfg={
            "max_ref_token": 4000,
            "parser_page_size": 500,
            "rag_searchers": ["local_vector_search", "keyword_search"],
        },
    )

    # Multi-turn conversation
    messages: List[Dict[str, Any]] = []

    # Turn 1: Ask about Qwen
    messages.append({"role": "user", "content": "What models are in the Qwen family?"})
    print("User: What models are in the Qwen family?")

    responses: List[Dict[str, Any]] = []
    for rsp in bot.run(messages):
        if isinstance(rsp, list):
            responses = rsp
    messages.extend(responses)  # Preserve history via extend
    print(f"Assistant: {messages[-1]['content']}\n")

    # Turn 2: Ask about attention (different document)
    messages.append({"role": "user", "content": "How does multi-head attention work?"})
    print("User: How does multi-head attention work?")

    responses = []
    for rsp in bot.run(messages):
        if isinstance(rsp, list):
            responses = rsp
    messages.extend(responses)
    print(f"Assistant: {messages[-1]['content']}\n")

    # Turn 3: Cross-reference question
    messages.append({"role": "user", "content": "Do Qwen models use multi-head attention?"})
    print("User: Do Qwen models use multi-head attention?")

    responses = []
    for rsp in bot.run(messages):
        if isinstance(rsp, list):
            responses = rsp
    messages.extend(responses)
    print(f"Assistant: {messages[-1]['content']}\n")


def example_custom_embedding_config():
    """Demonstrate custom embedding model configuration."""
    print("\n=== Example 4: Custom Embedding Configuration ===\n")

    llm_cfg = get_llm_config()

    # Create a custom-configured LocalVectorSearch instance
    # This allows you to specify custom embedding model and settings
    custom_search = LocalVectorSearch(
        cfg={
            "embedding_model": "qwen3-embedding:0.6b",  # Using faster 0.6b model
            "base_url": "http://localhost:11434",  # Ollama URL
            "max_content_length": 1500,  # Shorter chunks
        }
    )

    # Pass the custom tool instance to the Assistant
    bot = Assistant(
        llm=llm_cfg,
        name="Custom Embedding Assistant",
        files=[],
        rag_cfg={
            "max_ref_token": 4000,
            "parser_page_size": 500,
            "rag_searchers": ["local_vector_search"],  # Use the registered tool name
        },
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Summarize this document"},
                {"file": "https://arxiv.org/pdf/2310.08560.pdf"},
            ],
        }
    ]

    print("Query: Summarize this document (using faster 0.6b embedding model)")
    print("Config: Using qwen3-embedding:0.6b instead of default qwen3-embedding:4b")
    print("File: https://arxiv.org/pdf/2310.08560.pdf\n")

    for response in bot.run(messages):
        if response[-1]["role"] == "assistant":
            print(f"Assistant: {response[-1]['content']}\n")


def main():
    """Run all examples."""
    print("=" * 70)
    print("Local Vector RAG Examples with Qwen3 Embeddings")
    print("=" * 70)

    try:
        # Check dependencies
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS

        _, _ = OllamaEmbeddings, FAISS  # Mark as used
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("\nPlease install:")
        print("  pip install langchain langchain_community faiss-cpu")
        return

    # Check Ollama models

    # Load environment variables to get model names
    load_dotenv()
    chat_model = os.getenv("MODEL_NAME", "qwen3:4b-instruct-2507-q4_K_M")
    # The embedding model is hardcoded in the tool, but we check for a common one
    embedding_model_to_check = "qwen3-embedding:4b"

    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
    if chat_model not in result.stdout:
        print(f"\n❌ Chat model '{chat_model}' not found")
        print(f"Please run: ollama pull {chat_model}")
        return
    if embedding_model_to_check not in result.stdout:
        print(f"\n❌ Embedding model '{embedding_model_to_check}' not found")
        print(f"Please run: ollama pull {embedding_model_to_check}")
        return

    print("\n✅ All dependencies and models ready!\n")

    # Run examples
    example_basic_rag()
    example_hybrid_rag()
    example_multi_document()
    example_custom_embedding_config()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
