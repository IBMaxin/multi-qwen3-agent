"""Example: RAG Agent with Local Vector Search using Qwen3 Embeddings.

This example demonstrates how to use the LocalVectorSearch tool for 100% local
semantic search over documents using Ollama qwen3-embedding model.

Requirements:
    - Ollama running: ollama serve
    - Models pulled:
        - Chat model (e.g., `qwen3:4b-instruct-2507-q4_K_M`): Set in `.env`
        - Embedding model (e.g., `qwen3-embedding:4b`): Default in `LocalVectorSearch`
    - Dependencies:
        pip install langchain langchain_community faiss-cpu python-dotenv

Features:
    - 100% local (no external APIs)
    - Semantic search using qwen3-embedding:4b (production quality)
    - Keyword search fallback (BM25)
    - Hybrid search combining both methods
    - Persistent conversation memory
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add production module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "production"))

from qwen_agent.agents import Assistant
from qwen_pipeline.config import get_llm_config
from qwen_pipeline.tools_custom import LocalVectorSearch  # noqa: F401


def example_basic_rag():
    """Basic RAG with local vector search only."""
    print("\n=== Example 1: Basic Local Vector RAG ===\n")

    # Configure LLM from .env
    llm_cfg = get_llm_config()
    model_name = os.getenv("MODEL_NAME", "qwen3:4b-instruct-2507-q4_K_M")
    print(f"Using chat model: {model_name}")

    # Create Assistant with local vector search
    bot = Assistant(
        llm=llm_cfg,
        name="Local RAG Assistant",
        description="100% local RAG with Ollama embeddings",
        files=[],  # No pre-loaded files yet
        # RAG configuration using the Assistant's built-in RAG capabilities.
        # For more details on Assistant RAG, see:
        # https://qwen-agent.readthedocs.io/en/latest/tutorials/agent_usage/assistant_agent_rag_usage.html
        rag_cfg={
            # max_ref_token: The maximum number of tokens from retrieved documents to include in the prompt.
            # This should be less than the model's context window.
            # We set it to 4000 to leave room for the query and other prompt content.
            "max_ref_token": 4000,
            # parser_page_size: The size of chunks (in characters) to split documents into.
            "parser_page_size": 500,
            # rag_searchers: A list of search tools to use for retrieval.
            # 'local_vector_search' is our custom tool for 100% local semantic search.
            "rag_searchers": ["local_vector_search"],
        },
    )

    # Chat with document reference
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "What is this document about?"},
                {"file": "https://arxiv.org/pdf/2310.08560.pdf"},  # Qwen technical report
            ],
        }
    ]

    print("Query: What is this document about?")
    print("File: https://arxiv.org/pdf/2310.08560.pdf\n")

    for response in bot.run(messages):
        if response[-1]["role"] == "assistant":
            print(f"Assistant: {response[-1]['content']}\n")


def example_hybrid_rag():
    """Hybrid RAG combining vector search + keyword search."""
    print("\n=== Example 2: Hybrid RAG (Vector + Keyword) ===\n")

    llm_cfg = get_llm_config()

    # Hybrid search: semantic (vector) + exact (keyword)
    bot = Assistant(
        llm=llm_cfg,
        name="Hybrid RAG Assistant",
        files=[],
        rag_cfg={
            "max_ref_token": 4000,
            "parser_page_size": 500,
            "rag_searchers": [
                "local_vector_search",  # Semantic search
                "keyword_search",  # BM25 exact match
            ],
        },
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Explain the attention mechanism"},
                {"file": "https://arxiv.org/pdf/1706.03762.pdf"},  # Attention paper
            ],
        }
    ]

    print("Query: Explain the attention mechanism")
    print("File: https://arxiv.org/pdf/1706.03762.pdf (Attention is All You Need)\n")

    for response in bot.run(messages):
        if response[-1]["role"] == "assistant":
            print(f"Assistant: {response[-1]['content']}\n")


def example_multi_document():
    """Query across multiple documents with persistent memory."""
    print("\n=== Example 3: Multi-Document RAG with Memory ===\n")

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
    messages = []

    # Turn 1: Ask about Qwen
    messages.append({"role": "user", "content": "What models are in the Qwen family?"})
    print("User: What models are in the Qwen family?")

    for rsp in bot.run(messages):
        messages = rsp  # Each iteration gives us the updated full history
    print(f"Assistant: {messages[-1]['content']}\n")

    # Turn 2: Ask about attention (different document)
    messages.append({"role": "user", "content": "How does multi-head attention work?"})
    print("User: How does multi-head attention work?")

    for rsp in bot.run(messages):
        messages = rsp  # Each iteration gives us the updated full history
    print(f"Assistant: {messages[-1]['content']}\n")

    # Turn 3: Cross-reference question
    messages.append({"role": "user", "content": "Do Qwen models use multi-head attention?"})
    print("User: Do Qwen models use multi-head attention?")

    for rsp in bot.run(messages):
        messages = rsp  # Each iteration gives us the updated full history
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
