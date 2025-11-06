#!/usr/bin/env python
"""
Web Search to Vector Storage Ingestion Example.

Demonstrates automated knowledge ingestion from web search results into
a persistent FAISS vector store, with retrieval via RAG queries.

Features:
- Search a topic using web search
- Extract content from multiple URLs with retry logic
- Smart text chunking with paragraph boundaries and overlap
- Store embeddings in FAISS for fast similarity search
- Query stored vectors for RAG applications
- Topic-based storage organization

Usage:
    python web_to_vector_ingestion.py

Then choose from the interactive menu:
    1. Ingest new topic from web
    2. Query existing vector store
    3. List available stores
    4. Exit

Requirements:
    - Ollama running (models: qwen3-embedding:4b, qwen3:8b or similar)
    - SERPER_API_KEY set for web search
    - VECTOR_STORE_PATH configured in .env

Copyright: Based on Qwen-Agent patterns from QwenLM/Qwen-Agent
License: Apache License 2.0
"""

import os
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

from production.qwen_pipeline.web_rag_ingestion import (  # noqa: E402
    create_ingestion_agent,
    ingest_from_web,
    list_ingested_stores,
    query_ingested_store,
)


def display_welcome() -> None:
    """Display welcome message."""
    print("\n" + "=" * 70)
    print("  Web Search → Vector Storage Ingestion Demo")
    print("  Powered by Qwen-Agent + FAISS + Ollama Embeddings")
    print("=" * 70)
    print("\nThis tool demonstrates automated knowledge ingestion:")
    print("  1. Search topics using web search")
    print("  2. Extract and chunk content from URLs")
    print("  3. Embed chunks using local Ollama embeddings")
    print("  4. Store in FAISS for fast retrieval")
    print("  5. Query with RAG for knowledge synthesis\n")


def display_menu() -> None:
    """Display main menu."""
    print("\n" + "-" * 70)
    print("Main Menu:")
    print("  1. Ingest new topic from web")
    print("  2. Query existing vector store")
    print("  3. List available stores")
    print("  4. Exit")
    print("-" * 70)


def ingest_topic_flow() -> None:
    """Interactive flow for ingesting a new topic."""
    print("\n--- Ingest New Topic ---")
    topic = input("Enter topic to search (e.g., 'machine learning basics'): ").strip()

    if not topic:
        print("Topic cannot be empty.")
        return

    store_name_input = input("Enter store name (default: topic name, auto-sanitized): ").strip()

    store_name: str | None = store_name_input if store_name_input else None

    max_urls_str = input("Max URLs to process (default: 5): ").strip()
    max_urls = int(max_urls_str) if max_urls_str.isdigit() else 5

    print(f"\n📡 Ingesting topic: '{topic}'")
    print(f"   Max URLs: {max_urls}")
    print("   Chunking: 500-token max, 50-token overlap")
    print("   Retries: 2 per URL\n")

    try:
        result = ingest_from_web(topic, store_name=store_name, max_urls=max_urls)

        print("\n✅ Ingestion Complete!")
        print(f"   Store: {result['store_name']}")
        print(f"   URLs processed: {result['urls_processed']}")
        print(f"   URLs failed: {result['urls_failed']}")
        print(f"   Total chunks stored: {result['chunks_stored']}")
        print(f"   Storage path: {result['storage_path']}\n")

        if result["urls_failed"] > 0:
            print(f"⚠️  {result['urls_failed']} URLs failed to extract.")
            print("   (This is normal for some websites)")

    except Exception as e:
        logger.exception("ingestion_failed", topic=topic, error=str(e))
        print(f"\n❌ Ingestion failed: {e}")


def query_store_flow() -> None:
    """Interactive flow for querying a vector store."""
    print("\n--- Query Vector Store ---")

    # List available stores
    stores = list_ingested_stores()
    if not stores:
        print("No vector stores available. Ingest a topic first.")
        return

    print(f"\nAvailable stores: {', '.join(stores)}")
    store_name = input("Enter store name to query: ").strip()

    if store_name not in stores:
        print(f"Store '{store_name}' not found.")
        return

    query = input("Enter search query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    k_str = input("Number of results (default: 5): ").strip()
    k = int(k_str) if k_str.isdigit() else 5

    print(f"\n🔍 Querying '{store_name}' with: '{query}'\n")

    try:
        results = query_ingested_store(store_name, query, k=k)

        if not results:
            print("No results found.")
            return

        print(f"Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            print(f"--- Result {i} (score: {result['score']:.3f}) ---")
            print(f"Source: {result['metadata'].get('source', 'unknown')}")
            print(f"Content: {result['content'][:200]}...")
            print()

    except Exception as e:
        logger.exception("query_failed", store_name=store_name, query=query, error=str(e))
        print(f"\n❌ Query failed: {e}")


def list_stores_flow() -> None:
    """Display available vector stores."""
    print("\n--- Available Vector Stores ---")

    stores = list_ingested_stores()

    if not stores:
        print("No vector stores found.")
        return

    print(f"\nFound {len(stores)} store(s):\n")
    for i, store in enumerate(stores, 1):
        print(f"  {i}. {store}")

    print()


def main() -> None:
    """Main application loop."""
    display_welcome()

    # Check prerequisites
    if not os.getenv("SERPER_API_KEY"):
        print("⚠️  WARNING: SERPER_API_KEY not set in .env")
        print("   Web search will not work. Set SERPER_API_KEY to continue.\n")

    print("Initializing ingestion agent...")
    try:
        create_ingestion_agent()
        print("✅ Agent initialized successfully\n")
    except Exception as e:
        logger.exception("agent_initialization_failed", error=str(e))
        print(f"❌ Failed to initialize agent: {e}")
        return

    # Main loop
    while True:
        display_menu()
        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            ingest_topic_flow()
        elif choice == "2":
            query_store_flow()
        elif choice == "3":
            list_stores_flow()
        elif choice == "4":
            print("\n👋 Goodbye!\n")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user.\n")
        sys.exit(0)
    except Exception as e:
        logger.exception("application_error", error=str(e))
        print(f"\n❌ Application error: {e}")
        sys.exit(1)
