#!/usr/bin/env python
"""Unified RAG Ingestion CLI.

Interactive command-line interface for ingesting content from multiple
sources (web and local files) into a unified vector store.

Features:
- Ingest from web topics (search → extract → embed → store)
- Ingest from local files (load → chunk → embed → store)
- Mix both sources in single vector store
- Query with source attribution
- View store statistics and metadata
- List all available stores

Usage:
    python -m qwen_pipeline.ingest_cli

Environment:
    - SERPER_API_KEY: Required for web search
    - VECTOR_STORE_PATH: Storage location (default: ./workspace/vector_stores)
    - EMBEDDING_MODEL: Ollama model (default: qwen3-embedding:4b)

Copyright: Based on Qwen-Agent patterns from QwenLM/Qwen-Agent
License: Apache License 2.0
"""

import json
import sys

import structlog

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

from qwen_pipeline.tools_custom import LocalVectorSearch  # noqa: E402
from qwen_pipeline.unified_ingestion import ingest_unified  # noqa: E402

# Constants
CONTENT_PREVIEW_LENGTH = 200


def display_banner() -> None:
    """Display welcome banner."""
    print("\n" + "=" * 75)
    print("  🚀 Unified RAG Ingestion System")
    print("  Powered by Qwen-Agent + FAISS + Ollama")
    print("=" * 75)
    print("\n📚 Ingest from multiple sources into unified knowledge base:")
    print("   • Web content (via search and extraction)")
    print("   • Local files (MD, TXT, PDF, DOCX, RST)")
    print("   • Mixed sources with metadata tracking")
    print("   • Source attribution in queries\n")


def display_menu() -> None:
    """Display main menu options."""
    print("\n" + "-" * 75)
    print("Main Menu:")
    print("  1. Ingest from web topics")
    print("  2. Ingest from local files")
    print("  3. Ingest from both (unified)")
    print("  4. Query a vector store")
    print("  5. View store statistics")
    print("  6. List all vector stores")
    print("  7. Exit")
    print("-" * 75)


def get_input(prompt: str, default: str = "") -> str:
    """Get user input with optional default."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def ingest_web_flow() -> None:
    """Interactive flow for web ingestion."""
    print("\n" + "=" * 75)
    print("  📡 Ingest from Web Topics")
    print("=" * 75)

    store_name = get_input("Enter vector store name (e.g., 'ml_knowledge')")
    if not store_name:
        print("❌ Store name required")
        return

    print("\nEnter topics (comma-separated):")
    print("  Example: machine learning, python RAG, FAISS tutorial")
    topics_input = get_input("Topics")

    if not topics_input:
        print("❌ At least one topic required")
        return

    topics = [t.strip() for t in topics_input.split(",")]
    max_urls = int(get_input("Max URLs per topic", "5"))

    print(f"\n🔍 Ingesting {len(topics)} topics into '{store_name}'...")
    print(f"   Topics: {', '.join(topics)}")
    print(f"   Max URLs per topic: {max_urls}\n")

    try:
        result = ingest_unified(
            store_name=store_name,
            web_topics=topics,
            local_paths=None,
            max_urls_per_topic=max_urls,
        )

        if result["status"] == "completed":
            print("\n✅ Ingestion completed successfully!")
            print(f"   Total chunks: {result['total_chunks']}")
            print(f"   Web chunks: {result['web_chunks']}")
            print(f"   URLs processed: {result['web_urls_processed']}")
            print(f"   Storage path: {result['storage_path']}")
        else:
            print(f"\n❌ Ingestion failed: {result.get('error', 'Unknown error')}")

    except Exception:
        logger.exception("ingestion_failed")
        print("\n❌ Ingestion failed - see logs for details")


def ingest_local_flow() -> None:
    """Interactive flow for local file ingestion."""
    print("\n" + "=" * 75)
    print("  📁 Ingest from Local Files")
    print("=" * 75)

    store_name = get_input("Enter vector store name (e.g., 'local_docs')")
    if not store_name:
        print("❌ Store name required")
        return

    print("\nEnter file/directory paths (comma-separated):")
    print("  Example: ./docs, ./README.md, ./research_papers")
    print("  Supported: .md, .txt, .pdf, .docx, .rst")
    paths_input = get_input("Paths")

    if not paths_input:
        print("❌ At least one path required")
        return

    paths = [p.strip() for p in paths_input.split(",")]
    recursive = get_input("Scan subdirectories recursively?", "yes").lower() in {
        "yes",
        "y",
    }

    print(f"\n📂 Ingesting {len(paths)} paths into '{store_name}'...")
    print(f"   Paths: {', '.join(paths)}")
    print(f"   Recursive: {recursive}\n")

    try:
        result = ingest_unified(
            store_name=store_name,
            web_topics=None,
            local_paths=paths,
            recursive=recursive,
        )

        if result["status"] == "completed":
            print("\n✅ Ingestion completed successfully!")
            print(f"   Total chunks: {result['total_chunks']}")
            print(f"   Local chunks: {result['local_chunks']}")
            print(f"   Files processed: {result['local_files_processed']}")
            print(f"   Storage path: {result['storage_path']}")
        else:
            print(f"\n❌ Ingestion failed: {result.get('error', 'Unknown error')}")

    except Exception:
        logger.exception("ingestion_failed")
        print("\n❌ Ingestion failed - see logs for details")


def ingest_unified_flow() -> None:
    """Interactive flow for unified ingestion (web + local)."""
    print("\n" + "=" * 75)
    print("  🌐 Unified Ingestion (Web + Local)")
    print("=" * 75)

    store_name = get_input("Enter vector store name (e.g., 'unified_kb')")
    if not store_name:
        print("❌ Store name required")
        return

    # Get web topics
    print("\nWeb topics (comma-separated, or press Enter to skip):")
    topics_input = get_input("Topics")
    topics = [t.strip() for t in topics_input.split(",")] if topics_input else None

    # Get local paths
    print("\nLocal paths (comma-separated, or press Enter to skip):")
    paths_input = get_input("Paths")
    paths = [p.strip() for p in paths_input.split(",")] if paths_input else None

    if not topics and not paths:
        print("❌ Must provide either web topics or local paths")
        return

    max_urls = int(get_input("Max URLs per topic", "5")) if topics else 5
    recursive = (
        get_input("Scan subdirectories recursively?", "yes").lower() in {"yes", "y"}
        if paths
        else True
    )

    print(f"\n🚀 Starting unified ingestion into '{store_name}'...")
    if topics:
        print(f"   Web topics: {', '.join(topics)}")
    if paths:
        print(f"   Local paths: {', '.join(paths)}")
    print()

    try:
        result = ingest_unified(
            store_name=store_name,
            web_topics=topics,
            local_paths=paths,
            max_urls_per_topic=max_urls,
            recursive=recursive,
        )

        if result["status"] == "completed":
            print("\n✅ Unified ingestion completed successfully!")
            print(f"   Total chunks: {result['total_chunks']}")
            web_info = f"from {result['web_topics']} topics"
            print(f"   Web chunks: {result['web_chunks']} ({web_info})")
            local_info = f"from {result['local_files_processed']} files"
            print(f"   Local chunks: {result['local_chunks']} ({local_info})")
            print(f"   Storage path: {result['storage_path']}")
        else:
            print(f"\n❌ Ingestion failed: {result.get('error', 'Unknown error')}")

    except Exception:
        logger.exception("unified_ingestion_failed")
        print("\n❌ Ingestion failed - see logs for details")


def query_store_flow() -> None:
    """Interactive flow for querying a vector store."""
    print("\n" + "=" * 75)
    print("  🔍 Query Vector Store")
    print("=" * 75)

    # List available stores first
    vector_tool = LocalVectorSearch()
    stores_json = vector_tool.list_stores()
    stores = json.loads(stores_json)

    if not stores:
        print("❌ No vector stores found. Ingest some content first!")
        return

    print(f"\nAvailable stores: {', '.join(stores)}")
    store_name = get_input("Enter store name to query")

    if store_name not in stores:
        print(f"❌ Store '{store_name}' not found")
        return

    query = get_input("Enter search query")
    if not query:
        print("❌ Query required")
        return

    k = int(get_input("Number of results", "5"))

    print(f"\n🔍 Searching '{store_name}' for: {query}\n")

    try:
        result = vector_tool.query_with_source_attribution(store_name, query, k=k)

        print(f"✅ Found {result['total_results']} results:")
        print(f"   - {len(result['web_results'])} from web sources")
        print(f"   - {len(result['local_results'])} from local files\n")

        # Display results
        for i, res in enumerate(result["all_results"], 1):
            metadata = res["metadata"]
            source_type = metadata.get("source_type", "unknown")
            source = metadata.get("source", "unknown")

            print(f"\n[{i}] Score: {res['score']:.4f} | Type: {source_type}")
            print(f"    Source: {source}")

            if source_type == "web":
                print(f"    Topic: {metadata.get('topic', 'N/A')}")
            elif source_type == "local_file":
                print(f"    File type: {metadata.get('file_type', 'N/A')}")

            # Show content preview
            content_preview = res["content"][:CONTENT_PREVIEW_LENGTH]
            if len(res["content"]) > CONTENT_PREVIEW_LENGTH:
                content_preview += "..."
            print(f"    Content: {content_preview}")

    except Exception:
        logger.exception("query_failed")
        print("\n❌ Query failed - see logs for details")


def view_statistics_flow() -> None:
    """Interactive flow for viewing store statistics."""
    print("\n" + "=" * 75)
    print("  📊 View Store Statistics")
    print("=" * 75)

    # List available stores
    vector_tool = LocalVectorSearch()
    stores_json = vector_tool.list_stores()
    stores = json.loads(stores_json)

    if not stores:
        print("❌ No vector stores found")
        return

    print(f"\nAvailable stores: {', '.join(stores)}")
    store_name = get_input("Enter store name")

    if store_name not in stores:
        print(f"❌ Store '{store_name}' not found")
        return

    print(f"\n📊 Statistics for '{store_name}':\n")

    try:
        stats = vector_tool.get_store_statistics(store_name)

        if "error" in stats:
            print(f"❌ {stats['error']}")
            return

        print(f"Total documents: {stats['total_documents']}")
        print(f"Web documents: {stats['web_documents']}")
        print(f"Local documents: {stats['local_documents']}")

        if stats["file_types"]:
            print("\nFile types:")
            for file_type, count in stats["file_types"].items():
                print(f"  - {file_type}: {count}")

        if stats["topics"]:
            print(f"\nWeb topics: {', '.join(stats['topics'])}")

        print(f"\nStore path: {stats['store_path']}")

    except Exception:
        logger.exception("stats_retrieval_failed")
        print("\n❌ Failed to retrieve statistics")


def list_stores_flow() -> None:
    """List all available vector stores."""
    print("\n" + "=" * 75)
    print("  📚 Available Vector Stores")
    print("=" * 75)

    vector_tool = LocalVectorSearch()
    stores_json = vector_tool.list_stores()
    stores = json.loads(stores_json)

    if not stores:
        print("\n❌ No vector stores found")
        print("   Create one using option 1, 2, or 3\n")
        return

    print(f"\nFound {len(stores)} vector store(s):\n")
    for i, store in enumerate(stores, 1):
        print(f"  {i}. {store}")

        # Get quick stats
        try:
            stats = vector_tool.get_store_statistics(store)
            print(
                f"      {stats['total_documents']} docs "
                f"({stats['web_documents']} web, {stats['local_documents']} local)"
            )
        except Exception:
            print("      (statistics unavailable)")

    print()


def main() -> None:
    """Main CLI loop."""
    display_banner()

    while True:
        display_menu()
        choice = get_input("Select option (1-7)")

        if choice == "1":
            ingest_web_flow()
        elif choice == "2":
            ingest_local_flow()
        elif choice == "3":
            ingest_unified_flow()
        elif choice == "4":
            query_store_flow()
        elif choice == "5":
            view_statistics_flow()
        elif choice == "6":
            list_stores_flow()
        elif choice == "7":
            print("\n👋 Goodbye!\n")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please select 1-7.")


if __name__ == "__main__":
    main()
