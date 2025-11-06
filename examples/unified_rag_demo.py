#!/usr/bin/env python
"""Unified Multi-Source RAG Demonstration.

Complete example showing how to:
1. Ingest content from web searches
2. Ingest content from local documentation files
3. Store both in a unified FAISS vector store
4. Query with source attribution (web vs local)
5. Use in a RAG agent for question answering

This demonstrates the core benefit: single, maintainable RAG pipeline
for all your knowledge sources with transparent source tracking.

Usage:
    python examples/unified_rag_demo.py

Requirements:
    - Ollama running with qwen3-embedding:4b and qwen3:4b models
    - SERPER_API_KEY set for web search (optional, can skip web ingestion)
    - Sample local files in ./docs directory (or any other path)

Copyright: Based on Qwen-Agent patterns from QwenLM/Qwen-Agent
License: Apache License 2.0
"""

import sys
from pathlib import Path

import structlog

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

from production.qwen_pipeline.config import get_llm_config  # noqa: E402
from production.qwen_pipeline.tools_custom import LocalVectorSearch  # noqa: E402
from production.qwen_pipeline.unified_ingestion import ingest_unified  # noqa: E402
from qwen_agent.agents import ReActChat  # noqa: E402


def print_banner(title: str) -> None:
    """Print a section banner."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_ingestion() -> str:
    """Demonstrate unified ingestion from web and local sources."""
    print_banner("Step 1: Unified Ingestion (Web + Local Files)")

    store_name = "unified_demo_kb"

    # Define sources
    web_topics = ["Python RAG tutorial", "FAISS vector search"]
    local_paths = [
        str(PROJECT_ROOT / "docs" / "patterns"),
        str(PROJECT_ROOT / "README.md"),
    ]

    print(f"📚 Creating unified knowledge base: '{store_name}'")
    print(f"\n🌐 Web topics to ingest:")
    for topic in web_topics:
        print(f"   - {topic}")

    print(f"\n📁 Local paths to ingest:")
    for path in local_paths:
        print(f"   - {path}")

    print("\n⏳ Starting ingestion (this may take a minute)...\n")

    try:
        result = ingest_unified(
            store_name=store_name,
            web_topics=web_topics,
            local_paths=local_paths,
            max_urls_per_topic=3,  # Limit to 3 URLs per topic for faster demo
            recursive=True,
        )

        if result["status"] == "completed":
            print("✅ Ingestion completed successfully!\n")
            print(f"📊 Summary:")
            print(f"   Total chunks: {result['total_chunks']}")
            print(
                f"   Web chunks: {result['web_chunks']} (from {result['web_urls_processed']} URLs)"
            )
            print(
                f"   Local chunks: {result['local_chunks']} (from {result['local_files_processed']} files)"
            )
            print(f"   Storage path: {result['storage_path']}")
        else:
            print(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception:
        logger.exception("ingestion_failed")
        print("\n❌ Ingestion failed - see logs above")
        print("💡 Tip: Ensure Ollama is running and SERPER_API_KEY is set")
        sys.exit(1)

    return store_name


def demo_query_with_attribution(store_name: str) -> None:
    """Demonstrate querying with source attribution."""
    print_banner("Step 2: Query with Source Attribution")

    vector_tool = LocalVectorSearch()

    # Test query
    query = "What is RAG and how does it work?"
    print(f"🔍 Query: '{query}'\n")

    try:
        result = vector_tool.query_with_source_attribution(store_name=store_name, query=query, k=5)

        print(f"📊 Results:")
        print(f"   Total: {result['total_results']}")
        print(f"   From web: {len(result['web_results'])}")
        print(f"   From local files: {len(result['local_results'])}\n")

        # Display top 3 results with full metadata
        for i, res in enumerate(result["all_results"][:3], 1):
            metadata = res["metadata"]
            source_type = metadata.get("source_type", "unknown")

            print(f"[{i}] Similarity score: {res['score']:.4f}")
            print(f"    Source type: {source_type}")
            print(f"    Origin: {metadata.get('source', 'N/A')}")

            if source_type == "web":
                print(f"    Topic: {metadata.get('topic', 'N/A')}")
            elif source_type == "local_file":
                print(f"    File type: {metadata.get('file_type', 'N/A')}")

            # Content preview
            content = res["content"][:250]
            if len(res["content"]) > 250:
                content += "..."
            print(f"    Content: {content}\n")

    except Exception:
        logger.exception("query_failed")
        print("❌ Query failed - see logs above")


def demo_store_statistics(store_name: str) -> None:
    """Demonstrate viewing store statistics."""
    print_banner("Step 3: Store Statistics")

    vector_tool = LocalVectorSearch()

    try:
        stats = vector_tool.get_store_statistics(store_name)

        if "error" in stats:
            print(f"❌ {stats['error']}")
            return

        print(f"📊 Statistics for '{store_name}':\n")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Web documents: {stats['web_documents']}")
        print(f"Local documents: {stats['local_documents']}")

        if stats["file_types"]:
            print("\nLocal file types:")
            for file_type, count in stats["file_types"].items():
                print(f"   - {file_type}: {count}")

        if stats["topics"]:
            print(f"\nWeb topics: {', '.join(stats['topics'])}")

        print(f"\nStore location: {stats['store_path']}")

    except Exception:
        logger.exception("stats_failed")
        print("❌ Failed to retrieve statistics")


def demo_rag_agent(store_name: str) -> None:
    """Demonstrate using unified store in a RAG agent."""
    print_banner("Step 4: RAG Agent with Unified Knowledge Base")

    vector_tool = LocalVectorSearch()
    llm_cfg = get_llm_config()

    # Create agent with vector search tool
    agent = ReActChat(
        llm=llm_cfg,
        function_list=[vector_tool],
        system_message=(
            "You are a helpful AI assistant with access to a knowledge base. "
            "Use the local_vector_search tool to find relevant information "
            "and cite your sources (web or local file)."
        ),
        name="RAGAssistant",
    )

    # Test question
    question = (
        "Explain RAG patterns and cite whether the information comes from web or local sources"
    )
    print(f"💬 Question: '{question}'\n")
    print("🤖 Agent thinking...\n")

    messages = [{"role": "user", "content": question}]

    try:
        responses = list(agent.run(messages=messages))

        # Display agent's response
        for response in responses:
            if isinstance(response, dict):
                role = response.get("role", "unknown")
                content = response.get("content", "")

                if role == "assistant":
                    print("🤖 Assistant:")
                    print(f"   {content}\n")
                elif role == "function":
                    print(f"🔧 Tool used: {response.get('name', 'unknown')}")

    except Exception:
        logger.exception("agent_failed")
        print("❌ Agent failed - see logs above")
        print("💡 Tip: Ensure Ollama is running with qwen3:4b model")


def main() -> None:
    """Run complete unified RAG demonstration."""
    print("\n" + "🌟" * 40)
    print("  Unified Multi-Source RAG Demonstration")
    print("  Powered by Qwen-Agent + FAISS + Ollama")
    print("🌟" * 40)

    print("\nThis demo will:")
    print("  1. Ingest content from web searches (via SERPER API)")
    print("  2. Ingest content from local documentation files")
    print("  3. Store everything in a unified FAISS vector store")
    print("  4. Query with source attribution (web vs local)")
    print("  5. Use in a RAG agent for question answering")
    print("\n💡 Benefits of unified approach:")
    print("  ✓ Single source of truth for all knowledge")
    print("  ✓ Consistent retrieval interface")
    print("  ✓ Source transparency via metadata")
    print("  ✓ No code duplication")

    input("\nPress Enter to start demonstration...")

    # Step 1: Ingestion
    store_name = demo_ingestion()

    # Step 2: Query with attribution
    demo_query_with_attribution(store_name)

    # Step 3: View statistics
    demo_store_statistics(store_name)

    # Step 4: RAG agent
    demo_rag_agent(store_name)

    print_banner("Demonstration Complete!")

    print("✅ You've seen how to:")
    print("   • Ingest from multiple sources (web + local)")
    print("   • Query with source attribution")
    print("   • View store statistics")
    print("   • Use in a RAG agent\n")

    print("📚 Next steps:")
    print("   • Try the interactive CLI: python -m qwen_pipeline.ingest_cli")
    print("   • Read the docs: docs/patterns/RAG_PATTERNS.md (Pattern 6)")
    print("   • Explore the code: production/qwen_pipeline/unified_ingestion.py\n")

    print(f"🗂️  Your vector store is saved at:")
    vector_tool = LocalVectorSearch()
    print(f"   {vector_tool.vector_store_path}/{store_name}/")
    print("   You can query it anytime using LocalVectorSearch.query_store()\n")


if __name__ == "__main__":
    main()
