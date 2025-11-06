# Unified Multi-Source RAG System

**Complete solution for ingesting, storing, and retrieving content from web searches AND local documentation files using a single, maintainable pipeline.**

## 🎯 Overview

This system extends your existing web-based RAG pipeline to also support local documentation files, using the same storage, retrieval, and query logic. Both web and local sources are handled together in a unified vector store, with metadata to track the origin of each chunk.

### Key Features

✅ **Unified storage** - Single FAISS vector store for all content
✅ **Source attribution** - Metadata tracks whether chunks came from web or local files
✅ **No code duplication** - Reuses existing chunking, embedding, and storage logic
✅ **Multiple file formats** - Supports .md, .txt, .pdf, .docx, .rst
✅ **Interactive CLI** - Easy-to-use interface for ingestion management
✅ **Source-aware queries** - Separate web and local results for transparency
✅ **Extensible design** - Easy to add new source types (APIs, databases, etc.)

## 📁 Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   Web Topics    │         │  Local Files     │
│  (search URLs)  │         │  (.md/.txt/.pdf) │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         │    Unified Ingestion      │
         └───────────┬───────────────┘
                     │
         ┌───────────▼──────────────┐
         │  Chunk + Embed + Store   │
         │  (LocalVectorSearch)     │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │   FAISS Vector Store     │
         │  (with source metadata)  │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │   Query with Attribution │
         │  (web/local separation)  │
         └──────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core dependencies (already in production/pyproject.toml)
pip install -e production/

# Optional: For PDF support
pip install pypdf

# Optional: For Word document support
pip install python-docx
```

### 2. Set Up Environment

```bash
# .env file
VECTOR_STORE_PATH=./workspace/vector_stores
EMBEDDING_MODEL=qwen3-embedding:4b
EMBEDDING_BASE_URL=http://localhost:11434
SERPER_API_KEY=your_serper_api_key_here  # For web search
```

### 3. Start Ollama

```bash
ollama serve
ollama pull qwen3-embedding:4b
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

### 4. Run Interactive CLI

```bash
python -m qwen_pipeline.ingest_cli
```

Or run the complete demo:

```bash
python examples/unified_rag_demo.py
```

## 📚 Usage Examples

### Programmatic Usage

```python
from qwen_pipeline.unified_ingestion import ingest_unified
from qwen_pipeline.tools_custom import LocalVectorSearch

# Ingest from multiple sources into single store
result = ingest_unified(
    store_name="my_knowledge_base",
    web_topics=["machine learning basics", "Python best practices"],
    local_paths=["./docs", "./research_papers", "./README.md"],
    max_urls_per_topic=5,
    recursive=True,
)

print(f"Ingested {result['total_chunks']} chunks:")
print(f"  - {result['web_chunks']} from web")
print(f"  - {result['local_chunks']} from local files")

# Query with source attribution
vector_tool = LocalVectorSearch()
query_result = vector_tool.query_with_source_attribution(
    store_name="my_knowledge_base",
    query="What are Python decorators?",
    k=5
)

print(f"Found {query_result['total_results']} results:")
print(f"  - {len(query_result['web_results'])} from web")
print(f"  - {len(query_result['local_results'])} from local files")

# Examine results with metadata
for result in query_result['all_results']:
    metadata = result['metadata']
    print(f"\nSource: {metadata['source_type']} - {metadata['source']}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Score: {result['score']:.4f}")
```

### Interactive CLI

The CLI provides a user-friendly interface:

```bash
python -m qwen_pipeline.ingest_cli
```

Menu options:
1. **Ingest from web topics** - Web search → extract → store
2. **Ingest from local files** - Load files → chunk → store
3. **Ingest from both (unified)** - ⭐ Recommended for mixed sources
4. **Query a vector store** - Search with source attribution
5. **View store statistics** - Analyze content distribution
6. **List all vector stores** - See what's available

## 🏗️ Components

### Core Modules

| File | Purpose |
|------|---------|
| `production/qwen_pipeline/unified_ingestion.py` | Main ingestion logic for web + local files |
| `production/qwen_pipeline/tools_custom.py` | Enhanced LocalVectorSearch with attribution |
| `production/qwen_pipeline/ingest_cli.py` | Interactive command-line interface |
| `production/qwen_pipeline/web_rag_ingestion.py` | Existing web ingestion (reused) |

### Examples

| File | Purpose |
|------|---------|
| `examples/unified_rag_demo.py` | Complete demonstration of all features |
| `examples/web_to_vector_ingestion.py` | Web-only ingestion (still works) |
| `examples/local_vector_rag_example.py` | Local-only RAG example |

### Documentation

| File | Purpose |
|------|---------|
| `docs/patterns/RAG_PATTERNS.md` | Pattern 6: Unified Multi-Source RAG |
| `UNIFIED_RAG_SYSTEM.md` | This file |

## 📊 Metadata Schema

### Web Sources

```python
{
    "source": "https://example.com/article",
    "source_type": "web",
    "topic": "machine learning",
    "chunk_index": 0,
    "total_chunks": 10,
    "ingested_at": "2025-11-06T10:30:00Z"
}
```

### Local Files

```python
{
    "source": "/absolute/path/to/file.md",
    "source_type": "local_file",
    "file_type": "markdown",  # or "pdf", "text", "docx", "restructuredtext"
    "chunk_index": 0,
    "total_chunks": 5,
    "ingested_at": "2025-11-06T10:30:00Z"
}
```

## 🔧 Supported File Formats

| Extension | Type | Requires |
|-----------|------|----------|
| `.md` | Markdown | Built-in |
| `.txt` | Plain text | Built-in |
| `.rst` | reStructuredText | Built-in |
| `.pdf` | PDF | `pip install pypdf` |
| `.docx` | Word | `pip install python-docx` |

## 🎓 API Reference

### `ingest_unified()`

Main entry point for unified ingestion.

```python
def ingest_unified(
    store_name: str,
    web_topics: list[str] | None = None,
    local_paths: list[str] | list[Path] | None = None,
    max_urls_per_topic: int = 5,
    recursive: bool = True,
) -> dict[str, Any]:
    """
    Ingest from web and/or local sources into single vector store.

    Args:
        store_name: Name for the vector store
        web_topics: Optional list of web search topics
        local_paths: Optional list of local file/directory paths
        max_urls_per_topic: Max URLs per web topic (default: 5)
        recursive: Recursively scan local directories (default: True)

    Returns:
        Ingestion report with status, counts, and storage path
    """
```

### `LocalVectorSearch.query_with_source_attribution()`

Query with source type separation.

```python
def query_with_source_attribution(
    self, store_name: str, query: str, k: int = 5
) -> dict[str, Any]:
    """
    Query and group results by source type.

    Returns:
        {
            "query": "search query",
            "total_results": 5,
            "web_results": [...],
            "local_results": [...],
            "all_results": [...]
        }
    """
```

### `LocalVectorSearch.get_store_statistics()`

Analyze vector store content.

```python
def get_store_statistics(self, store_name: str) -> dict[str, Any]:
    """
    Get statistics about store content.

    Returns:
        {
            "total_documents": 100,
            "web_documents": 60,
            "local_documents": 40,
            "file_types": {"markdown": 25, "pdf": 15},
            "topics": ["ML", "Python"],
            "store_path": "..."
        }
    """
```

## 🚦 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Local file loading | ~0.01s | Text formats (MD, TXT, RST) |
| PDF file loading | ~0.5s | Requires pypdf |
| Web extraction | ~2-3s | Per URL (with retries) |
| Chunking | ~0.1s | Per 5000 tokens |
| Embedding | ~0.5s | Per chunk (qwen3-embedding:4b) |
| Query | <0.1s | Top 5 results |

## 🛣️ Future Enhancements

The system is designed for easy extension:

- **Reranking** - Add cross-encoder reranking for improved relevance
- **User feedback** - Track query effectiveness and result quality
- **Incremental updates** - Detect changed files and re-ingest only deltas
- **Hybrid search** - Combine vector similarity with keyword search
- **Source prioritization** - Weight results by source type or recency
- **More file types** - Add support for HTML, XML, JSON, etc.
- **Cloud sources** - Ingest from S3, Google Drive, Notion, etc.

## 📖 Related Documentation

- **[RAG Patterns](docs/patterns/RAG_PATTERNS.md)** - Pattern 6: Unified Multi-Source RAG
- **[Qwen Standards](QWEN_STANDARDS.md)** - Official Qwen-Agent patterns
- **[Production README](docs/Production-README.md)** - Production module overview

## 🤝 Integration with Agents

Use the unified store with Qwen-Agent:

```python
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config
from qwen_pipeline.tools_custom import LocalVectorSearch

vector_tool = LocalVectorSearch()
agent = ReActChat(
    llm=get_llm_config(),
    function_list=[vector_tool],
    system_message=(
        "You are a helpful assistant with access to a knowledge base. "
        "Use local_vector_search to find information and cite sources."
    )
)

messages = [{"role": "user", "content": "Explain RAG and cite your sources"}]
for response in agent.run(messages=messages):
    print(response)
```

## ✅ Benefits Summary

| Benefit | Description |
|---------|-------------|
| **Unified pipeline** | Single codebase for all sources - no duplication |
| **Source transparency** | Metadata tracks origin for proper attribution |
| **Extensible** | Easy to add new source types without refactoring |
| **Maintainable** | Reuses existing components (chunking, embedding, storage) |
| **Production-ready** | Built on proven Qwen-Agent patterns |
| **Interactive** | CLI for non-programmers; API for automation |

## 🐛 Troubleshooting

### "No URLs found in search results"
- Check SERPER_API_KEY is set in `.env`
- Try different web topics

### "PDF support missing"
```bash
pip install pypdf
```

### "DOCX support missing"
```bash
pip install python-docx
```

### "Store not found"
- Use `list_stores_flow()` or menu option 6 to see available stores
- Check VECTOR_STORE_PATH in `.env`

### "Ollama connection failed"
```bash
ollama serve
ollama pull qwen3-embedding:4b
```

## 📝 License

Apache License 2.0 - Based on official Qwen-Agent patterns from QwenLM/Qwen-Agent.

---

**Created:** 2025-11-06
**Version:** 1.0
**Python:** 3.10.x only
