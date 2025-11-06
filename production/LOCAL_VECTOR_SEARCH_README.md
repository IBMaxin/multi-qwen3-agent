# Local Vector Search for Qwen-Agent

100% local semantic search using Ollama qwen3-embedding model as a drop-in replacement for DashScope's vector_search.

## Overview

This module provides `LocalVectorSearch`, a custom Qwen-Agent tool that enables semantic document search using locally-running Ollama embeddings instead of external API services.

## Architecture

```
┌─────────────────────────────────────────────┐
│         Qwen-Agent Assistant                │
│  (Chat model via Ollama, configured in `.env`)          │
└──────────────┬──────────────────────────────┘
               │
               │ uses rag_cfg
               ↓
┌─────────────────────────────────────────────┐
│      LocalVectorSearch Tool                 │
│  (inherits from BaseSearch)                 │
│  - @register_tool("local_vector_search")    │
│  - sort_by_scores() implementation          │
└──────────────┬──────────────────────────────┘
               │
               │ embeds with
               ↓
┌─────────────────────────────────────────────┐
│   Ollama qwen3-embedding:4b                 │
│  (via langchain_community.embeddings)       │
└──────────────┬──────────────────────────────┘
               │
               │ indexes in
               ↓
┌─────────────────────────────────────────────┐
│        FAISS Vector Store                   │
│  (in-memory similarity search)              │
└─────────────────────────────────────────────┘
```

## Features

- ✅ **100% Local**: No external APIs (DashScope, OpenAI, etc.)
- ✅ **Official Patterns**: Follows Qwen-Agent tool registration patterns
- ✅ **Drop-in Replacement**: Compatible with existing `rag_cfg` configuration
- ✅ **Hybrid Search**: Works alongside `keyword_search` (BM25)
- ✅ **Persistent Memory**: Built into Assistant agent
- ✅ **Multi-Document**: Query across multiple files
- ✅ **Customizable**: Configure embedding model, chunk size, etc.

## Installation

### 1. Install Dependencies

```bash
# Core requirements
pip install langchain langchain_community faiss-cpu

# For GPU acceleration (optional)
pip install faiss-gpu
```

### 2. Install Ollama Models

```bash
# Chat model
ollama pull qwen3:4b-instruct-2507-q4_K_M  # Or your chosen chat model

# Embedding model (recommended for production)
ollama pull qwen3-embedding:4b

# Alternative: Fastest, smaller size
ollama pull qwen3-embedding:0.6b

# Alternative: Best quality, larger size
ollama pull qwen3-embedding:8b
```

### 3. Verify Setup

```bash
ollama list
# Should show:
# - Your chosen chat model (e.g., `qwen3:4b-instruct-2507-q4_K_M`)
# - qwen3-embedding:4b
```

## Usage

### Basic Example

```python
from qwen_agent.agents import Assistant
from qwen_pipeline.config import get_llm_config
from qwen_pipeline.tools_custom import LocalVectorSearch  # Auto-registers

# Configure Ollama Chat Model in .env
Create or edit a `.env` file in the root of the project and set the `MODEL_NAME`:

```.env
MODEL_NAME=qwen3:4b-instruct-2507-q4_K_M
```
llm_cfg = get_llm_config()

# Create Assistant with local vector search
bot = Assistant(
    llm=llm_cfg,
    files=['document.pdf'],  # Auto-parsed and indexed
    rag_cfg={
        'max_ref_token': 20000,
        'parser_page_size': 500,
        'rag_searchers': ['local_vector_search']  # Use our custom tool!
    }
)

# Query with semantic search
messages = [{'role': 'user', 'content': 'What is this document about?'}]
for response in bot.run(messages):
    print(response[-1]['content'])
```

### Hybrid Search (Semantic + Keyword)

```python
bot = Assistant(
    llm=llm_cfg,
    files=['doc1.pdf', 'doc2.pdf'],
    rag_cfg={
        'max_ref_token': 20000,
        'parser_page_size': 500,
        'rag_searchers': [
            'local_vector_search',  # Semantic search (meaning-based)
            'keyword_search'         # BM25 search (exact match)
        ]
    }
)
```

### Custom Configuration

```python
bot = Assistant(
    llm=llm_cfg,
    rag_cfg={
        'rag_searchers': [{
            'name': 'local_vector_search',
            'embedding_model': 'qwen3-embedding:8b',  # Higher quality
            'base_url': 'http://localhost:11434',
            'max_content_length': 1500  # Shorter chunks
        }]
    }
)
```

### Multi-Document Chat with Memory

```python
bot = Assistant(
    llm=llm_cfg,
    files=[
        'https://arxiv.org/pdf/2310.08560.pdf',  # Qwen report
        'https://arxiv.org/pdf/1706.03762.pdf'   # Attention paper
    ],
    rag_cfg={'rag_searchers': ['local_vector_search', 'keyword_search']}
)

messages = []

# Turn 1
messages.append({'role': 'user', 'content': 'What is Qwen?'})
for response in bot.run(messages):
    pass
messages.extend(response)

# Turn 2 (remembers context)
messages.append({'role': 'user', 'content': 'How does it compare to transformers?'})
for response in bot.run(messages):
    pass
messages.extend(response)
```

## Configuration Options

### `LocalVectorSearch` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | str | `"qwen3-embedding:4b"` | Ollama embedding model name |
| `base_url` | str | `"http://localhost:11434"` | Ollama server URL |
| `max_content_length` | int | `2000` | Max characters per chunk |

### `rag_cfg` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_ref_token` | int | `20000` | Max tokens for RAG context |
| `parser_page_size` | int | `500` | Chunk size in tokens |
| `rag_searchers` | list | `['keyword_search', 'front_page_search']` | Search strategies |

## Implementation Details

### Following Official Qwen-Agent Patterns

The `LocalVectorSearch` tool strictly follows official patterns from `qwen_agent/tools/search_tools/vector_search.py`:

1. **Tool Registration**: Uses `@register_tool("local_vector_search")` decorator
2. **Inheritance**: Extends `BaseSearch` class
3. **Method Signature**: Implements `sort_by_scores(query: str, docs: List[Record]) -> List[Tuple[str, int, float]]`
4. **Return Format**: `(source_file, chunk_id, similarity_score)` tuples
5. **Import Strategy**: Lazy imports with helpful error messages (matches official tool)

### Key Code Excerpt

```python
from qwen_agent.tools.base import register_tool
from qwen_agent.tools.search_tools.base_search import BaseSearch

@register_tool("local_vector_search")
class LocalVectorSearch(BaseSearch):
    """Mirrors official VectorSearch but uses OllamaEmbeddings."""

    def sort_by_scores(self, query: str, docs: list[Record]) -> list[tuple[str, int, float]]:
        # Import dependencies (matches official pattern)
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS

        # Create local embeddings (replaces DashScopeEmbeddings)
        embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.base_url
        )

        # Build FAISS index and search (same as official)
        db = FAISS.from_documents(all_chunks, embeddings)
        chunk_and_score = db.similarity_search_with_score(query, k=len(all_chunks))

        return [(chk.metadata['source'], chk.metadata['chunk_id'], score)
                for chk, score in chunk_and_score]
```

## Supported File Types

Via Qwen-Agent's built-in `DocParser`:

- PDF (`.pdf`)
- Word (`.docx`)
- PowerPoint (`.pptx`)
- Text (`.txt`)
- CSV/TSV (`.csv`, `.tsv`)
- Excel (`.xlsx`, `.xls`)
- HTML (`.html`, `.htm`)

## Performance Comparison

### Embedding Model Benchmarks (Ollama)

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `qwen3-embedding:0.6b` | 639MB | ⚡⚡⚡ Fast | Good | Quick tests, development |
| `qwen3-embedding:4b` | 2.5GB | ⚡⚡ Medium | Excellent | Production (default) |
| `qwen3-embedding:8b` | 4.7GB | ⚡ Slower | Best | Maximum accuracy |

### Search Strategy Comparison

| Strategy | Type | Speed | Quality | Use Case |
|----------|------|-------|---------|----------|
| `keyword_search` | BM25 | ⚡⚡⚡ | Good | Exact terms, names |
| `local_vector_search` | Semantic | ⚡⚡ | Excellent | Concepts, meanings |
| Both (Hybrid) | Combined | ⚡⚡ | Best | Production |

## Examples

See `examples/local_vector_rag_example.py` for complete working examples:

```bash
cd examples
python local_vector_rag_example.py
```

Includes:
1. Basic RAG with vector search only
2. Hybrid RAG (vector + keyword)
3. Multi-document with memory
4. Custom embedding configuration

## Troubleshooting

### "ModuleNotFoundError: langchain"

```bash
pip install langchain langchain_community faiss-cpu
```

### "Connection refused to Ollama"

```bash
# Start Ollama server
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### "Model not found: qwen3-embedding:0.6b"

```bash
ollama pull qwen3-embedding:0.6b
ollama list  # Verify installation
```

### Slow embedding generation

- Use smaller model: `qwen3-embedding:0.6b`
- Reduce chunk size: `max_content_length: 1000`
- Limit document size: `parser_page_size: 300`

### Import errors in IDE

The lazy imports (inside methods) are intentional and follow official Qwen-Agent patterns. They provide better error messages for optional dependencies. Suppress IDE warnings with:

```python
# type: ignore[import]
```

## Comparison with Official `vector_search`

| Feature | Official `vector_search` | `LocalVectorSearch` |
|---------|-------------------------|---------------------|
| Embeddings | DashScope API | Ollama (local) |
| Internet Required | ✅ Yes | ❌ No |
| API Key Required | ✅ Yes | ❌ No |
| Cost | $ Pay per use | Free |
| Privacy | Cloud | 100% local |
| Speed | Depends on API | Depends on hardware |
| Quality | Excellent | Excellent |

## Contributing

When extending this tool:

1. **Follow Official Patterns**: Check `qwen_agent/tools/search_tools/` for reference
2. **Use Type Hints**: Python 3.10+ syntax (`list[str]` not `List[str]`)
3. **Lazy Imports**: Import optional dependencies inside methods
4. **Error Messages**: Provide helpful installation instructions
5. **Test Coverage**: Add tests to `production/tests/`

## References

- [Qwen-Agent Official Repo](https://github.com/QwenLM/Qwen-Agent)
- [Qwen-Agent Documentation](https://github.com/QwenLM/Qwen-Agent/blob/main/README.md)
- [Official VectorSearch Implementation](https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/tools/search_tools/vector_search.py)
- [Ollama Embeddings](https://ollama.com/library/qwen3-embedding)
- [LangChain Community](https://python.langchain.com/docs/integrations/text_embedding/ollama)

## License

Apache License 2.0 (same as Qwen-Agent)
