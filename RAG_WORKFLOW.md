# RAG Workflow - Complete Integration Guide

## 🏗️ Architecture Overview

```
rag_hub.py (Single Entry Point)
    ↓
[Ingestion]          [Query]         [Analytics]
    ↓                   ↓                  ↓
unified_ingestion.py  query_helper.py   operations.jsonl
    ↓                   ↓
  FAISS        [Vector Stores]
             workspace/vector_stores/
```

## 📁 Directory Structure

```
workspace/
├── vector_stores/              # Persistent FAISS indexes
│   ├── qwen_agent_docs/        # Example: Qwen documentation
│   ├── github_code/            # Example: GitHub code
│   └── {store_name}/           # Any custom store
└── tools/

rag_workspace/
├── source_documents/           # Staging area for ingestion
│   ├── markdown/               # .md files
│   ├── code/                   # .py, .js, etc.
│   ├── pdf/                    # .pdf files
│   └── web/                    # Downloaded web content
├── vector_stores/              # (UNUSED - kept for reference)
└── queries/
    └── operations.jsonl        # Query/ingestion logs
```

## 🚀 Quick Start

### 1. Ingest Local Files

```python
from rag_hub import ingest_documents

# Single source
result = ingest_documents(
    source_paths=["rag_workspace/source_documents/markdown"],
    store_name="my_knowledge_base",
    source_type="local_file"
)
print(result)
# Output: {"status": "success", "chunks": 42, "store": "my_knowledge_base"}

# Multiple sources
result = ingest_documents(
    source_paths=["folder1", "folder2", "file.md"],
    store_name="combined_store",
    source_type="local_file"
)
```

### 2. Query Documents

```python
from rag_hub import query_store

results = query_store(
    store_name="my_knowledge_base",
    query="How to use this feature?",
    k=5
)

for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['score']:.2%}")
    print(f"   Content: {result['content'][:200]}...")
    print(f"   Source: {result['metadata'].get('source', 'unknown')}\n")
```

### 3. Ingest from Web

```python
from rag_hub import ingest_documents

result = ingest_documents(
    source_paths=["Python best practices"],  # Topic string
    store_name="web_knowledge",
    source_type="web"
)
```

### 4. List & Inspect Stores

```python
from rag_hub import list_stores, get_store_info

# List all stores
stores = list_stores()
print(f"Available stores: {stores}")

# Get detailed info
info = get_store_info("my_knowledge_base")
print(f"Store info: {info}")
```

## 📊 Analytics & Logging

All operations are automatically logged to `rag_workspace/queries/operations.jsonl`:

```bash
# View recent operations
tail -20 rag_workspace/queries/operations.jsonl
```

Each entry contains:
```json
{
  "timestamp": "2024-12-20T15:30:45.123456+00:00",
  "operation": "ingest|query",
  "store_name": "my_knowledge_base",
  "details": {...}
}
```

## 🔄 Integration with Production Pipeline

### Using rag_hub in agents:

```python
from rag_hub import query_store

class DocumentSearchTool:
    def call(self, params):
        store_name = params.get("store_name", "qwen_agent_docs")
        query = params.get("query")
        results = query_store(store_name, query, k=5)
        return json.dumps(results)
```

### Using rag_hub in CLI:

```python
from rag_hub import list_stores, query_store

def main():
    stores = list_stores()
    print(f"Available: {stores}")
    # ... user selects store and query ...
    results = query_store(store, user_query, k=5)
```

## 🎯 Common Workflows

### Workflow 1: Ingest Entire Documentation Folder

```bash
# Copy docs to staging
cp -r ../docs/api rag_workspace/source_documents/markdown/
cp -r ../docs/guides rag_workspace/source_documents/markdown/

# Ingest
python -c "
from rag_hub import ingest_documents
ingest_documents(
    ['rag_workspace/source_documents/markdown'],
    'api_documentation'
)
"
```

### Workflow 2: Build Specialized Code Store

```bash
# Stage code files
mkdir -p rag_workspace/source_documents/code
cp ../src/**/*.py rag_workspace/source_documents/code/

# Ingest with code-specific chunking
ingest_documents(['rag_workspace/source_documents/code'], 'codebase')
```

### Workflow 3: Multi-Source Knowledge Base

```python
from rag_hub import ingest_documents

# Ingest local files
ingest_documents(['rag_workspace/source_documents'], 'knowledge', 'local_file')

# Add web content
ingest_documents(['AI research papers'], 'knowledge', 'web')  # Merges with existing
```

## ⚙️ Configuration

### Environment Variables (in `.env`)

```env
# Vector store location
RAG_VECTOR_STORE_DIR=workspace/vector_stores

# Embedding model
EMBEDDING_MODEL=qwen3-embedding:4b

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Search
K_RESULTS=5
RELEVANCE_THRESHOLD=0.5
```

### Supported File Types

| Type | Extensions | Loader |
|------|-----------|--------|
| Markdown | `.md` | Direct text |
| Text | `.txt` | Direct text |
| Code | `.py`, `.js`, `.ts`, etc. | Direct text |
| PDF | `.pdf` | pypdf |
| Word | `.docx` | python-docx |
| RST | `.rst` | Direct text |

## 🔍 Troubleshooting

### Problem: "Store not found"

```bash
# Check what stores exist
python -c "from rag_hub import list_stores; print(list_stores())"

# Verify file system
ls -la workspace/vector_stores/
```

### Problem: "Slow queries"

```bash
# Check store size
python -c "
from rag_hub import get_store_info
info = get_store_info('store_name')
print(f'Files: {info[\"files\"]}')
"

# Consider splitting into multiple stores if >10k chunks
```

### Problem: "Low relevance scores"

1. **Adjust query**: Be more specific or provide more context
2. **Check indexing**: Verify documents were ingested correctly
3. **Increase k**: Get more results and filter manually
4. **Tune parameters**: Adjust CHUNK_SIZE, RELEVANCE_THRESHOLD

## 📚 Supported Operations

| Operation | Function | Returns |
|-----------|----------|---------|
| Ingest local | `ingest_documents(...)` | `{"status": "success", ...}` |
| Ingest web | `ingest_documents(..., source_type="web")` | `{"status": "success", ...}` |
| Query | `query_store(store, query)` | `[{"content": "...", "score": 0.95, ...}]` |
| List | `list_stores()` | `["store1", "store2"]` |
| Inspect | `get_store_info(store)` | `{"status": "exists", ...}` |

## 🔐 No Code Duplication

This architecture ensures:

✅ **Single source of truth**: All ingestion via `unified_ingestion.py`
✅ **Single query interface**: All queries via `query_helper.py`
✅ **Centralized logging**: All operations logged to `operations.jsonl`
✅ **Clear separation**: `rag_hub.py` orchestrates without reimplementing

Each component has one responsibility:
- **unified_ingestion.py**: Load & chunk documents
- **query_helper.py**: FAISS search & retrieval
- **rag_hub.py**: Orchestration & logging (this module)

## 📖 Documentation Links

- Full system docs: `UNIFIED_RAG_SYSTEM.md`
- Ingestion patterns: `docs/patterns/RAG_PATTERNS.md`
- Query examples: `production/qwen_pipeline/query_helper.py`
- CLI interface: `production/qwen_pipeline/ingest_cli.py`

---

**Last Updated**: 2024-12-20
**Status**: ✅ Production Ready
**Python Version**: 3.10+
