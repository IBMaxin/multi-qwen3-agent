# RAG Hub - Quick Reference

Complete copy-paste examples for all common use cases.

## 📌 Installation Check

```python
from rag_hub import ingest_documents, query_store, list_stores
print("✓ rag_hub imported successfully")
```

## 🚀 Quick Start (30 seconds)

```python
from rag_hub import ingest_documents, query_store

# 1. Ingest
ingest_documents(["rag_workspace/source_documents"], "my_docs")

# 2. Query
results = query_store("my_docs", "What is this?", k=5)

# 3. Display
for r in results:
    print(f"- {r['content'][:100]} ({r['score']:.0%})")
```

## 📚 Complete Examples

### Example 1: Ingest Local Files

```python
from rag_hub import ingest_documents

result = ingest_documents(
    source_paths=["rag_workspace/source_documents/markdown"],
    store_name="my_knowledge_base",
    source_type="local_file"
)

print(f"✓ Ingested {result.get('chunks', 0)} chunks")
```

### Example 2: Query with Results

```python
from rag_hub import query_store

results = query_store(
    store_name="my_knowledge_base",
    query="How do I use this?",
    k=5
)

for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result['score']:.2%}")
    print(f"   {result['content'][:150]}...")
    print(f"   Source: {result['metadata']['source']}")
```

### Example 3: List Stores

```python
from rag_hub import list_stores, get_store_info

stores = list_stores()
print(f"Available: {stores}")

for store in stores:
    info = get_store_info(store)
    print(f"  {store}: {info['files']} files")
```

### Example 4: Ingest Multiple Sources

```python
from rag_hub import ingest_documents

result = ingest_documents(
    source_paths=[
        "rag_workspace/source_documents/markdown",
        "rag_workspace/source_documents/code",
        "single_file.md"
    ],
    store_name="combined"
)
```

### Example 5: Web Ingestion

```python
from rag_hub import ingest_documents

result = ingest_documents(
    source_paths=["Python best practices"],
    store_name="web_knowledge",
    source_type="web"
)
```

### Example 6: Batch Queries

```python
from rag_hub import query_store

queries = ["What is X?", "How do I Y?", "Explain Z"]

for query in queries:
    results = query_store("my_store", query, k=3)
    print(f"Q: {query}")
    print(f"   A: {results[0]['content'][:100]}\n")
```

### Example 7: Filtered Results (High Relevance Only)

```python
from rag_hub import query_store

results = query_store("docs", "specific topic", k=10)

# Keep only high-relevance results
high_quality = [r for r in results if r["score"] > 0.7]

print(f"Found {len(high_quality)} highly relevant results")
for r in high_quality:
    print(f"  - {r['content'][:80]}")
```

### Example 8: Extract Structured Data

```python
import json
from rag_hub import query_store

results = query_store("code_base", "function signature", k=5)

structured = [
    {
        "content": r["content"],
        "relevance": f"{r['score']:.0%}",
        "source": r["metadata"].get("source", "unknown"),
    }
    for r in results
    if r["score"] > 0.6
]

print(json.dumps(structured, indent=2))
```

## 📊 Analytics

### View Operation Logs

```python
import json
from pathlib import Path

log_file = Path("rag_workspace/queries/operations.jsonl")

print("Last 10 operations:")
with open(log_file) as f:
    for line in f.readlines()[-10:]:
        event = json.loads(line)
        print(f"  {event['timestamp']}: {event['operation']} on {event['store_name']}")
```

### Count Total Chunks

```python
import json
from pathlib import Path

log_file = Path("rag_workspace/queries/operations.jsonl")
total_chunks = 0

with open(log_file) as f:
    for line in f:
        event = json.loads(line)
        if event["operation"] == "ingest":
            chunks = event["details"].get("chunks", 0)
            total_chunks += chunks

print(f"Total chunks: {total_chunks}")
```

## 🔧 Integration with Agents

### Use as Agent Tool

```python
import json
from qwen_agent.tools import BaseTool, register_tool
from rag_hub import query_store


@register_tool("knowledge_search")
class KnowledgeSearchTool(BaseTool):
    description = "Search the knowledge base for answers"
    parameters = [
        {"name": "query", "type": "string", "required": True},
        {"name": "store_name", "type": "string", "required": False, "default": "qwen_agent_docs"},
    ]

    def call(self, params: str, **kwargs) -> str:
        params_dict = json.loads(params)
        results = query_store(
            params_dict["store_name"],
            params_dict["query"],
            k=5
        )
        return json.dumps(results)
```

## 🔍 Troubleshooting

### Store Not Found

```python
from rag_hub import list_stores

stores = list_stores()
if not stores:
    print("No stores found. Ingest some documents first.")
else:
    print(f"Available stores: {stores}")
```

### Low Relevance Scores

```python
from rag_hub import query_store

# Try a more specific query
results = query_store(
    "store_name",
    "specific feature with implementation details",  # More specific
    k=10  # Get more results
)

high_quality = [r for r in results if r["score"] > 0.5]
print(f"Found {len(high_quality)} relevant results")
```

### Check Store Info

```python
from rag_hub import get_store_info

info = get_store_info("store_name")
print(f"Store exists: {info['status']}")
print(f"Files: {info.get('files', 0)}")
print(f"Path: {info.get('path', 'N/A')}")
```

## 📖 Documentation

- **Full Guide**: `RAG_WORKFLOW.md`
- **Architecture**: `RAG_INTEGRATION_SUMMARY.md`
- **Query Functions**: `production/qwen_pipeline/query_helper.py`
- **Ingestion**: `production/qwen_pipeline/unified_ingestion.py`

## ✨ Common Workflows

### Workflow 1: From Empty to Operational

```bash
# 1. Copy docs
cp ~/myproject/docs/*.md rag_workspace/source_documents/markdown/

# 2. Python
python

# 3. Inside Python
from rag_hub import ingest_documents, query_store

ingest_documents(["rag_workspace/source_documents/markdown"], "myproject")

results = query_store("myproject", "What features does it have?")
for r in results[:3]:
    print(r['content'][:100])
```

### Workflow 2: Multi-Source Knowledge Base

```python
from rag_hub import ingest_documents

# Stage 1: Local docs
ingest_documents(
    ["rag_workspace/source_documents/markdown"],
    "knowledge_base"
)

# Stage 2: Add code
ingest_documents(
    ["rag_workspace/source_documents/code"],
    "knowledge_base"  # Same store - merges!
)

# Stage 3: Add web content
ingest_documents(
    ["AI best practices"],
    "knowledge_base",
    source_type="web"
)
```

### Workflow 3: Daily Updates

```python
import json
from pathlib import Path
from rag_hub import ingest_documents, query_store, list_stores

# Update existing store
ingest_documents(
    ["rag_workspace/source_documents/markdown"],
    "myproject"
)

# Verify update
results = query_store("myproject", "latest features")
print(f"Found {len(results)} recent results")
```

## 🎯 Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `ingest_documents()` | Ingest files/web | `ingest_documents(["path"], "store")` |
| `query_store()` | Search | `query_store("store", "query")` |
| `list_stores()` | List available | `list_stores()` |
| `get_store_info()` | Get metadata | `get_store_info("store")` |

---

**Status**: Production Ready
**Python**: 3.10+
**Last Updated**: 2024-12-20
