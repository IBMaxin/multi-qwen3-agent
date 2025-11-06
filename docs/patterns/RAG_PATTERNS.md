# RAG Patterns for Qwen-Agent

Official Qwen-Agent patterns for Retrieval-Augmented Generation (RAG). **Follow these patterns exclusively—no custom inventions.**

## Overview

RAG enhances LLM responses by retrieving relevant documents before generation. Qwen-Agent provides:
- **Built-in tools**: `retrieval` (vector search), `storage` (knowledge base)
- **Custom RAG tools**: Extend `BaseTool` for domain-specific retrieval
- **Multi-turn context**: Conversation history with retrieved documents

## Pattern 1: Built-in RAG with `retrieval` Tool

**Use case:** Quick RAG with official vector search tool.

```python
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config

# Create agent with retrieval tool
llm_cfg = get_llm_config()
agent = ReActChat(
    llm=llm_cfg,
    function_list=["retrieval"],  # Built-in RAG tool
    system_message="You are a helpful assistant with access to a knowledge base."
)

# Single-turn RAG query
messages = [{"role": "user", "content": "What is Qwen-Agent?"}]
responses = []
for response in agent.run(messages=messages):
    responses.append(response)
final = responses[-1]
print(final[0]["content"])
```

**Configuration via environment:**
```bash
# .env file
DASHSCOPE_API_KEY=your_key_here  # Required for retrieval tool
```

## Pattern 2: Custom RAG Tool (LocalVectorSearch)

**Use case:** Offline RAG with local embeddings and FAISS indexing.

**Implementation:** See `production/qwen_pipeline/tools_custom.py:LocalVectorSearch`

```python
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config
from qwen_pipeline.tools_custom import LocalVectorSearch

# Initialize custom RAG tool
rag_tool = LocalVectorSearch(cfg={
    "embedding_model": "http://localhost:11434/v1",
    "embedding_model_name": "qwen2.5:0.5b",
    "top_k": 3,
    "chunk_size": 500
})

# Create agent with custom tool
llm_cfg = get_llm_config()
agent = ReActChat(
    llm=llm_cfg,
    function_list=[rag_tool],
    system_message="You are a helpful assistant. Use the local_vector_search tool to find relevant information."
)

# Query with tool invocation
messages = [{"role": "user", "content": "Search for: Python design patterns"}]
for response in agent.run(messages=messages):
    print(response)
```

**Key methods:**
- `sort_by_scores(query, docs)`: Vector similarity search
- Automatic FAISS indexing for fast retrieval
- Ollama embeddings via OpenAI-compatible API

## Pattern 3: Multi-Turn RAG Conversation

**Use case:** Conversational RAG with memory.

```python
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config

llm_cfg = get_llm_config()
agent = ReActChat(
    llm=llm_cfg,
    function_list=["retrieval"],
    system_message="You are a helpful assistant with access to a knowledge base."
)

# Maintain conversation history
messages = []

# Turn 1: Initial query
messages.append({"role": "user", "content": "What is Qwen-Agent?"})
responses = []
for response in agent.run(messages=messages):
    responses.append(response)
# Extend (don't reassign) to preserve context
messages.extend(responses)

# Turn 2: Follow-up question (uses previous context)
messages.append({"role": "user", "content": "How do I install it?"})
responses = []
for response in agent.run(messages=messages):
    responses.append(response)
messages.extend(responses)

print("Final conversation:", messages)
```

**Critical pattern:** Use `messages.extend(responses)` to accumulate context. **Never** use `messages = responses` (loses history).

## Pattern 4: RAG with GroupChat Pipeline

**Use case:** Multi-agent RAG with planning, retrieval, and review.

```python
from qwen_pipeline.agent import create_agents
from qwen_pipeline.pipeline import run_pipeline

# Create 3-agent pipeline with RAG tools
manager = create_agents(["retrieval", "code_interpreter"])

# Run RAG query through pipeline
query = "Explain the differences between Assistant and ReActChat agents"
result = run_pipeline(query, manager=manager)
print(result)
```

**Agent roles:**
- **Planner**: Breaks down RAG query into sub-tasks
- **Coder**: Executes retrieval tool, processes results
- **Reviewer**: Validates retrieved information quality

## Pattern 5: Web Search to Vector Storage Persistence

**Use case:** Automated knowledge ingestion from web search into persistent FAISS vector stores.

**Key features:**
- Autonomous web search and content extraction
- Smart text chunking (500-token max, 50-token overlap)
- Retry logic for robust URL processing (2 retries per URL)
- Topic-based persistent storage organization
- Fast similarity-based retrieval via FAISS

**Implementation:** See `production/qwen_pipeline/web_rag_ingestion.py`

```python
from production.qwen_pipeline.web_rag_ingestion import (
    ingest_from_web,
    query_ingested_store,
    list_ingested_stores,
)

# Step 1: Ingest a topic from the web (autonomous agent workflow)
# - Searches topic using web_search tool
# - Extracts content from URLs with 2-retry logic
# - Chunks content intelligently
# - Embeds using qwen3-embedding:4b
# - Stores in FAISS at ./workspace/vector_stores/{topic}/
result = ingest_from_web(
    topic="machine learning algorithms",
    store_name=None,  # Auto-sanitizes from topic
    max_urls=5,       # Process up to 5 URLs
)

print(f"Ingested {result['chunks_stored']} chunks from {result['urls_processed']} URLs")

# Step 2: Query the stored vectors (returns empty list if store not found)
results = query_ingested_store(
    store_name="machine_learning_algorithms",
    query="what is gradient descent?",
    k=5,  # Top 5 results
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Source: {result['metadata']['source']}")

# Step 3: List all available stores
stores = list_ingested_stores()
print(f"Available stores: {stores}")
```

**Configuration:**
```bash
# .env file
VECTOR_STORE_PATH=./workspace/vector_stores
EMBEDDING_MODEL=qwen3-embedding:4b
EMBEDDING_BASE_URL=http://localhost:11434
SERPER_API_KEY=your_serper_api_key_here
```

**Ingestion pipeline steps:**
1. **Web Search**: Query SERPER API for relevant URLs (top 5 by default)
2. **Content Extraction**: Fetch and parse HTML from each URL with retries
3. **Smart Chunking**: Split by paragraphs, preserve context with overlap
4. **Embedding**: Generate embeddings using local Ollama model (qwen3-embedding:4b)
5. **Persistence**: Save FAISS index to disk under topic-based directory
6. **Retrieval**: Query stored vectors with similarity search

**Error handling:**
- Graceful URL extraction failures (retries up to 2 times)
- Automatic store name sanitization (removes invalid path characters)
- Query returns empty list if store doesn't exist (safe degradation)
- Full logging with structlog for debugging

**Interactive example:**
```bash
python examples/web_to_vector_ingestion.py
```

Menu options:
1. **Ingest new topic** - Search web and build vector store
2. **Query existing store** - RAG retrieval from ingested knowledge
3. **List available stores** - View all topic-based stores
4. **Exit**

**Performance characteristics:**
- Chunking: ~0.1s per 5000 tokens (local tokenizer)
- Embedding: ~0.5s per chunk (Ollama with qwen3-embedding:4b)
- Indexing: <0.1s per store (FAISS)
- Query: <0.1s for k=5 (FAISS similarity search)

**See also:**
- `production/tests/test_vector_persistence.py` - 27 comprehensive tests
- `production/qwen_pipeline/tools_custom.py:LocalVectorSearch` - Core persistence methods
- `production/qwen_pipeline/web_rag_ingestion.py` - Orchestration layer

## Pattern 6: Unified Multi-Source RAG (Web + Local Files)

**Use case:** Centralized knowledge base ingesting from web searches AND local documentation files with unified storage, retrieval, and source attribution.

**Key features:**
- Single vector store for mixed-source content
- Metadata tracking for source type, origin, and file format
- Source-aware query results (web vs local attribution)
- Consistent chunking and embedding across sources
- Interactive CLI for ingestion management

**Architecture:**
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

**Implementation:** See `production/qwen_pipeline/unified_ingestion.py`

```python
from qwen_pipeline.unified_ingestion import ingest_unified
from qwen_pipeline.tools_custom import LocalVectorSearch

# Step 1: Ingest from multiple sources into single store
result = ingest_unified(
    store_name="my_knowledge_base",
    web_topics=["machine learning basics", "Python best practices"],
    local_paths=["./docs", "./research_papers", "./README.md"],
    max_urls_per_topic=5,
    recursive=True,  # Scan subdirectories
)

print(f"Ingested {result['total_chunks']} chunks:")
print(f"  - {result['web_chunks']} from {result['web_urls_processed']} web URLs")
print(f"  - {result['local_chunks']} from {result['local_files_processed']} local files")

# Step 2: Query with source attribution
vector_tool = LocalVectorSearch()
query_result = vector_tool.query_with_source_attribution(
    store_name="my_knowledge_base",
    query="What are Python decorators?",
    k=5
)

print(f"\nQuery: {query_result['query']}")
print(f"Total results: {query_result['total_results']}")
print(f"  - Web sources: {len(query_result['web_results'])}")
print(f"  - Local sources: {len(query_result['local_results'])}")

# Step 3: Examine results with source metadata
for result in query_result['all_results']:
    metadata = result['metadata']
    print(f"\nSource type: {metadata['source_type']}")
    print(f"Origin: {metadata['source']}")

    if metadata['source_type'] == 'web':
        print(f"Topic: {metadata['topic']}")
    elif metadata['source_type'] == 'local_file':
        print(f"File type: {metadata['file_type']}")

    print(f"Content: {result['content'][:200]}...")
    print(f"Similarity score: {result['score']:.4f}")

# Step 4: View store statistics
stats = vector_tool.get_store_statistics("my_knowledge_base")
print(f"\n📊 Store Statistics:")
print(f"Total documents: {stats['total_documents']}")
print(f"Web documents: {stats['web_documents']}")
print(f"Local documents: {stats['local_documents']}")
print(f"File types: {stats['file_types']}")
print(f"Topics: {stats['topics']}")
```

**Interactive CLI usage:**
```bash
# Start unified ingestion CLI
python -m qwen_pipeline.ingest_cli

# Menu options:
# 1. Ingest from web topics only
# 2. Ingest from local files only
# 3. Ingest from both (unified) ← Recommended
# 4. Query with source attribution
# 5. View store statistics
# 6. List all vector stores
```

**Supported local file formats:**
- `.md` - Markdown (direct read)
- `.txt` - Plain text (direct read)
- `.rst` - reStructuredText (direct read)
- `.pdf` - PDF (requires: `pip install pypdf`)
- `.docx` - Word documents (requires: `pip install python-docx`)

**Metadata schema:**

For **web sources**:
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

For **local files**:
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

**Benefits of unified approach:**
- **No code duplication**: Reuses chunking, embedding, storage logic
- **Single source of truth**: One vector store for all knowledge
- **Consistent retrieval**: Same query interface for all sources
- **Source transparency**: Metadata tracks origin for attribution
- **Extensible**: Easy to add new source types (APIs, databases, etc.)

**Future enhancements (roadmap):**
- **Reranking**: Add cross-encoder reranking for improved relevance
- **User feedback**: Track query effectiveness and result quality
- **Incremental updates**: Detect changed files and re-ingest only deltas
- **Hybrid search**: Combine vector similarity with keyword search
- **Source prioritization**: Weight results by source type or recency

**Configuration:**
```bash
# .env file
VECTOR_STORE_PATH=./workspace/vector_stores
EMBEDDING_MODEL=qwen3-embedding:4b
EMBEDDING_BASE_URL=http://localhost:11434
SERPER_API_KEY=your_serper_api_key_here  # For web search
```

**Performance characteristics:**
- Local file loading: ~0.01s per file (text formats), ~0.5s per file (PDF)
- Web extraction: ~2-3s per URL (with retries)
- Chunking: ~0.1s per 5000 tokens
- Embedding: ~0.5s per chunk (qwen3-embedding:4b)
- Query: <0.1s for k=5 results

**Example: Complete workflow**
```python
from qwen_pipeline.unified_ingestion import ingest_unified
from qwen_pipeline.tools_custom import LocalVectorSearch

# 1. Create unified knowledge base
result = ingest_unified(
    store_name="ml_research",
    web_topics=["neural networks", "gradient descent"],
    local_paths=["./ml_papers", "./ml_notes.md"],
    max_urls_per_topic=3,
)

# 2. Query across all sources
tool = LocalVectorSearch()
results = tool.query_with_source_attribution(
    "ml_research",
    "explain backpropagation",
    k=5
)

# 3. Use in RAG agent
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config

agent = ReActChat(
    llm=get_llm_config(),
    function_list=[tool],
    system_message="Use local_vector_search to find ML research information."
)

messages = [{"role": "user", "content": "Explain backpropagation with sources"}]
for response in agent.run(messages=messages):
    print(response)
```

**See also:**
- `production/qwen_pipeline/unified_ingestion.py` - Core unified ingestion logic
- `production/qwen_pipeline/ingest_cli.py` - Interactive CLI
- `examples/unified_rag_demo.py` - Complete demonstration
- `production/qwen_pipeline/tools_custom.py` - Enhanced LocalVectorSearch with attribution

## Pattern 7: Custom RAG with GitHub Code Search

**Use case:** RAG over GitHub repositories.

```python
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config
from qwen_pipeline.tools_github import GitHubSearchTool

# Initialize GitHub search tool
github_tool = GitHubSearchTool(cfg={
    "github_token": "your_github_token_here"  # Optional, increases rate limits
})

llm_cfg = get_llm_config()
agent = ReActChat(
    llm=llm_cfg,
    function_list=[github_tool],
    system_message="You are a code assistant. Use github_search to find relevant code snippets."
)

# Search GitHub for code examples
messages = [{
    "role": "user",
    "content": "Find examples of ReActChat usage in QwenLM/Qwen-Agent repository"
}]
for response in agent.run(messages=messages):
    print(response)
```

**Tool parameters:**
- `repo`: Repository in format `owner/repo`
- `query`: Search query string

## Pattern 8: RAG Configuration Best Practices

**Environment-driven configuration:**
```bash
# .env file
MODEL_SERVER=http://localhost:11434/v1
MODEL_NAME=qwen3:4b-instruct-2507-q4_K_M
DASHSCOPE_API_KEY=your_key_here  # For built-in retrieval
GITHUB_TOKEN=your_token_here      # For GitHub search
ENABLE_ALL_OFFICIAL_TOOLS=false   # Manual tool selection
```

**Code-driven configuration:**
```python
from qwen_pipeline.config import get_llm_config

# LLM config with RAG-optimized settings
llm_cfg = get_llm_config()
# Returns:
# {
#     "model": "qwen3:4b-instruct-2507-q4_K_M",
#     "model_server": "http://localhost:11434/v1",
#     "api_key": "EMPTY",
#     "generate_cfg": {
#         "top_p": 0.8,
#         "temperature": 0.7,  # Higher for diverse RAG responses
#         "max_input_tokens": 6000  # Sufficient for multi-doc context
#     }
# }
```

## Pattern 9: Custom RAG Tool Template

**Use case:** Create domain-specific RAG tools.

```python
from qwen_agent.tools.base import BaseTool, register_tool
import json5
import json

@register_tool("custom_rag")
class CustomRAGTool(BaseTool):
    description = "Custom RAG tool for domain-specific retrieval."
    parameters = [
        {"name": "query", "type": "string", "required": True, "description": "Search query"}
    ]

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        # Initialize your RAG backend (vector DB, search engine, etc.)
        self.retriever = self._init_retriever(cfg or {})

    def _init_retriever(self, cfg: dict):
        # Custom retriever initialization
        return YourRetrieverClass(cfg)

    def call(self, params: str, **kwargs) -> str:
        """
        Execute RAG retrieval.

        Args:
            params: JSON5 string like '{"query": "search term"}'
            **kwargs: Additional context (ignored per Qwen-Agent patterns)

        Returns:
            JSON string with retrieved documents
        """
        # Parse JSON5 input
        params_dict = json5.loads(params)
        query = params_dict.get("query", "")

        if not query:
            return json.dumps({"error": "Missing required parameter: query"})

        try:
            # Perform retrieval
            docs = self.retriever.search(query)

            # Format results
            results = [
                {"content": doc["text"], "source": doc["metadata"]}
                for doc in docs
            ]

            return json.dumps({"results": results, "count": len(results)})

        except Exception as e:
            return json.dumps({"error": f"Retrieval failed: {str(e)}"})
```

**Usage:**
```python
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config

# Tool is auto-registered via @register_tool decorator
agent = ReActChat(
    llm=get_llm_config(),
    function_list=["custom_rag"],  # Reference by string name
    system_message="Use custom_rag tool for specialized searches."
)
```

## Common Pitfalls

### ❌ Wrong: Losing conversation context
```python
messages = [{"role": "user", "content": "Query"}]
messages = agent.run(messages=messages)  # OVERWRITES history!
messages.append({"role": "user", "content": "Follow-up"})  # Previous context lost
```

### ✅ Correct: Preserving context
```python
messages = [{"role": "user", "content": "Query"}]
responses = list(agent.run(messages=messages))
messages.extend(responses)  # APPENDS to history
messages.append({"role": "user", "content": "Follow-up"})  # Context preserved
```

### ❌ Wrong: Custom tool I/O format
```python
def call(self, params: dict, **kwargs) -> dict:  # Wrong types!
    return {"results": [...]}
```

### ✅ Correct: Official Qwen-Agent I/O
```python
def call(self, params: str, **kwargs) -> str:  # Correct types
    params_dict = json5.loads(params)  # Parse JSON5 input
    return json.dumps({"results": [...]})  # Return JSON string
```

### ❌ Wrong: Direct API key in code
```python
agent = ReActChat(
    llm={"api_key": "sk-xxxxx", ...},  # Hardcoded secret!
    function_list=["retrieval"]
)
```

### ✅ Correct: Environment-driven config
```python
from qwen_pipeline.config import get_llm_config

agent = ReActChat(
    llm=get_llm_config(),  # Reads from .env securely
    function_list=["retrieval"]
)
```

## Official Examples Reference

**Qwen-Agent repository examples:**
- `examples/assistant_add_custom_tool.py`: Custom tool registration
- `examples/react_data_analysis.py`: ReActChat with tools
- `examples/group_chat.py`: Multi-agent pipeline
- `examples/assistant_growing_memory.py`: Multi-turn conversation

**Local examples:**
- `examples/qwen3-agentV2-complete.py`: Full toolset demo (498 lines)
- `examples/run_qwen_gui.py`: Gradio GUI with RAG
- `production/qwen_pipeline/tools_custom.py`: LocalVectorSearch implementation
- `production/qwen_pipeline/tools_github.py`: GitHubSearchTool implementation

## Performance Tips

1. **Chunk size tuning**: Adjust `chunk_size` (300-1000) based on document structure
2. **Top-k optimization**: Start with `top_k=3`, increase only if needed
3. **Embedding model**: Use `qwen2.5:0.5b` for speed, larger models for accuracy
4. **FAISS indexing**: Pre-build index for large document sets (>1000 docs)
5. **Context window**: Monitor `max_input_tokens` to avoid truncation

## Testing RAG Tools

```python
# Unit test pattern for custom RAG tools
from unittest.mock import patch, MagicMock
import json

def test_custom_rag_tool():
    tool = CustomRAGTool(cfg={"api_key": "test"})

    # Mock retriever
    with patch.object(tool, "retriever") as mock_retriever:
        mock_retriever.search.return_value = [
            {"text": "Result 1", "metadata": {"source": "doc1"}}
        ]

        # Test tool invocation
        params = json.dumps({"query": "test query"})
        result_str = tool.call(params)
        result = json.loads(result_str)

        assert result["count"] == 1
        assert result["results"][0]["content"] == "Result 1"
```

**See:** `production/tests/test_tools_custom.py` for comprehensive test examples.

---

**Last updated:** 2025-11-06
**Qwen-Agent version:** Compatible with official patterns as of Nov 2025
