"""RAG Workspace - Dedicated area for RAG operations with central hub.

⚠️  PRIMARY ENTRY POINT: Use rag_hub.py for ALL RAG operations!

This avoids code duplication and ensures:
  ✅ Single source of truth (unified_ingestion.py)
  ✅ Centralized logging (operations.jsonl)
  ✅ No redundant implementations

Directory Structure:
  source_documents/  - Staging area (organize by type)
    ├── markdown/   - .md files
    ├── code/       - .py, .ts, .java, etc.
    ├── pdfs/       - .pdf documents
    └── other/      - .txt, .docx, .rst, etc.

  queries/          - Operational logs (auto-created)
    └── operations.jsonl  - All ingestion/query events

  vector_stores/    - DEPRECATED (use workspace/vector_stores/ via rag_hub)

Quick Start:
  from rag_hub import ingest_documents, query_store, list_stores

  # 1. Stage documents
  # cp *.md rag_workspace/source_documents/markdown/

  # 2. Ingest
  ingest_documents(['rag_workspace/source_documents/markdown'], 'my_docs')

  # 3. Query
  results = query_store('my_docs', 'What is...?', k=5)

  # 4. List stores
  stores = list_stores()

See RAG_WORKFLOW.md for complete documentation and examples.
"""

from pathlib import Path

# Workspace paths
RAG_WORKSPACE = Path(__file__).parent
SOURCE_DOCUMENTS = RAG_WORKSPACE / "source_documents"
QUERIES_DIR = RAG_WORKSPACE / "queries"

# For reference only - actual vectors stored in workspace/vector_stores/
DEPRECATED_VECTOR_STORES = RAG_WORKSPACE / "vector_stores"

# Subdirectories for source documents
MARKDOWN_DOCS = SOURCE_DOCUMENTS / "markdown"
CODE_DOCS = SOURCE_DOCUMENTS / "code"
PDF_DOCS = SOURCE_DOCUMENTS / "pdfs"
OTHER_DOCS = SOURCE_DOCUMENTS / "other"

# Initialize directories
RAG_WORKSPACE.mkdir(exist_ok=True)
SOURCE_DOCUMENTS.mkdir(exist_ok=True)
MARKDOWN_DOCS.mkdir(exist_ok=True)
CODE_DOCS.mkdir(exist_ok=True)
PDF_DOCS.mkdir(exist_ok=True)
OTHER_DOCS.mkdir(exist_ok=True)
QUERIES_DIR.mkdir(exist_ok=True)

__all__ = [
    "RAG_WORKSPACE",
    "SOURCE_DOCUMENTS",
    "MARKDOWN_DOCS",
    "CODE_DOCS",
    "PDF_DOCS",
    "OTHER_DOCS",
    "QUERIES_DIR",
]

__all__ = [
    "RAG_WORKSPACE",
    "SOURCE_DOCUMENTS",
    "VECTOR_STORES",
    "QUERIES_DIR",
    "MARKDOWN_DOCS",
    "CODE_DOCS",
    "PDF_DOCS",
    "OTHER_DOCS",
]
