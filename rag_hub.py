"""RAG System Hub - Centralized management for all RAG operations.

This module coordinates all RAG functionality without code duplication.
It leverages existing components:
- production/qwen_pipeline/unified_ingestion.py (core ingestion)
- production/qwen_pipeline/ingest_cli.py (interactive interface)
- production/qwen_pipeline/query_helper.py (query utilities)
- workspace/vector_stores/ (persistent storage)

Architecture:
  rag_hub.py (this file)
     ↓
  [Unified Ingestion] ← [Query Helper] ← [CLI Interface]
     ↓                      ↓
  [FAISS Stores]      [Source Attribution]
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from production.qwen_pipeline.query_helper import query_vector_store
from production.qwen_pipeline.unified_ingestion import (
    ingest_from_web,
    ingest_local_files,
)

# Workspace locations
RAG_WORKSPACE = Path("rag_workspace")
SOURCE_DOCUMENTS = RAG_WORKSPACE / "source_documents"
VECTOR_STORES_DIR = Path("workspace/vector_stores")
QUERY_LOGS = RAG_WORKSPACE / "queries"

# Ensure directories exist
RAG_WORKSPACE.mkdir(exist_ok=True)
SOURCE_DOCUMENTS.mkdir(exist_ok=True)
VECTOR_STORES_DIR.mkdir(exist_ok=True)
QUERY_LOGS.mkdir(exist_ok=True)


def ingest_documents(
    source_paths: list[str | Path],
    store_name: str,
    source_type: str = "local_file",
) -> dict[str, Any]:
    """Ingest documents into a vector store.

    Args:
        source_paths: Paths to documents (files or folders)
        store_name: Name for the vector store
        source_type: "local_file" or "web"

    Returns:
        Ingestion result with status and statistics
    """
    paths = [Path(p) for p in source_paths]

    if source_type == "local_file":
        result = ingest_local_files(paths, store_name)
    elif source_type == "web":
        # For web ingestion, use the first "path" as a topic string
        topic = str(source_paths[0]) if source_paths else "general knowledge"
        result = ingest_from_web(topic, store_name)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    # Log ingestion event
    _log_operation("ingest", store_name, result)
    return result


def query_store(
    store_name: str,
    query: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Query a vector store.

    Args:
        store_name: Name of the vector store
        query: Search query
        k: Number of results to return

    Returns:
        List of results with content, metadata, and scores
    """
    results = query_vector_store(store_name, query, k=k)

    # Log query event
    _log_operation("query", store_name, {"query": query, "results_count": len(results)})

    return results


def list_stores() -> list[str]:
    """List all available vector stores."""
    if not VECTOR_STORES_DIR.exists():
        return []
    return [d.name for d in VECTOR_STORES_DIR.iterdir() if d.is_dir()]


def get_store_info(store_name: str) -> dict[str, Any]:
    """Get information about a vector store."""
    store_path = VECTOR_STORES_DIR / store_name
    if not store_path.exists():
        return {"status": "not_found", "store_name": store_name}

    # Count files in store
    faiss_files = list(store_path.glob("*.faiss")) + list(store_path.glob("*.pkl"))

    return {
        "status": "exists",
        "store_name": store_name,
        "path": str(store_path),
        "files": len(faiss_files),
        "created": store_path.stat().st_ctime,
        "modified": store_path.stat().st_mtime,
    }


def _log_operation(operation: str, store_name: str, details: dict) -> None:
    """Log RAG operations for analytics."""
    log_file = QUERY_LOGS / "operations.jsonl"

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "store_name": store_name,
        "details": details,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


if __name__ == "__main__":
    print("RAG Hub initialized. Use this module for centralized RAG operations.")
    print(f"Vector stores directory: {VECTOR_STORES_DIR}")
    print(f"Source documents directory: {SOURCE_DOCUMENTS}")
    print(f"Available stores: {list_stores()}")
