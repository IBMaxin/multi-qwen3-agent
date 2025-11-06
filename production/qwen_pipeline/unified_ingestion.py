"""Unified Document Ingestion Pipeline for Web and Local Sources.

This module provides a centralized ingestion system that handles:
1. Web content (via web search and extraction)
2. Local documentation files (MD, TXT, PDF, DOCX)
3. Unified metadata tracking (source type, origin, timestamps)
4. Single FAISS vector store with mixed content

Architecture:
- Reuses web_rag_ingestion.py for web workflow
- Adds local file loaders with format detection
- Maintains consistent chunking and embedding
- Tracks source metadata for attribution

Usage:
    from qwen_pipeline.unified_ingestion import ingest_unified

    # Ingest from multiple sources into one store
    result = ingest_unified(
        store_name="my_knowledge_base",
        web_topics=["machine learning", "python best practices"],
        local_paths=["./docs", "./research_papers"],
        max_urls_per_topic=5
    )

Copyright: Based on Qwen-Agent patterns from QwenLM/Qwen-Agent
License: Apache License 2.0
"""

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from qwen_pipeline.tools_custom import LocalVectorSearch
from qwen_pipeline.web_rag_ingestion import (
    INGESTION_CONFIG,
    create_ingestion_agent,
    extract_url_with_retry,
)

# Optional imports for file format support
try:
    from pypdf import PdfReader

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()

# Supported file extensions for local ingestion
SUPPORTED_EXTENSIONS = {
    ".md": "markdown",
    ".txt": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".rst": "restructuredtext",
}


def _load_text_file(file_path: Path) -> str:
    """Load a plain text file (md, txt, rst)."""
    return file_path.read_text(encoding="utf-8")


def _load_pdf_file(file_path: Path) -> str | None:
    """Load a PDF file."""
    if not PDF_AVAILABLE:
        logger.warning(
            "pdf_support_missing",
            file=str(file_path),
            hint="Install pypdf: pip install pypdf",
        )
        return None
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() + "\n" for page in reader.pages)


def _load_docx_file(file_path: Path) -> str | None:
    """Load a DOCX file."""
    if not DOCX_AVAILABLE:
        logger.warning(
            "docx_support_missing",
            file=str(file_path),
            hint="Install python-docx: pip install python-docx",
        )
        return None
    doc = DocxDocument(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


# Map suffixes to loader functions
_LOADERS: dict[str, Callable[[Path], str | None]] = {
    ".md": _load_text_file,
    ".txt": _load_text_file,
    ".rst": _load_text_file,
    ".pdf": _load_pdf_file,
    ".docx": _load_docx_file,
}


def load_local_file(file_path: Path) -> str | None:
    """Load content from a local file with format detection.

    Supports: .md, .txt, .pdf, .docx, .rst

    Args:
        file_path: Path to file to load

    Returns:
        File content as string, or None if loading failed
    """
    suffix = file_path.suffix.lower()
    loader = _LOADERS.get(suffix)

    if not loader:
        logger.warning("unsupported_file_type", file=str(file_path), suffix=suffix)
        return None

    try:
        return loader(file_path)
    except Exception:
        logger.exception("file_load_failed", file=str(file_path))
        return None


def ingest_local_files(
    paths: list[str] | list[Path],
    recursive: bool = True,
) -> dict[str, Any]:
    """Ingest local documentation files into memory structures.

    Scans directories for supported file types, loads content, and chunks.
    Does NOT store in vector DB yet - use ingest_unified() for that.

    Args:
        paths: List of file or directory paths to ingest
        recursive: Recursively scan subdirectories (default: True)

    Returns:
        Dictionary with:
        {
            "chunks": ["text chunk 1", "text chunk 2", ...],
            "metadata": [{"source": "path", "type": "markdown", ...}, ...],
            "files_processed": 5,
            "files_failed": 1,
        }

    Example:
        >>> result = ingest_local_files(["./docs", "./README.md"])
        >>> print(f"Loaded {len(result['chunks'])} chunks")
    """
    all_chunks = []
    all_metadata = []
    files_processed = 0
    files_failed = 0

    # Normalize paths
    normalized_paths = [Path(p) for p in paths]

    # Collect all files to process
    files_to_process: list[Path] = []

    for path in normalized_paths:
        if not path.exists():
            logger.warning("path_not_found", path=str(path))
            continue

        if path.is_file():
            files_to_process.append(path)
        elif path.is_dir():
            # Scan directory
            glob_pattern = "**/*" if recursive else "*"
            for ext in SUPPORTED_EXTENSIONS:
                files_to_process.extend(path.glob(f"{glob_pattern}{ext}"))

    logger.info("files_discovered", count=len(files_to_process))

    # Process each file
    for file_path in files_to_process:
        content = load_local_file(file_path)

        if not content:
            files_failed += 1
            continue

        # Check minimum content length
        if len(content) < INGESTION_CONFIG["min_content_length"]:
            logger.warning("content_too_short", file=str(file_path), length=len(content))
            files_failed += 1
            continue

        # Chunk the content
        chunks = LocalVectorSearch.chunk_text(
            content,
            max_chunk_tokens=INGESTION_CONFIG["max_chunk_tokens"],
            overlap_tokens=INGESTION_CONFIG["chunk_overlap"],
        )

        # Create metadata for each chunk
        file_type = SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "unknown")
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append(
                {
                    "source": str(file_path.absolute()),
                    "source_type": "local_file",
                    "file_type": file_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        files_processed += 1
        logger.info("file_processed", file=str(file_path), chunks=len(chunks))

    logger.info(
        "local_ingestion_completed",
        files_processed=files_processed,
        files_failed=files_failed,
        total_chunks=len(all_chunks),
    )

    return {
        "chunks": all_chunks,
        "metadata": all_metadata,
        "files_processed": files_processed,
        "files_failed": files_failed,
    }


def ingest_from_web_unified(topics: list[str], max_urls_per_topic: int = 5) -> dict[str, Any]:
    """Ingest web content from multiple topics with unified metadata.

    Extended version of web_rag_ingestion.ingest_from_web() that:
    1. Handles multiple topics
    2. Adds source_type metadata for unified storage
    3. Returns in-memory structures (doesn't store yet)

    Args:
        topics: List of search topics
        max_urls_per_topic: Max URLs to process per topic (default: 5)

    Returns:
        Dictionary with:
        {
            "chunks": ["text chunk 1", ...],
            "metadata": [{"source": "url", "source_type": "web", ...}, ...],
            "topics_processed": 2,
            "total_urls": 8,
            "urls_failed": 1,
        }

    Example:
        >>> result = ingest_from_web_unified(["Python RAG", "FAISS tutorial"])
        >>> print(f"Loaded {len(result['chunks'])} web chunks")
    """
    agent = create_ingestion_agent()
    all_chunks = []
    all_metadata = []
    total_urls = 0
    total_failed = 0

    for topic in topics:
        logger.info("processing_topic", topic=topic)

        # Step 1: Search web
        search_messages = [{"role": "user", "content": f"Search the web for: {topic}"}]
        search_responses = list(agent.run(messages=search_messages))

        # Parse URLs from search results
        urls = []
        for response in search_responses:
            if isinstance(response, dict) and "content" in response:
                content = response["content"]
                if "http" in content:
                    url_pattern = r"https?://[^\s\)\]]+"
                    found_urls = re.findall(url_pattern, content)
                    urls.extend(found_urls[:max_urls_per_topic])
                    break

        if not urls:
            logger.warning("no_urls_found_for_topic", topic=topic)
            continue

        logger.info("urls_found_for_topic", topic=topic, count=len(urls))

        # Step 2: Extract content from each URL
        for url in urls[:max_urls_per_topic]:
            content = extract_url_with_retry(
                agent,
                url,
                max_retries=int(INGESTION_CONFIG["retry_failed_urls"]),
                delay=INGESTION_CONFIG["retry_delay"],
            )

            if not content:
                total_failed += 1
                continue

            total_urls += 1

            # Chunk the content
            chunks = LocalVectorSearch.chunk_text(
                content,
                max_chunk_tokens=INGESTION_CONFIG["max_chunk_tokens"],
                overlap_tokens=INGESTION_CONFIG["chunk_overlap"],
            )

            # Add metadata for each chunk
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append(
                    {
                        "source": url,
                        "source_type": "web",
                        "topic": topic,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            logger.info("url_processed", url=url, chunks=len(chunks))

    logger.info(
        "web_ingestion_completed",
        topics=len(topics),
        urls=total_urls,
        failed=total_failed,
        chunks=len(all_chunks),
    )

    return {
        "chunks": all_chunks,
        "metadata": all_metadata,
        "topics_processed": len(topics),
        "total_urls": total_urls,
        "urls_failed": total_failed,
    }


def ingest_unified(
    store_name: str,
    web_topics: list[str] | None = None,
    local_paths: list[str] | list[Path] | None = None,
    max_urls_per_topic: int = 5,
    recursive: bool = True,
) -> dict[str, Any]:
    """Unified ingestion from web and local sources into single vector store.

    This is the main entry point for multi-source RAG ingestion. It:
    1. Ingests from web topics (if provided)
    2. Ingests from local files (if provided)
    3. Merges all chunks with source metadata
    4. Stores in unified FAISS vector store

    Args:
        store_name: Name for the unified vector store
        web_topics: Optional list of web search topics
        local_paths: Optional list of local file/directory paths
        max_urls_per_topic: Max URLs per web topic (default: 5)
        recursive: Recursively scan local directories (default: True)

    Returns:
        Comprehensive ingestion report:
        {
            "status": "completed",
            "store_name": "my_knowledge_base",
            "total_chunks": 150,
            "web_chunks": 80,
            "local_chunks": 70,
            "web_topics": 2,
            "local_files": 10,
            "storage_path": "./workspace/vector_stores/my_knowledge_base"
        }

    Example:
        >>> result = ingest_unified(
        ...     store_name="ml_knowledge",
        ...     web_topics=["machine learning basics"],
        ...     local_paths=["./docs/ml_papers"],
        ... )
        >>> print(f"Stored {result['total_chunks']} chunks from mixed sources")
    """
    logger.info(
        "starting_unified_ingestion",
        store_name=store_name,
        web_topics=web_topics,
        local_paths=local_paths,
    )

    all_chunks = []
    all_metadata = []

    # Ingest from web
    web_result = None
    if web_topics:
        web_result = ingest_from_web_unified(web_topics, max_urls_per_topic)
        all_chunks.extend(web_result["chunks"])
        all_metadata.extend(web_result["metadata"])

    # Ingest from local files
    local_result = None
    if local_paths:
        local_result = ingest_local_files(local_paths, recursive)
        all_chunks.extend(local_result["chunks"])
        all_metadata.extend(local_result["metadata"])

    # Validate we have content
    if not all_chunks:
        logger.error("no_content_ingested")
        return {
            "status": "failed",
            "error": "No content could be ingested from provided sources",
            "web_topics": web_topics,
            "local_paths": local_paths,
        }

    logger.info("content_merged", total_chunks=len(all_chunks))

    # Store in unified vector database
    try:
        vector_tool = LocalVectorSearch()
        result_json = vector_tool.store_documents(all_chunks, store_name, metadata=all_metadata)
        storage_result = json.loads(result_json)

        logger.info("unified_ingestion_completed", store_name=store_name, chunks=len(all_chunks))

        return {
            "status": "completed",
            "store_name": store_name,
            "total_chunks": storage_result["count"],
            "web_chunks": len(web_result["chunks"]) if web_result else 0,
            "local_chunks": len(local_result["chunks"]) if local_result else 0,
            "web_topics": len(web_topics) if web_topics else 0,
            "web_urls_processed": web_result["total_urls"] if web_result else 0,
            "local_files_processed": (local_result["files_processed"] if local_result else 0),
            "storage_path": storage_result["path"],
        }

    except Exception:
        logger.exception("unified_storage_failed")
        return {
            "status": "failed",
            "error": "Storage failed - see logs for details",
            "chunks_attempted": len(all_chunks),
        }
