"""Query helper functions for vector store operations.

Provides high-level query interfaces for vector stores with
result formatting, filtering, and source attribution.

Copyright: Based on Qwen-Agent patterns from QwenLM/Qwen-Agent
License: Apache License 2.0
"""

import json
from typing import Any

import structlog

from qwen_pipeline.tools_custom import LocalVectorSearch

structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()


def query_vector_store(
    store_name: str,
    query: str,
    k: int = 5,
    min_score: float | None = None,
    source_type_filter: str | None = None,
) -> dict[str, Any]:
    """Query a vector store with optional filtering and scoring.

    Performs semantic similarity search on a FAISS vector store with
    optional filters for relevance score and source type (web vs local).

    Args:
        store_name: Name of the vector store to query
        query: Search query text
        k: Number of results to return (default: 5)
        min_score: Optional minimum similarity score threshold (lower is better)
        source_type_filter: Optional filter by source type ("web" or "local_file")

    Returns:
        Dictionary containing query results and metadata:
        {
            "query": "search text",
            "store_name": "my_store",
            "total_results": 5,
            "filtered_results": 3,
            "results": [
                {
                    "content": "text chunk",
                    "metadata": {...},
                    "score": 0.234,
                    "source_type": "web"
                },
                ...
            ]
        }

    Raises:
        FileNotFoundError: If the vector store does not exist
        ValueError: If invalid source_type_filter provided

    Example:
        >>> result = query_vector_store("ml_kb", "What is RAG?", k=3)
        >>> print(f"Found {result['total_results']} results")
        >>> for r in result['results']:
        ...     print(f"Score: {r['score']:.4f}, Source: {r['source_type']}")
    """
    logger.info(
        "vector_store_query_started",
        store_name=store_name,
        query=query[:50],
        k=k,
        min_score=min_score,
        source_filter=source_type_filter,
    )

    # Validate source type filter
    if source_type_filter and source_type_filter not in {"web", "local_file"}:
        logger.error("invalid_source_filter", filter=source_type_filter)
        msg = f"Invalid source_type_filter: {source_type_filter}. Must be 'web' or 'local_file'"
        raise ValueError(msg)

    # Initialize vector search tool
    vector_tool = LocalVectorSearch()

    try:
        # Perform query
        results_json = vector_tool.query_store(store_name, query, k=k)
        raw_results: list[dict] = json.loads(results_json)

        if not raw_results:
            logger.warning("no_results_found", store_name=store_name, query=query[:50])
            return {
                "query": query,
                "store_name": store_name,
                "total_results": 0,
                "filtered_results": 0,
                "results": [],
            }

        # Apply filters
        filtered_results = raw_results

        # Filter by score threshold
        if min_score is not None:
            filtered_results = [r for r in filtered_results if r["score"] <= min_score]
            logger.debug(
                "score_filter_applied",
                threshold=min_score,
                before=len(raw_results),
                after=len(filtered_results),
            )

        # Filter by source type
        if source_type_filter:
            filtered_results = [
                r
                for r in filtered_results
                if r.get("metadata", {}).get("source_type") == source_type_filter
            ]
            logger.debug(
                "source_filter_applied",
                source_type=source_type_filter,
                before=len(raw_results),
                after=len(filtered_results),
            )

        # Enhance results with source_type at top level
        enhanced_results = []
        for result in filtered_results:
            enhanced_result = {
                "content": result["content"],
                "metadata": result["metadata"],
                "score": result["score"],
                "source_type": result.get("metadata", {}).get("source_type", "unknown"),
            }
            enhanced_results.append(enhanced_result)

        logger.info(
            "vector_store_query_completed",
            store_name=store_name,
            total_results=len(raw_results),
            filtered_results=len(enhanced_results),
        )

        return {
            "query": query,
            "store_name": store_name,
            "total_results": len(raw_results),
            "filtered_results": len(enhanced_results),
            "results": enhanced_results,
        }

    except FileNotFoundError:
        logger.exception("vector_store_not_found", store_name=store_name)
        raise
    except Exception:
        logger.exception("vector_store_query_failed", store_name=store_name)
        raise


def query_with_context(
    store_name: str,
    query: str,
    k: int = 5,
    max_context_length: int = 2000,
) -> dict[str, Any]:
    """Query vector store and format results for LLM context.

    Performs semantic search and formats results into a concatenated
    context string suitable for passing to an LLM, with source citations.

    Args:
        store_name: Name of the vector store to query
        query: Search query text
        k: Number of results to retrieve (default: 5)
        max_context_length: Maximum total character length for context (default: 2000)

    Returns:
        Dictionary with formatted context:
        {
            "query": "search text",
            "context": "Combined text from all results with citations",
            "sources": ["source1", "source2", ...],
            "num_chunks": 3
        }

    Example:
        >>> result = query_with_context("ml_kb", "Explain backpropagation", k=3)
        >>> print(result["context"])
        [Source: https://example.com/ml]
        Backpropagation is a method...

        [Source: /path/to/doc.md]
        The algorithm works by...
    """
    logger.info(
        "context_query_started",
        store_name=store_name,
        query=query[:50],
        k=k,
        max_length=max_context_length,
    )

    # Query vector store
    query_result = query_vector_store(store_name, query, k=k)

    if not query_result["results"]:
        logger.warning("no_context_available", store_name=store_name)
        return {
            "query": query,
            "context": "",
            "sources": [],
            "num_chunks": 0,
        }

    # Build context string with citations
    context_parts = []
    sources = []
    current_length = 0

    for result in query_result["results"]:
        source = result["metadata"].get("source", "Unknown")
        content = result["content"]

        # Build citation
        citation = f"\n[Source: {source}]\n{content}\n"

        # Check if adding this would exceed max length
        if current_length + len(citation) > max_context_length:
            logger.debug(
                "context_length_limit_reached",
                current_length=current_length,
                max_length=max_context_length,
            )
            break

        context_parts.append(citation)
        sources.append(source)
        current_length += len(citation)

    combined_context = "\n".join(context_parts)

    logger.info(
        "context_query_completed",
        num_chunks=len(context_parts),
        context_length=len(combined_context),
        sources_count=len(sources),
    )

    return {
        "query": query,
        "context": combined_context,
        "sources": sources,
        "num_chunks": len(context_parts),
    }
