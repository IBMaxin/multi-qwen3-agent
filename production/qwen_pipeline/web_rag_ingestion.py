"""Web-to-Vector RAG Ingestion Pipeline.

This module provides automated workflow for ingesting web content into
a persistent vector database:
1. Search web for a topic (web_search tool)
2. Extract content from top URLs (web_extractor tool)
3. Chunk and embed text (LocalVectorSearch)
4. Store vectors to disk (FAISS persistence)
5. Make retrievable for RAG queries

Architecture:
- Uses ReActChat for autonomous tool chaining
- Implements retry logic (2 attempts per URL)
- Smart chunking with overlap for context preservation
- Topic-based storage organization

Copyright: Based on Qwen-Agent patterns from QwenLM/Qwen-Agent
License: Apache License 2.0
"""

import json
import re
import time
from typing import Any

import structlog
from qwen_agent.agents import ReActChat

from qwen_pipeline.config import get_llm_config
from qwen_pipeline.tools_custom import LocalVectorSearch

structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()

# Configuration for ingestion workflow
INGESTION_CONFIG = {
    "max_chunk_tokens": 500,  # Aligns with DEFAULT_PARSER_PAGE_SIZE
    "chunk_overlap": 50,  # Preserve context between chunks
    "min_content_length": 100,  # Skip very short extracts (likely noise)
    "max_urls_per_search": 5,  # Limit to top 5 search results
    "retry_failed_urls": 2,  # Retry failed URLs 2 times then skip
    "retry_delay": 1.0,  # Seconds to wait between retries
}


def create_ingestion_agent() -> ReActChat:
    """Create ReActChat agent with web tools and vector storage.

    Returns:
        Configured ReActChat agent with web_search, web_extractor,
        and LocalVectorSearch tools.

    Example:
        >>> agent = create_ingestion_agent()
        >>> result = ingest_from_web(agent, "machine learning basics")
    """
    llm_cfg = get_llm_config()

    # Initialize tools
    vector_tool = LocalVectorSearch()

    # Tool list with built-in and custom tools
    tools = [
        "web_search",  # Requires SERPER_API_KEY
        "web_extractor",  # No API key needed
        vector_tool,  # Custom tool with persistence
    ]

    system_message = """You are an expert research assistant that ingests web \
content into a vector database.

Your workflow:
1. Use web_search to find high-quality sources about the topic
2. Extract content from the top URLs using web_extractor
3. Clean and validate the extracted content
4. Store the content in the vector database

Guidelines:
- Focus on authoritative sources (documentation, research papers, reputable sites)
- Skip URLs that return errors or insufficient content
- Provide clear status updates about what you're doing
- Return structured results with source attribution
"""

    agent = ReActChat(
        llm=llm_cfg,
        function_list=tools,
        system_message=system_message,
        name="WebRAGIngestionAgent",
    )

    logger.info(
        "ingestion_agent_created",
        tools=["web_search", "web_extractor", "local_vector_search"],
    )

    return agent


def extract_url_with_retry(
    agent: ReActChat, url: str, max_retries: int = 2, delay: float = 1.0
) -> str | None:
    """Extract content from URL with retry logic.

    Args:
        agent: ReActChat agent with web_extractor tool
        url: URL to extract
        max_retries: Number of retry attempts (default: 2)
        delay: Seconds to wait between retries (default: 1.0)

    Returns:
        Extracted content as string, or None if all attempts failed

    Example:
        >>> agent = create_ingestion_agent()
        >>> content = extract_url_with_retry(agent, "https://example.com")
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            logger.info("extracting_url", url=url, attempt=attempt + 1)

            # Ask agent to extract content
            messages = [
                {
                    "role": "user",
                    "content": f"Extract the main content from this URL: {url}",
                }
            ]

            responses = list(agent.run(messages=messages))

            # Extract text from response
            if responses:
                last_response = responses[-1]
                if isinstance(last_response, dict) and "content" in last_response:
                    content = last_response["content"]
                    min_length = INGESTION_CONFIG["min_content_length"]
                    if isinstance(content, str) and len(content) >= min_length:
                        logger.info("url_extracted", url=url, length=len(content))
                        return content

            logger.warning("insufficient_content", url=url, attempt=attempt + 1)

        except Exception:
            logger.exception("extraction_failed", url=url, attempt=attempt + 1)

        # Wait before retry (except on last attempt)
        if attempt < max_retries:
            time.sleep(delay)

    logger.error("url_extraction_failed_all_attempts", url=url, max_retries=max_retries)
    return None


def ingest_from_web(
    topic: str, store_name: str | None = None, max_urls: int | None = None
) -> dict[str, Any]:
    """Ingest web content about a topic into persistent vector database.

    Complete workflow:
    1. Search web for topic
    2. Extract content from top URLs (with retry logic)
    3. Chunk content with smart overlap
    4. Embed and store in FAISS
    5. Return ingestion summary

    Args:
        topic: Search query topic (e.g., "machine learning basics")
        store_name: Optional custom store name (default: sanitized topic)
        max_urls: Maximum URLs to process (default: from INGESTION_CONFIG)

    Returns:
        Dictionary with ingestion results:
        {
            "status": "completed",
            "topic": "machine learning basics",
            "store_name": "machine_learning_basics",
            "urls_processed": 5,
            "urls_failed": 1,
            "chunks_stored": 42,
            "storage_path": "./workspace/vector_stores/machine_learning_basics"
        }

    Example:
        >>> result = ingest_from_web("Python type hints tutorial")
        >>> print(f"Stored {result['chunks_stored']} chunks")
    """
    # Initialize
    agent = create_ingestion_agent()
    vector_tool = LocalVectorSearch()
    # Handle mixed types in config dict
    configured_max = INGESTION_CONFIG["max_urls_per_search"]
    max_urls_value = max_urls if max_urls is not None else int(configured_max)

    # Sanitize store_name
    if not store_name:
        store_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in topic.lower())

    logger.info("starting_ingestion", topic=topic, store_name=store_name, max_urls=max_urls_value)

    # Step 1: Search web
    logger.info("searching_web", topic=topic)
    search_messages = [{"role": "user", "content": f"Search the web for: {topic}"}]

    search_responses = list(agent.run(messages=search_messages))

    # Parse search results to extract URLs
    urls = []
    for response in search_responses:
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
            # Extract URLs from response (simple heuristic)
            if "http" in content:
                # Try to extract URLs from markdown links or plain text
                url_pattern = r"https?://[^\s\)\]]+"
                found_urls = re.findall(url_pattern, content)
                urls.extend(found_urls[:max_urls])
                break

    if not urls:
        logger.error("no_urls_found", topic=topic)
        return {
            "status": "failed",
            "error": "No URLs found in search results",
            "topic": topic,
        }

    logger.info("urls_found", count=len(urls), urls=urls[:3])  # Log first 3

    # Step 2: Extract content from URLs with retry
    extracted_contents = []
    failed_urls = 0

    for url in urls[:max_urls_value]:
        content = extract_url_with_retry(
            agent,
            url,
            max_retries=int(INGESTION_CONFIG["retry_failed_urls"]),
            delay=INGESTION_CONFIG["retry_delay"],
        )

        if content:
            extracted_contents.append({"url": url, "content": content})
        else:
            failed_urls += 1

    if not extracted_contents:
        logger.error("all_extractions_failed", topic=topic)
        return {
            "status": "failed",
            "error": "All URL extractions failed",
            "topic": topic,
            "urls_failed": failed_urls,
        }

    logger.info("content_extracted", successful=len(extracted_contents), failed=failed_urls)

    # Step 3: Chunk all content
    all_chunks = []
    all_metadata = []

    for item in extracted_contents:
        chunks = LocalVectorSearch.chunk_text(
            item["content"],
            max_chunk_tokens=INGESTION_CONFIG["max_chunk_tokens"],
            overlap_tokens=INGESTION_CONFIG["chunk_overlap"],
        )

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({"source": item["url"], "chunk_index": i, "topic": topic})

    logger.info("content_chunked", total_chunks=len(all_chunks))

    # Step 4: Store in vector database
    try:
        result_json = vector_tool.store_documents(all_chunks, store_name, metadata=all_metadata)
        result = json.loads(result_json)

        logger.info("ingestion_completed", store_name=store_name, chunks=len(all_chunks))

        return {
            "status": "completed",
            "topic": topic,
            "store_name": store_name,
            "urls_processed": len(extracted_contents),
            "urls_failed": failed_urls,
            "chunks_stored": result["count"],
            "storage_path": result["path"],
        }

    except Exception:
        logger.exception("storage_failed")
        return {
            "status": "failed",
            "error": "Storage failed - see logs for details",
            "topic": topic,
            "urls_processed": len(extracted_contents),
            "urls_failed": failed_urls,
        }


def query_ingested_store(store_name: str, query: str, k: int = 5) -> list[dict]:
    """Query a previously ingested vector store.

    Args:
        store_name: Name of the vector store to query
        query: Search query
        k: Number of results to return (default: 5)

    Returns:
        List of result dicts with content, metadata, and scores

    Example:
        >>> results = query_ingested_store("python_type_hints", "how to use Optional")
        >>> for res in results:
        >>>     print(f"Score: {res['score']}, Source: {res['metadata']['source']}")
    """
    vector_tool = LocalVectorSearch()

    try:
        results_json = vector_tool.query_store(store_name, query, k=k)
        results: list[dict] = json.loads(results_json)
    except FileNotFoundError:
        logger.exception("store_not_found", store_name=store_name)
        return []
    else:
        logger.info("query_completed", store_name=store_name, results=len(results))
        return results


def list_ingested_stores() -> list[str]:
    """List all available ingested vector stores.

    Returns:
        List of store names

    Example:
        >>> stores = list_ingested_stores()
        >>> print(f"Available stores: {', '.join(stores)}")
    """
    vector_tool = LocalVectorSearch()
    stores_json = vector_tool.list_stores()
    stores: list[str] = json.loads(stores_json)
    return stores
