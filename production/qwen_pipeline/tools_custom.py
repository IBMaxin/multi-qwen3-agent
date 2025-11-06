"""Custom tools for Qwen-Agent with local Ollama embeddings.

This module provides custom implementations of Qwen-Agent tools that use
local Ollama models instead of external API services (like DashScope).

Copyright: Based on Qwen-Agent patterns from QwenLM/Qwen-Agent
License: Apache License 2.0
"""

import json
from typing import Any, ClassVar

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from qwen_agent.tools.base import register_tool
from qwen_agent.tools.doc_parser import Record
from qwen_agent.tools.search_tools.base_search import BaseSearch

load_dotenv()


@register_tool("local_vector_search")
class LocalVectorSearch(BaseSearch):
    """Local vector search using Ollama qwen3-embedding model.

    This tool provides semantic search capabilities using locally-running
    Ollama embeddings instead of DashScope API. It's a drop-in replacement
    for the official 'vector_search' tool but runs 100% locally.

    Architecture:
    - Inherits from BaseSearch (official Qwen-Agent pattern)
    - Uses OllamaEmbeddings (langchain_community)
    - Builds FAISS vector store for similarity search
    - Returns scored chunks: (source, chunk_id, similarity_score)

    Configuration:
        In rag_cfg, use: 'rag_searchers': ['local_vector_search', 'keyword_search']

    Environment:
        - Requires Ollama running at http://localhost:11434
        - Requires qwen3-embedding model: `ollama pull qwen3-embedding:0.6b`

    Example:
        >>> from qwen_agent.agents import Assistant
        >>> model_name = os.getenv('MODEL_NAME', 'qwen3:4b-instruct-2507-q4_K_M')
        >>> bot = Assistant(
        ...     llm={'model': model_name, 'model_server': 'http://localhost:11434/v1'},
        ...     files=['document.pdf'],
        ...     rag_cfg={'rag_searchers': ['local_vector_search']}
        ... )
    """

    description = (
        "Semantic search over documents using local Ollama embeddings. "
        "Finds relevant text chunks based on meaning rather than exact keywords."
    )
    parameters: ClassVar[list[dict]] = [
        {
            "name": "query",
            "type": "string",
            "description": "Search query text",
            "required": True,
        }
    ]

    def __init__(self, cfg: dict | None = None):
        """Initialize local vector search tool.

        Args:
            cfg: Optional configuration dict with:
                - embedding_model: Ollama model name (default: 'qwen3-embedding:4b')
                - base_url: Ollama server URL (default: 'http://localhost:11434')
                - max_content_length: Max chars per chunk (default: 2000)
        """
        super().__init__(cfg)
        self.cfg = cfg or {}
        self.embedding_model = self.cfg.get("embedding_model", "qwen3-embedding:4b")
        self.base_url = self.cfg.get("base_url", "http://localhost:11434")
        self.max_content_length = self.cfg.get("max_content_length", 2000)

    def sort_by_scores(
        self, query: str, docs: list[Record], **kwargs: Any  # noqa: ARG002
    ) -> list[tuple[str, str, float]]:
        """Sort document chunks by semantic similarity to query.

        This method implements the core vector search logic using local
        Ollama embeddings. It mirrors the official VectorSearch implementation
        but replaces DashScopeEmbeddings with OllamaEmbeddings.

        Args:
            query: Search query text (may be JSON with 'text' field)
            docs: List of parsed document Records with chunks

        Returns:
            List of tuples: (source_file, chunk_id, similarity_score)
            Lower scores = more similar (FAISS distance metric)

        Raises:
            ConnectionError: If Ollama server not reachable
        """
        # Extract raw query text (handle JSON format)
        try:
            query_json = json.loads(query)
            # This assumes user input won't contain json str with 'text' attribute
            if "text" in query_json:
                query = query_json["text"]
        except json.decoder.JSONDecodeError:
            pass  # Query is already plain text

        # Flatten all chunks from all docs into LangChain Documents
        all_chunks = []
        for doc in docs:
            for chk in doc.raw:
                all_chunks.append(
                    Document(
                        page_content=chk.content[: self.max_content_length],
                        metadata=chk.metadata,
                    )
                )

        if not all_chunks:
            return []  # No content to search

        # Create Ollama embeddings (LOCAL - no external API!)
        embeddings = OllamaEmbeddings(model=self.embedding_model, base_url=self.base_url)

        # Build FAISS vector store from documents
        db = FAISS.from_documents(all_chunks, embeddings)

        # Perform similarity search with scores
        chunk_and_score = db.similarity_search_with_score(query, k=len(all_chunks))

        # Return in expected format: (source, chunk_id, score)
        return [
            (chk.metadata["source"], chk.metadata["chunk_id"], score)
            for chk, score in chunk_and_score
        ]
