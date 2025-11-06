"""Custom tools for Qwen-Agent with local Ollama embeddings.

This module provides custom implementations of Qwen-Agent tools that use
local Ollama models instead of external API services (like DashScope).

Copyright: Based on Qwen-Agent patterns from QwenLM/Qwen-Agent
License: Apache License 2.0
"""

import json
import os
from pathlib import Path
from typing import Any, ClassVar

import structlog
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from qwen_agent.tools.base import register_tool
from qwen_agent.tools.doc_parser import Record
from qwen_agent.tools.search_tools.base_search import BaseSearch
from qwen_agent.utils.tokenization_qwen import count_tokens

load_dotenv()

structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()


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
        # Prefer explicit cfg, then environment, then sensible defaults
        self.embedding_model = self.cfg.get(
            "embedding_model",
            os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b"),
        )
        self.base_url = self.cfg.get(
            "base_url",
            os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434"),
        )
        self.max_content_length = self.cfg.get("max_content_length", 2000)
        self.vector_store_path = self.cfg.get(
            "vector_store_path",
            os.getenv("VECTOR_STORE_PATH", "./workspace/vector_stores"),
        )
        # Ensure vector store directory exists
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)

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

    def store_documents(
        self, texts: list[str], store_name: str, metadata: list[dict] | None = None
    ) -> str:
        """Embed and persist documents to disk using FAISS.

        This method creates a new vector store from text documents, embeds them
        using local Ollama embeddings, and saves the FAISS index to disk for
        later retrieval. Follows official langchain_community patterns.

        Args:
            texts: List of text strings to embed and store
            store_name: Unique name for this vector store (becomes subdirectory)
            metadata: Optional list of metadata dicts (one per text)

        Returns:
            JSON string with status and storage path

        Raises:
            ValueError: If texts is empty or store_name is invalid
            ConnectionError: If Ollama server not reachable

        Example:
            >>> tool = LocalVectorSearch()
            >>> docs = ["Machine learning is...", "Deep learning uses..."]
            >>> result = tool.store_documents(docs, "ml_basics")
            >>> print(result)
            {"status": "saved", "path": "./workspace/vector_stores/ml_basics", "count": 2}
        """
        if not texts:
            msg = "Cannot store empty document list"
            raise ValueError(msg)

        # Sanitize store_name (remove invalid path characters)
        store_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in store_name)

        if not store_name:
            msg = "Store name must contain valid characters"
            raise ValueError(msg)

        logger.info("storing_documents", store_name=store_name, count=len(texts))

        # Create embeddings instance
        embeddings = OllamaEmbeddings(model=self.embedding_model, base_url=self.base_url)

        # Convert texts to Documents with optional metadata
        if metadata:
            if len(metadata) != len(texts):
                msg = "Metadata list length must match texts list length"
                raise ValueError(msg)
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadata, strict=True)
            ]
        else:
            documents = [
                Document(page_content=text, metadata={"index": i}) for i, text in enumerate(texts)
            ]

        # Build FAISS vector store
        db = FAISS.from_documents(documents, embeddings)

        # Save to disk
        save_path = Path(self.vector_store_path) / store_name
        save_path.mkdir(parents=True, exist_ok=True)
        db.save_local(str(save_path))

        logger.info("documents_stored", path=save_path, count=len(texts))

        return json.dumps(
            {"status": "saved", "path": str(save_path), "count": len(texts)}, ensure_ascii=False
        )

    def load_store(self, store_name: str) -> FAISS:
        """Load a previously saved vector store from disk.

        This method loads a FAISS index that was saved using store_documents().
        The embeddings model must match the one used during storage.

        Args:
            store_name: Name of the vector store to load (subdirectory name)

        Returns:
            FAISS vector store instance ready for querying

        Raises:
            FileNotFoundError: If store does not exist
            ValueError: If store_name is invalid

        Example:
            >>> tool = LocalVectorSearch()
            >>> db = tool.load_store("ml_basics")
            >>> results = db.similarity_search("what is machine learning", k=3)
        """
        if not store_name:
            msg = "Store name cannot be empty"
            raise ValueError(msg)

        load_path = Path(self.vector_store_path) / store_name

        if not load_path.exists():
            msg = f"Vector store not found: {load_path}"
            raise FileNotFoundError(msg)

        logger.info("loading_vector_store", path=load_path)

        # Create embeddings instance (must match the one used for storage)
        embeddings = OllamaEmbeddings(model=self.embedding_model, base_url=self.base_url)

        # Load FAISS index
        db = FAISS.load_local(
            str(load_path),
            embeddings,
            allow_dangerous_deserialization=True,  # We trust our own data
        )

        logger.info("vector_store_loaded", path=load_path)

        return db

    def query_store(self, store_name: str, query: str, k: int = 5) -> str:
        """Query a stored vector database and return top-k results.

        Convenience method that loads a store and performs similarity search.
        Returns results in JSON format for easy agent consumption.

        Args:
            store_name: Name of the vector store to query
            query: Search query text
            k: Number of results to return (default: 5)

        Returns:
            JSON string with search results (empty list if store not found)

        Example:
            >>> tool = LocalVectorSearch()
            >>> results = tool.query_store("ml_basics", "what is deep learning", k=3)
            >>> print(results)
            [{"content": "Deep learning uses...", "metadata": {...}, "score": 0.23}]
        """
        try:
            db = self.load_store(store_name)
        except FileNotFoundError:
            logger.exception("store_not_found", store_name=store_name)
            return json.dumps([], ensure_ascii=False)

        # Perform similarity search with scores
        results_with_scores = db.similarity_search_with_score(query, k=k)

        # Format results
        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),  # Convert numpy float to Python float
            }
            for doc, score in results_with_scores
        ]

        return json.dumps(formatted_results, ensure_ascii=False)

    def list_stores(self) -> str:
        """List all available vector stores.

        Returns:
            JSON string with list of store names

        Example:
            >>> tool = LocalVectorSearch()
            >>> stores = tool.list_stores()
            >>> print(stores)  # ["ml_basics", "python_docs", "research_papers"]
        """
        store_path = Path(self.vector_store_path)

        if not store_path.exists():
            return json.dumps([])

        # Get all subdirectories (each is a store)
        stores = [
            d.name for d in store_path.iterdir() if d.is_dir() and (d / "index.faiss").exists()
        ]

        return json.dumps(sorted(stores))

    @staticmethod
    def chunk_text(text: str, max_chunk_tokens: int = 500, overlap_tokens: int = 50) -> list[str]:
        """Smart text chunking with overlap for better context preservation.

        Splits text into chunks based on token count, preserving paragraph
        boundaries when possible. Adds overlap between chunks to maintain context.

        Args:
            text: Text to chunk
            max_chunk_tokens: Maximum tokens per chunk (default: 500)
            overlap_tokens: Tokens to overlap between chunks (default: 50)

        Returns:
            List of text chunks

        Example:
            >>> text = "Long document content..."
            >>> chunks = LocalVectorSearch.chunk_text(text, max_chunk_tokens=500)
        """
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)

            # If single paragraph exceeds max, split it
            if para_tokens > max_chunk_tokens:
                # Save current chunk if any
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split long paragraph by sentences
                sentences = para.split(". ")
                for sentence in sentences:
                    sent_tokens = count_tokens(sentence)
                    if current_tokens + sent_tokens > max_chunk_tokens and current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        # Keep last sentence for overlap
                        if overlap_tokens > 0 and current_chunk:
                            current_chunk = [current_chunk[-1]]
                            current_tokens = count_tokens(current_chunk[0])
                        else:
                            current_chunk = []
                            current_tokens = 0
                    current_chunk.append(sentence)
                    current_tokens += sent_tokens
                continue

            # Normal paragraph processing
            if current_tokens + para_tokens > max_chunk_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Keep last paragraph for overlap
                if overlap_tokens > 0 and current_chunk:
                    current_chunk = [current_chunk[-1]]
                    current_tokens = count_tokens(current_chunk[0])
                else:
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(para)
            current_tokens += para_tokens

        # Add remaining chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def query_with_source_attribution(
        self, store_name: str, query: str, k: int = 5
    ) -> dict[str, Any]:
        """Query vector store and group results by source type.

        Enhanced query method that provides source attribution, grouping
        results by web vs local sources for better context understanding.

        Args:
            store_name: Name of the vector store to query
            query: Search query text
            k: Number of results to return (default: 5)

        Returns:
            Dictionary with grouped results:
            {
                "query": "search query",
                "total_results": 5,
                "web_results": [...],
                "local_results": [...],
                "all_results": [...]
            }

        Example:
            >>> tool = LocalVectorSearch()
            >>> result = tool.query_with_source_attribution("knowledge_base", "ML")
            >>> print(f"Found {result['total_results']} results")
            >>> print(f"  - {len(result['web_results'])} from web")
            >>> print(f"  - {len(result['local_results'])} from local files")
        """
        results_json = self.query_store(store_name, query, k=k)
        results: list[dict] = json.loads(results_json)

        # Group by source type
        web_results = [r for r in results if r.get("metadata", {}).get("source_type") == "web"]
        local_results = [
            r for r in results if r.get("metadata", {}).get("source_type") == "local_file"
        ]

        return {
            "query": query,
            "total_results": len(results),
            "web_results": web_results,
            "local_results": local_results,
            "all_results": results,
        }

    def get_store_statistics(self, store_name: str) -> dict[str, Any]:
        """Get statistics about a vector store's content.

        Analyzes a stored vector database to provide insights about
        source distribution, content types, and ingestion metadata.

        Args:
            store_name: Name of the vector store to analyze

        Returns:
            Dictionary with statistics:
            {
                "total_documents": 100,
                "web_documents": 60,
                "local_documents": 40,
                "file_types": {"markdown": 25, "pdf": 15},
                "topics": ["ML", "Python"],
                "store_path": "./workspace/vector_stores/..."
            }

        Example:
            >>> tool = LocalVectorSearch()
            >>> stats = tool.get_store_statistics("my_knowledge")
            >>> print(f"Total docs: {stats['total_documents']}")
        """
        try:
            db = self.load_store(store_name)
        except FileNotFoundError:
            return {"error": "Store not found", "store_name": store_name}

        # Get all documents from the vector store
        # FAISS doesn't provide direct access to docs, so we use a broad query
        all_docs_with_scores = db.similarity_search_with_score("", k=10000)

        web_count = 0
        local_count = 0
        file_types: dict[str, int] = {}
        topics: set[str] = set()

        for doc, _score in all_docs_with_scores:
            metadata = doc.metadata
            source_type = metadata.get("source_type", "unknown")

            if source_type == "web":
                web_count += 1
                if "topic" in metadata:
                    topics.add(metadata["topic"])
            elif source_type == "local_file":
                local_count += 1
                file_type = metadata.get("file_type", "unknown")
                file_types[file_type] = file_types.get(file_type, 0) + 1

        store_path = Path(self.vector_store_path) / store_name

        return {
            "total_documents": len(all_docs_with_scores),
            "web_documents": web_count,
            "local_documents": local_count,
            "file_types": file_types,
            "topics": sorted(topics),
            "store_path": str(store_path),
        }
