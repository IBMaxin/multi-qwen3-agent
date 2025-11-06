"""Tests for custom tools (LocalVectorSearch).

This test module follows official Qwen-Agent testing patterns and ensures
100% local testing without requiring actual Ollama server connection.

Copyright: Apache License 2.0
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from qwen_agent.tools.doc_parser import Chunk, Record

from qwen_pipeline.tools_custom import LocalVectorSearch


@pytest.fixture
def mock_embeddings():
    """Mock OllamaEmbeddings to avoid requiring actual Ollama server."""
    with patch("qwen_pipeline.tools_custom.OllamaEmbeddings") as mock:
        yield mock


@pytest.fixture
def mock_faiss():
    """Mock FAISS vector store to avoid embedding dependencies."""
    with patch("qwen_pipeline.tools_custom.FAISS") as mock:
        yield mock


@pytest.fixture
def sample_records() -> list[Record]:
    """Create sample document records for testing."""
    chunk1 = Chunk(
        content="The Qwen model family includes Qwen3 and Qwen2.5.",
        metadata={"source": "doc1.pdf", "chunk_id": "chunk_0"},
        token=10,
    )
    chunk2 = Chunk(
        content="Multi-head attention is a key component of transformers.",
        metadata={"source": "doc1.pdf", "chunk_id": "chunk_1"},
        token=12,
    )
    chunk3 = Chunk(
        content="FAISS provides efficient similarity search.",
        metadata={"source": "doc2.pdf", "chunk_id": "chunk_0"},
        token=8,
    )

    record1 = Record(url="doc1.pdf", title="Qwen Documentation", raw=[chunk1, chunk2])
    record2 = Record(url="doc2.pdf", title="Vector Search", raw=[chunk3])

    return [record1, record2]


class TestLocalVectorSearchInit:
    """Test LocalVectorSearch initialization."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        tool = LocalVectorSearch()

        assert tool.embedding_model == "qwen3-embedding:4b"
        assert tool.base_url == "http://localhost:11434"
        assert tool.max_content_length == 2000

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        cfg = {
            "embedding_model": "qwen3-embedding:0.6b",
            "base_url": "http://custom-ollama:11434",
            "max_content_length": 1500,
        }
        tool = LocalVectorSearch(cfg=cfg)

        assert tool.embedding_model == "qwen3-embedding:0.6b"
        assert tool.base_url == "http://custom-ollama:11434"
        assert tool.max_content_length == 1500

    def test_init_partial_config(self):
        """Test initialization with partial custom configuration."""
        cfg = {"embedding_model": "custom-model"}
        tool = LocalVectorSearch(cfg=cfg)

        assert tool.embedding_model == "custom-model"
        assert tool.base_url == "http://localhost:11434"  # default
        assert tool.max_content_length == 2000  # default

    def test_tool_registered(self):
        """Test that tool is properly registered with Qwen-Agent."""
        from qwen_agent.tools.base import TOOL_REGISTRY

        assert "local_vector_search" in TOOL_REGISTRY
        assert TOOL_REGISTRY["local_vector_search"] == LocalVectorSearch


class TestLocalVectorSearchSortByScores:
    """Test sort_by_scores method (core vector search functionality)."""

    def test_sort_with_plain_text_query(self, mock_embeddings, mock_faiss, sample_records):
        """Test sorting with plain text query."""
        tool = LocalVectorSearch()

        # Mock FAISS to return scored results
        mock_db = MagicMock()
        mock_faiss.from_documents.return_value = mock_db

        # Simulate FAISS similarity search results (Document, score tuples)
        mock_doc1 = MagicMock(spec=Document)
        mock_doc1.metadata = {"source": "doc1.pdf", "chunk_id": "chunk_0"}
        mock_doc2 = MagicMock(spec=Document)
        mock_doc2.metadata = {"source": "doc1.pdf", "chunk_id": "chunk_1"}

        mock_db.similarity_search_with_score.return_value = [
            (mock_doc1, 0.15),  # Lower score = more similar
            (mock_doc2, 0.82),
        ]

        results = tool.sort_by_scores("Qwen models", sample_records)

        # Verify embeddings were created with correct config
        mock_embeddings.assert_called_once_with(
            model="qwen3-embedding:4b", base_url="http://localhost:11434"
        )

        # Verify FAISS was built from documents
        assert mock_faiss.from_documents.called
        call_args = mock_faiss.from_documents.call_args
        documents = call_args[0][0]
        assert len(documents) == 3  # 3 chunks total

        # Verify results format: (source, chunk_id, score)
        assert len(results) == 2
        assert results[0] == ("doc1.pdf", "chunk_0", 0.15)
        assert results[1] == ("doc1.pdf", "chunk_1", 0.82)

    def test_sort_with_json_query(self, mock_embeddings, mock_faiss, sample_records):
        """Test sorting with JSON-formatted query (text extraction)."""
        tool = LocalVectorSearch()

        mock_db = MagicMock()
        mock_faiss.from_documents.return_value = mock_db
        mock_db.similarity_search_with_score.return_value = []

        # JSON query format (as used by Qwen-Agent internally)
        json_query = '{"text": "multi-head attention", "keywords": ["attention"]}'

        tool.sort_by_scores(json_query, sample_records)

        # Verify query was extracted correctly
        call_args = mock_db.similarity_search_with_score.call_args
        actual_query = call_args[0][0]
        assert actual_query == "multi-head attention"

    def test_sort_with_empty_docs(self, mock_embeddings, mock_faiss):
        """Test sorting with empty document list."""
        tool = LocalVectorSearch()

        results = tool.sort_by_scores("test query", [])

        # Should return empty list without calling FAISS
        assert results == []
        mock_faiss.from_documents.assert_not_called()

    def test_sort_with_no_chunks(self, mock_embeddings, mock_faiss):
        """Test sorting when records have no chunks."""
        tool = LocalVectorSearch()
        empty_record = Record(url="empty.pdf", title="Empty", raw=[])

        results = tool.sort_by_scores("test query", [empty_record])

        assert results == []
        mock_faiss.from_documents.assert_not_called()

    def test_sort_truncates_long_content(self, mock_embeddings, mock_faiss):
        """Test that content is truncated to max_content_length."""
        cfg = {"max_content_length": 50}
        tool = LocalVectorSearch(cfg=cfg)

        long_content = "A" * 100  # 100 chars
        chunk = Chunk(
            content=long_content,
            metadata={"source": "doc.pdf", "chunk_id": "chunk_0"},
            token=10,
        )
        record = Record(url="doc.pdf", title="Test", raw=[chunk])

        mock_db = MagicMock()
        mock_faiss.from_documents.return_value = mock_db
        mock_db.similarity_search_with_score.return_value = []

        tool.sort_by_scores("test", [record])

        # Check that documents were truncated
        call_args = mock_faiss.from_documents.call_args
        documents = call_args[0][0]
        assert len(documents[0].page_content) == 50

    def test_sort_preserves_metadata(self, mock_embeddings, mock_faiss, sample_records):
        """Test that chunk metadata is preserved correctly."""
        tool = LocalVectorSearch()

        mock_db = MagicMock()
        mock_faiss.from_documents.return_value = mock_db
        mock_db.similarity_search_with_score.return_value = []

        tool.sort_by_scores("test", sample_records)

        # Verify metadata was passed to FAISS documents
        call_args = mock_faiss.from_documents.call_args
        documents = call_args[0][0]

        assert documents[0].metadata["source"] == "doc1.pdf"
        assert documents[0].metadata["chunk_id"] == "chunk_0"
        assert documents[1].metadata["source"] == "doc1.pdf"
        assert documents[1].metadata["chunk_id"] == "chunk_1"
        assert documents[2].metadata["source"] == "doc2.pdf"
        assert documents[2].metadata["chunk_id"] == "chunk_0"

    def test_sort_handles_kwargs(self, mock_embeddings, mock_faiss, sample_records):
        """Test that **kwargs are accepted (even if unused)."""
        tool = LocalVectorSearch()

        mock_db = MagicMock()
        mock_faiss.from_documents.return_value = mock_db
        mock_db.similarity_search_with_score.return_value = []

        # Should not raise error with extra kwargs
        tool.sort_by_scores("test", sample_records, extra_param="ignored", another=123)

        # Verify method still executed
        mock_faiss.from_documents.assert_called_once()


class TestLocalVectorSearchErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_json_query_fallback(self, mock_embeddings, mock_faiss, sample_records):
        """Test that invalid JSON query falls back to plain text."""
        tool = LocalVectorSearch()

        mock_db = MagicMock()
        mock_faiss.from_documents.return_value = mock_db
        mock_db.similarity_search_with_score.return_value = []

        # Malformed JSON should be treated as plain text
        invalid_json = '{"incomplete": "json'

        tool.sort_by_scores(invalid_json, sample_records)

        # Verify it used the raw query string
        call_args = mock_db.similarity_search_with_score.call_args
        actual_query = call_args[0][0]
        assert actual_query == invalid_json

    def test_embeddings_connection_error(self, mock_embeddings, mock_faiss, sample_records):
        """Test handling of Ollama connection errors."""
        tool = LocalVectorSearch()

        # Simulate connection error when building FAISS
        mock_faiss.from_documents.side_effect = ConnectionError("Cannot reach Ollama server")

        with pytest.raises(ConnectionError, match="Cannot reach Ollama server"):
            tool.sort_by_scores("test", sample_records)

    def test_faiss_build_error(self, mock_embeddings, mock_faiss, sample_records):
        """Test handling of FAISS build errors."""
        tool = LocalVectorSearch()

        # Simulate FAISS error
        mock_faiss.from_documents.side_effect = ValueError("Invalid embedding dimension")

        with pytest.raises(ValueError, match="Invalid embedding dimension"):
            tool.sort_by_scores("test", sample_records)


class TestLocalVectorSearchIntegration:
    """Integration-style tests (still mocked but more comprehensive)."""

    def test_full_workflow_with_multiple_docs(self, mock_embeddings, mock_faiss):
        """Test complete workflow with realistic multi-document scenario."""
        tool = LocalVectorSearch(cfg={"max_content_length": 1000})

        # Create realistic document structure
        chunks_doc1 = [
            Chunk(
                content=f"Document 1, Chunk {i}: Content about Qwen models.",
                metadata={"source": "qwen_docs.pdf", "chunk_id": f"chunk_{i}"},
                token=10,
            )
            for i in range(3)
        ]
        chunks_doc2 = [
            Chunk(
                content=f"Document 2, Chunk {i}: Content about vector search.",
                metadata={"source": "vector_search.pdf", "chunk_id": f"chunk_{i}"},
                token=10,
            )
            for i in range(2)
        ]

        records = [
            Record(url="qwen_docs.pdf", title="Qwen", raw=chunks_doc1),
            Record(url="vector_search.pdf", title="Search", raw=chunks_doc2),
        ]

        # Mock FAISS results
        mock_db = MagicMock()
        mock_faiss.from_documents.return_value = mock_db

        mock_results = [
            (
                MagicMock(
                    metadata={"source": "qwen_docs.pdf", "chunk_id": "chunk_1"}, spec=Document
                ),
                0.05,
            ),
            (
                MagicMock(
                    metadata={"source": "vector_search.pdf", "chunk_id": "chunk_0"},
                    spec=Document,
                ),
                0.15,
            ),
        ]
        mock_db.similarity_search_with_score.return_value = mock_results

        results = tool.sort_by_scores("Qwen models", records)

        # Verify correct number of documents processed
        call_args = mock_faiss.from_documents.call_args
        documents = call_args[0][0]
        assert len(documents) == 5  # 3 + 2 chunks

        # Verify results
        assert len(results) == 2
        assert results[0] == ("qwen_docs.pdf", "chunk_1", 0.05)
        assert results[1] == ("vector_search.pdf", "chunk_0", 0.15)

    def test_tool_metadata_attributes(self):
        """Test that tool has required Qwen-Agent metadata."""
        tool = LocalVectorSearch()

        # Verify description exists
        assert hasattr(tool, "description")
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

        # Verify parameters schema
        assert hasattr(tool, "parameters")
        assert isinstance(tool.parameters, list)
        assert len(tool.parameters) > 0

        # Check parameter structure
        param = tool.parameters[0]
        assert "name" in param
        assert "type" in param
        assert "description" in param
        assert param["name"] == "query"
        assert param["type"] == "string"
        assert param["required"] is True
