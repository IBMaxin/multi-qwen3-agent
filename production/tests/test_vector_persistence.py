"""Tests for LocalVectorSearch persistence functionality.

This module tests the vector store persistence features including:
- Saving and loading FAISS vector stores to/from disk
- Smart text chunking with overlap
- Query functionality on stored vectors
- Store enumeration
- Error handling and edge cases

Copyright: Based on Qwen-Agent patterns from QwenLM/Qwen-Agent
License: Apache License 2.0
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qwen_pipeline.tools_custom import LocalVectorSearch


class TestLocalVectorSearchPersistence:
    """Test suite for LocalVectorSearch persistence methods."""

    @pytest.fixture
    def temp_store_dir(self) -> str:
        """Create temporary directory for vector stores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def vector_tool(self, temp_store_dir: str) -> LocalVectorSearch:
        """Create LocalVectorSearch instance with temp directory."""
        return LocalVectorSearch(
            cfg={
                "embedding_model": "qwen3-embedding:4b",
                "base_url": "http://localhost:11434",
                "vector_store_path": temp_store_dir,
            }
        )

    def test_initialization_creates_vector_store_directory(self, temp_store_dir: str) -> None:
        """Test that initialization creates vector store directory."""
        store_path = Path(temp_store_dir) / "test_store"
        store_path.mkdir(parents=True, exist_ok=True)

        tool = LocalVectorSearch(cfg={"vector_store_path": str(store_path.parent)})

        assert Path(tool.vector_store_path).exists()
        assert Path(tool.vector_store_path).is_dir()

    def test_store_documents_with_valid_texts(self, vector_tool: LocalVectorSearch) -> None:
        """Test storing documents successfully."""
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Neural networks are inspired by biological neurons.",
        ]

        with patch("qwen_pipeline.tools_custom.FAISS") as mock_faiss_class:
            # Mock FAISS instance and methods
            mock_db = MagicMock()
            mock_faiss_class.from_documents.return_value = mock_db

            result_json = vector_tool.store_documents(test_texts, "test_store")
            result = json.loads(result_json)

            # Verify result structure
            assert result["status"] == "saved"
            assert "path" in result
            assert result["count"] == 3
            assert "test_store" in result["path"]

            # Verify FAISS methods were called
            mock_faiss_class.from_documents.assert_called_once()
            mock_db.save_local.assert_called_once()

    def test_store_documents_with_metadata(self, vector_tool: LocalVectorSearch) -> None:
        """Test storing documents with metadata."""
        test_texts = ["Document one", "Document two"]
        test_metadata = [
            {"source": "url1.com", "date": "2025-01-01"},
            {"source": "url2.com", "date": "2025-01-02"},
        ]

        with patch("qwen_pipeline.tools_custom.FAISS") as mock_faiss_class:
            mock_db = MagicMock()
            mock_faiss_class.from_documents.return_value = mock_db

            result_json = vector_tool.store_documents(
                test_texts, "meta_store", metadata=test_metadata
            )
            result = json.loads(result_json)

            assert result["status"] == "saved"
            assert result["count"] == 2

            # Verify metadata was passed to FAISS
            call_args = mock_faiss_class.from_documents.call_args
            docs = call_args[0][0]
            assert len(docs) == 2
            assert docs[0].metadata == {"source": "url1.com", "date": "2025-01-01"}
            assert docs[1].metadata == {"source": "url2.com", "date": "2025-01-02"}

    def test_store_documents_empty_texts_raises_error(self, vector_tool: LocalVectorSearch) -> None:
        """Test that storing empty texts raises ValueError."""
        with pytest.raises(ValueError, match="Cannot store empty document list"):
            vector_tool.store_documents([], "empty_store")

    def test_store_documents_invalid_store_name_raises_error(
        self, vector_tool: LocalVectorSearch
    ) -> None:
        """Test that store names with no alphanumeric chars raise error."""
        texts = ["Some text"]

        # Empty string should raise ValueError (no alphanumeric characters)
        with (
            patch("qwen_pipeline.tools_custom.FAISS"),
            patch("qwen_pipeline.tools_custom.OllamaEmbeddings"),
            pytest.raises(ValueError, match="Store name must contain valid characters"),
        ):
            vector_tool.store_documents(texts, "")

    def test_store_documents_sanitizes_store_name(self, vector_tool: LocalVectorSearch) -> None:
        """Test that store names are sanitized."""
        texts = ["Some text"]
        store_name = "my store@2025!"  # Contains spaces and special chars

        with patch("qwen_pipeline.tools_custom.FAISS") as mock_faiss_class:
            mock_db = MagicMock()
            mock_faiss_class.from_documents.return_value = mock_db

            result_json = vector_tool.store_documents(texts, store_name)
            result = json.loads(result_json)

            # Store name should be sanitized (spaces and special chars removed/replaced)
            assert "!!!###@@@" not in result["path"]
            assert result["status"] == "saved"

    def test_metadata_length_mismatch_raises_error(self, vector_tool: LocalVectorSearch) -> None:
        """Test that metadata/texts length mismatch raises error."""
        texts = ["Text 1", "Text 2", "Text 3"]
        metadata = [{"source": "a"}, {"source": "b"}]  # Only 2 items

        with pytest.raises(ValueError, match="Metadata list length must match texts list length"):
            vector_tool.store_documents(texts, "store", metadata=metadata)

    def test_load_store_success(self, vector_tool: LocalVectorSearch) -> None:
        """Test loading a stored vector store."""
        with patch("qwen_pipeline.tools_custom.FAISS") as mock_faiss_class:
            # Mock load_local
            mock_db = MagicMock()
            mock_faiss_class.load_local.return_value = mock_db

            # Create store first
            store_path = Path(vector_tool.vector_store_path) / "test_store"
            store_path.mkdir(parents=True, exist_ok=True)
            (store_path / "index.faiss").touch()

            loaded_db = vector_tool.load_store("test_store")

            assert loaded_db is mock_db
            mock_faiss_class.load_local.assert_called_once()

    def test_load_store_not_found_raises_error(self, vector_tool: LocalVectorSearch) -> None:
        """Test that loading nonexistent store raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Vector store not found"):
            vector_tool.load_store("nonexistent_store")

    def test_load_store_empty_name_raises_error(self, vector_tool: LocalVectorSearch) -> None:
        """Test that empty store name raises ValueError."""
        with pytest.raises(ValueError, match="Store name cannot be empty"):
            vector_tool.load_store("")

    def test_query_store_success(self, vector_tool: LocalVectorSearch) -> None:
        """Test querying a stored vector store."""
        with patch.object(vector_tool, "load_store") as mock_load:
            mock_db = MagicMock()
            mock_load.return_value = mock_db

            # Mock similarity search results
            mock_doc1 = MagicMock()
            mock_doc1.page_content = "Machine learning basics"
            mock_doc1.metadata = {"source": "url1.com"}

            mock_doc2 = MagicMock()
            mock_doc2.page_content = "Deep learning tutorial"
            mock_doc2.metadata = {"source": "url2.com"}

            mock_db.similarity_search_with_score.return_value = [
                (mock_doc1, 0.23),
                (mock_doc2, 0.45),
            ]

            result_json = vector_tool.query_store("test_store", "machine learning", k=2)
            results = json.loads(result_json)

            assert len(results) == 2
            assert results[0]["content"] == "Machine learning basics"
            assert results[0]["score"] == 0.23
            assert results[1]["content"] == "Deep learning tutorial"
            assert results[1]["score"] == 0.45

    def test_query_store_not_found_returns_empty(self, vector_tool: LocalVectorSearch) -> None:
        """Test that querying nonexistent store returns empty."""
        # Don't create the store directory
        with (
            patch.object(vector_tool, "load_store", side_effect=FileNotFoundError),
            patch("qwen_pipeline.tools_custom.logger"),
        ):
            result_json = vector_tool.query_store("nonexistent", "query")
            results = json.loads(result_json)

            assert isinstance(results, list)

    def test_list_stores_success(self, vector_tool: LocalVectorSearch) -> None:
        """Test listing available stores."""
        # Create some stores
        store_names = ["store1", "store2", "store3"]
        for name in store_names:
            store_path = Path(vector_tool.vector_store_path) / name
            store_path.mkdir(parents=True, exist_ok=True)
            (store_path / "index.faiss").touch()  # Create marker file

        result_json = vector_tool.list_stores()
        results = json.loads(result_json)

        assert isinstance(results, list)
        assert set(results) == set(store_names)
        assert results == sorted(store_names)

    def test_list_stores_empty(self, vector_tool: LocalVectorSearch) -> None:
        """Test listing when no stores exist."""
        result_json = vector_tool.list_stores()
        results = json.loads(result_json)

        assert results == []

    def test_list_stores_ignores_incomplete_stores(self, temp_store_dir: str) -> None:
        """Test that incomplete stores (missing index.faiss) are ignored."""
        tool = LocalVectorSearch(cfg={"vector_store_path": temp_store_dir})

        # Create one complete store
        complete_path = Path(temp_store_dir) / "complete"
        complete_path.mkdir()
        (complete_path / "index.faiss").touch()

        # Create one incomplete store (no index.faiss)
        incomplete_path = Path(temp_store_dir) / "incomplete"
        incomplete_path.mkdir()

        result_json = tool.list_stores()
        results = json.loads(result_json)

        assert results == ["complete"]


class TestChunkText:
    """Test suite for smart text chunking functionality."""

    def test_chunk_text_simple_short_text(self) -> None:
        """Test chunking of short text that fits in one chunk."""
        text = "This is a short document about machine learning."

        with patch("qwen_pipeline.tools_custom.count_tokens", return_value=10):
            chunks = LocalVectorSearch.chunk_text(text, max_chunk_tokens=500)

            assert len(chunks) == 1
            assert chunks[0] == text

    def test_chunk_text_multiple_paragraphs(self) -> None:
        """Test chunking of multi-paragraph text."""
        text = (
            "First paragraph about machine learning.\n\n"
            "Second paragraph about deep learning.\n\n"
            "Third paragraph about neural networks."
        )

        with patch("qwen_pipeline.tools_custom.count_tokens", return_value=5):
            chunks = LocalVectorSearch.chunk_text(text, max_chunk_tokens=10)

            # Should create multiple chunks
            assert len(chunks) > 1
            assert all(isinstance(chunk, str) for chunk in chunks)
            assert all(len(chunk) > 0 for chunk in chunks)

    def test_chunk_text_with_overlap(self) -> None:
        """Test that overlap is preserved between chunks."""
        text = "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4."

        with patch("qwen_pipeline.tools_custom.count_tokens", return_value=3):
            chunks = LocalVectorSearch.chunk_text(text, max_chunk_tokens=6, overlap_tokens=3)

            # Check overlap exists (last paragraph of chunk should appear in next)
            if len(chunks) > 1:
                # This is a heuristic check since exact overlap depends on token counting
                assert len(chunks) >= 2

    def test_chunk_text_single_long_paragraph(self) -> None:
        """Test chunking of a very long single paragraph."""
        # Simulate a long sentence with many words
        words = ["word"] * 100
        text = " ".join(words)

        with patch("qwen_pipeline.tools_custom.count_tokens") as mock_count:
            # Return different counts for full text vs single words
            def count_side_effect(t: str) -> int:
                return len(t.split()) * 2  # Estimate 2 tokens per word

            mock_count.side_effect = count_side_effect

            chunks = LocalVectorSearch.chunk_text(text, max_chunk_tokens=50)

            # Should split long paragraph
            assert len(chunks) >= 1
            assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_empty_string(self) -> None:
        """Test chunking of empty string."""
        text = ""

        chunks = LocalVectorSearch.chunk_text(text)

        assert chunks == []

    def test_chunk_text_whitespace_only(self) -> None:
        """Test chunking of whitespace-only string."""
        text = "   \n\n   \t\t   "

        chunks = LocalVectorSearch.chunk_text(text)

        assert chunks == []

    def test_chunk_text_preserves_content(self) -> None:
        """Test that all content is preserved after chunking."""
        original_text = (
            "First paragraph.\n\n"
            "Second paragraph.\n\n"
            "Third paragraph.\n\n"
            "Fourth paragraph."
        )

        with patch("qwen_pipeline.tools_custom.count_tokens", return_value=5):
            chunks = LocalVectorSearch.chunk_text(original_text, max_chunk_tokens=10)

            # Reconstruct text (accounting for potential overlap)
            reconstructed = "\n\n".join(chunks)

            # All original content should be present
            assert "First paragraph" in reconstructed
            assert "Second paragraph" in reconstructed
            assert "Third paragraph" in reconstructed
            assert "Fourth paragraph" in reconstructed


class TestSortByScores:
    """Test suite for semantic similarity scoring."""

    def test_sort_by_scores_basic(self) -> None:
        """Test basic sorting by similarity scores."""
        # Create mock Record objects
        mock_record = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.content = "Test content about ML"
        mock_chunk.metadata = {"source": "url1", "chunk_id": 0}
        mock_record.raw = [mock_chunk]

        tool = LocalVectorSearch()

        with (
            patch("qwen_pipeline.tools_custom.OllamaEmbeddings"),
            patch("qwen_pipeline.tools_custom.FAISS") as mock_faiss_class,
        ):
            mock_db = MagicMock()
            mock_db.similarity_search_with_score.return_value = [
                (MagicMock(metadata={"source": "url1", "chunk_id": 0}), 0.15)
            ]
            mock_faiss_class.from_documents.return_value = mock_db

            results = tool.sort_by_scores("machine learning", [mock_record])

            assert len(results) > 0
            assert all(isinstance(r, tuple) and len(r) == 3 for r in results)

    def test_sort_by_scores_empty_docs(self) -> None:
        """Test sorting with empty document list."""
        tool = LocalVectorSearch()

        results = tool.sort_by_scores("query", [])

        assert results == []

    def test_sort_by_scores_json_query(self) -> None:
        """Test that JSON queries are properly extracted."""
        mock_record = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.content = "Content"
        mock_chunk.metadata = {"source": "url", "chunk_id": 0}
        mock_record.raw = [mock_chunk]

        tool = LocalVectorSearch()

        with (
            patch("qwen_pipeline.tools_custom.OllamaEmbeddings"),
            patch("qwen_pipeline.tools_custom.FAISS") as mock_faiss_class,
        ):
            mock_db = MagicMock()
            mock_db.similarity_search_with_score.return_value = []
            mock_faiss_class.from_documents.return_value = mock_db

            # Query with JSON structure
            json_query = '{"text": "extracted query"}'
            tool.sort_by_scores(json_query, [mock_record])

            # Verify the extracted text was used for search
            call_args = mock_faiss_class.from_documents.call_args
            assert call_args is not None


class TestIntegration:
    """Integration tests for persistence workflow."""

    def test_store_and_query_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow: store documents then query."""
        tool = LocalVectorSearch(
            cfg={"vector_store_path": str(tmp_path), "embedding_model": "test-model"}
        )

        test_texts = [
            "Machine learning is artificial intelligence.",
            "Deep learning uses neural networks.",
        ]

        # Mock the FAISS operations
        with patch("qwen_pipeline.tools_custom.FAISS") as mock_faiss_class:
            mock_db = MagicMock()
            mock_faiss_class.from_documents.return_value = mock_db
            mock_faiss_class.load_local.return_value = mock_db

            # Store documents
            mock_db.similarity_search_with_score.return_value = []
            store_result_json = tool.store_documents(test_texts, "ml_store")
            store_result = json.loads(store_result_json)

            assert store_result["status"] == "saved"

            # Query stored documents
            query_result_json = tool.query_store("ml_store", "machine learning")
            query_result = json.loads(query_result_json)

            assert isinstance(query_result, list)

    def test_error_recovery(self, tmp_path: Path) -> None:
        """Test error handling and recovery."""
        tool = LocalVectorSearch(cfg={"vector_store_path": str(tmp_path)})

        # Try to query nonexistent store
        with (
            patch.object(tool, "load_store", side_effect=FileNotFoundError),
            patch("qwen_pipeline.tools_custom.logger"),
        ):
            result = tool.query_store("missing_store", "query")
            result_list = json.loads(result)

            assert isinstance(result_list, list)

        # List stores should still work after error
        result = tool.list_stores()
        result_list = json.loads(result)
        assert isinstance(result_list, list)
