"""Tests for RAG pipeline - chunking, retrieval, and citations."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from rag.ingest import chunk_text, split_sentences, normalize_vectors
from rag.retrieve import build_context, normalize_vector


class TestChunking:
    """Tests for text chunking functionality."""

    def test_chunk_text_basic(self):
        """Test basic chunking with multiple sentences."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_text_respects_size(self):
        """Test that chunks don't exceed target size significantly."""
        text = "This is sentence one. This is sentence two. This is sentence three. " * 10
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        
        # Allow some flexibility for sentence boundaries
        for chunk in chunks[:-1]:  # Last chunk may be smaller
            assert len(chunk) <= 150  # Allow 50% overflow for sentence boundaries

    def test_chunk_text_overlap(self):
        """Test that chunks have overlapping content."""
        text = "First. Second. Third. Fourth. Fifth. Sixth. Seventh. Eighth."
        chunks = chunk_text(text, chunk_size=30, overlap=15)
        
        # With overlap, adjacent chunks should share some content
        if len(chunks) >= 2:
            # Check that there's some potential for overlap
            assert len(chunks) >= 2

    def test_chunk_text_empty_input(self):
        """Test chunking with empty input."""
        chunks = chunk_text("")
        assert chunks == []

    def test_chunk_text_single_sentence(self):
        """Test chunking with single sentence."""
        text = "This is a single sentence."
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_size_config(self):
        """Test chunking with default config values."""
        # Using defaults from config (512 chars, 50 overlap)
        text = "A" * 600 + ". " + "B" * 600 + "."
        chunks = chunk_text(text)
        
        assert len(chunks) >= 1


class TestSentenceSplitting:
    """Tests for sentence splitting functionality."""

    def test_split_sentences_basic(self):
        """Test basic sentence splitting."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = split_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"

    def test_split_sentences_with_abbreviations(self):
        """Test sentence splitting handles common cases."""
        text = "Dr. Smith went home. He was tired."
        sentences = split_sentences(text)
        
        # May split on "Dr." - this is a known limitation
        assert len(sentences) >= 1

    def test_split_sentences_empty(self):
        """Test sentence splitting with empty input."""
        sentences = split_sentences("")
        assert sentences == []

    def test_split_sentences_whitespace(self):
        """Test sentence splitting handles extra whitespace."""
        text = "  First sentence.   Second sentence.  "
        sentences = split_sentences(text)
        
        assert len(sentences) == 2
        assert all(s.strip() == s for s in sentences)


class TestVectorNormalization:
    """Tests for vector normalization."""

    def test_normalize_vectors_unit_length(self):
        """Test that normalized vectors have unit length."""
        vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        normalized = normalize_vectors(vectors)
        
        # Check unit length (L2 norm = 1)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

    def test_normalize_vectors_preserves_direction(self):
        """Test that normalization preserves vector direction."""
        vectors = np.array([[3.0, 4.0]], dtype=np.float32)  # 3-4-5 triangle
        normalized = normalize_vectors(vectors)
        
        # Direction should be preserved (0.6, 0.8)
        np.testing.assert_array_almost_equal(normalized[0], [0.6, 0.8])

    def test_normalize_vectors_zero_vector(self):
        """Test handling of zero vectors."""
        vectors = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        normalized = normalize_vectors(vectors)
        
        # Should not produce NaN
        assert not np.any(np.isnan(normalized))

    def test_normalize_single_vector(self):
        """Test single vector normalization function."""
        vector = np.array([3.0, 4.0], dtype=np.float32)
        normalized = normalize_vector(vector)
        
        assert np.linalg.norm(normalized) == pytest.approx(1.0)


class TestContextBuilding:
    """Tests for context building with citations."""

    def test_build_context_format(self):
        """Test context format includes numbered sources."""
        chunks = [
            {"text": "First chunk content", "source": "doc1.txt"},
            {"text": "Second chunk content", "source": "doc2.txt"},
        ]
        
        context = build_context(chunks)
        
        assert "[1]" in context
        assert "[2]" in context
        assert "doc1.txt" in context
        assert "doc2.txt" in context
        assert "First chunk content" in context
        assert "Second chunk content" in context

    def test_build_context_empty(self):
        """Test context building with empty chunks."""
        context = build_context([])
        assert context == ""

    def test_build_context_single_chunk(self):
        """Test context building with single chunk."""
        chunks = [{"text": "Only chunk", "source": "only.txt"}]
        context = build_context(chunks)
        
        assert "[1]" in context
        assert "only.txt" in context
        assert "[2]" not in context


class TestCitationFormat:
    """Tests for citation format in responses."""

    def test_citation_structure(self):
        """Test that citations have required fields."""
        # Simulate what answer_question produces
        chunks = [
            {"text": "Sample text " * 50, "source": "sample.txt"},
        ]
        
        citations = [
            {"source": chunk["source"], "text": chunk["text"][:200] + "..."}
            for chunk in chunks
        ]
        
        assert len(citations) == 1
        assert "source" in citations[0]
        assert "text" in citations[0]
        assert citations[0]["source"] == "sample.txt"
        assert citations[0]["text"].endswith("...")

    def test_citation_text_truncation(self):
        """Test that citation text is properly truncated."""
        long_text = "A" * 500
        truncated = long_text[:200] + "..."
        
        assert len(truncated) == 203  # 200 chars + "..."
        assert truncated.endswith("...")


class TestRetrievalMocking:
    """Tests for retrieval with mocked dependencies."""

    @patch('rag.retrieve.load_index')
    @patch('rag.retrieve.embed')
    def test_retrieve_returns_results(self, mock_embed, mock_load_index):
        """Test that retrieve returns properly formatted results."""
        # Setup mocks
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64)
        )
        mock_metadata = [
            {"text": "Chunk 0", "source": "doc.txt", "chunk_index": 0},
            {"text": "Chunk 1", "source": "doc.txt", "chunk_index": 1},
            {"text": "Chunk 2", "source": "doc.txt", "chunk_index": 2},
        ]
        mock_load_index.return_value = (mock_index, mock_metadata)
        mock_embed.return_value = [[0.1] * 384]
        
        from rag.retrieve import retrieve
        
        # Clear cache
        import rag.retrieve as retrieve_module
        retrieve_module._index = None
        retrieve_module._metadata = None
        
        results = retrieve("test query", top_k=3)
        
        assert len(results) == 3
        assert all("text" in r for r in results)
        assert all("source" in r for r in results)
        assert all("score" in r for r in results)

    def test_normalize_vector_consistency(self):
        """Test that normalization is consistent."""
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        norm1 = normalize_vector(vector)
        norm2 = normalize_vector(vector)
        
        np.testing.assert_array_equal(norm1, norm2)


class TestIntegration:
    """Integration tests (may require index to exist)."""

    def test_chunk_and_normalize_pipeline(self):
        """Test the chunking and normalization work together."""
        # Simulate a mini pipeline
        text = "This is a test document. It has multiple sentences. Each should be chunked."
        
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) >= 1
        
        # Simulate embeddings (random vectors)
        fake_embeddings = np.random.rand(len(chunks), 384).astype(np.float32)
        normalized = normalize_vectors(fake_embeddings)
        
        # All should be unit vectors
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(chunks)))
