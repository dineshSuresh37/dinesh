"""
test_pipeline.py – Tests for the RAG pipeline layer.

Covers:
  - TextChunker  : size, overlap, metadata preservation
  - Embedder     : shape, caching/determinism (SentenceTransformer mocked)
  - VectorStoreManager : FAISS index, search, search_all_versions
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ingestion.loader import Document
from app.pipeline.chunker import Chunk, TextChunker
from app.pipeline.embedder import Embedder
from app.pipeline.vector_store import VectorStoreManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBED_DIM = 64  # small synthetic dimension used across all tests


def _unit_rand(n: int, dim: int = _EMBED_DIM) -> list[np.ndarray]:
    """Return *n* random unit-normalised float32 vectors of *dim* dimensions."""
    vecs = np.random.default_rng(0).random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return list(vecs / norms)


def _make_doc(text: str, **meta) -> Document:
    base = {"source_file": "test.pdf", "version_label": "v1", "page_number": 1,
            "section_title": "Intro", "file_type": "pdf"}
    base.update(meta)
    return Document(text=text, metadata=base)


def _make_chunk(text: str, idx: int = 0, version: str = "v1") -> Chunk:
    return Chunk(
        text=text,
        metadata={
            "source_file": "test.pdf",
            "version_label": version,
            "page_number": 1,
            "section_title": "Intro",
            "file_type": "pdf",
            "chunk_index": idx,
            "total_chunks": 1,
        },
    )


def _mock_st_model(dim: int = _EMBED_DIM) -> MagicMock:
    """Return a SentenceTransformer mock whose .encode() returns deterministic vecs."""
    model = MagicMock()

    def _encode(texts, normalize_embeddings=True, show_progress_bar=False):
        # Return the same unit vector for every text so results are deterministic.
        return np.ones((len(texts), dim), dtype=np.float32) / (dim ** 0.5)

    model.encode.side_effect = _encode
    return model


# ---------------------------------------------------------------------------
# TextChunker
# ---------------------------------------------------------------------------


class TestTextChunker:
    CHUNK_SIZE = 200
    OVERLAP = 50

    @pytest.fixture()
    def chunker(self):
        return TextChunker(chunk_size=self.CHUNK_SIZE, chunk_overlap=self.OVERLAP)

    # ── size constraint ──────────────────────────────────────────────────────

    def test_chunks_do_not_exceed_chunk_size(self, chunker):
        doc = _make_doc("x" * 1_000)
        chunks = chunker.chunk([doc])
        assert chunks
        for c in chunks:
            assert len(c.text) <= self.CHUNK_SIZE, (
                f"Chunk too long ({len(c.text)} > {self.CHUNK_SIZE}): {c.text[:40]!r}"
            )

    def test_short_text_produces_single_chunk(self, chunker):
        doc = _make_doc("Short text.")
        chunks = chunker.chunk([doc])
        assert len(chunks) == 1

    def test_empty_text_produces_no_chunks(self, chunker):
        doc = _make_doc("")
        chunks = chunker.chunk([doc])
        assert chunks == []

    # ── overlap ─────────────────────────────────────────────────────────────

    def test_adjacent_chunks_share_overlap(self, chunker):
        # Use text without sentence boundaries so the hard-cut logic is used.
        # chunk[0] = text[0:200], chunk[1] starts at 200-50=150.
        text = "a" * 800
        doc = _make_doc(text)
        chunks = chunker.chunk([doc])
        assert len(chunks) >= 2
        tail = chunks[0].text[-self.OVERLAP:]
        head = chunks[1].text[:self.OVERLAP]
        assert tail == head, (
            f"Expected {self.OVERLAP}-char overlap: tail={tail!r} head={head!r}"
        )

    def test_no_overlap_when_chunk_covers_whole_text(self, chunker):
        doc = _make_doc("a" * (self.CHUNK_SIZE - 1))
        chunks = chunker.chunk([doc])
        # Only one chunk → overlap rule never fires.
        assert len(chunks) == 1

    # ── metadata propagation ─────────────────────────────────────────────────

    def test_parent_metadata_preserved(self, chunker):
        meta = {
            "source_file": "report.pdf",
            "version_label": "final",
            "page_number": 3,
            "section_title": "Results",
            "file_type": "pdf",
        }
        doc = Document(text="word " * 300, metadata=meta)
        chunks = chunker.chunk([doc])
        for c in chunks:
            for key, val in meta.items():
                assert c.metadata[key] == val, (
                    f"metadata[{key!r}] = {c.metadata.get(key)!r}, expected {val!r}"
                )

    def test_chunk_index_and_total_added(self, chunker):
        doc = _make_doc("word " * 300)
        chunks = chunker.chunk([doc])
        total = len(chunks)
        for idx, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == idx
            assert c.metadata["total_chunks"] == total

    def test_multiple_documents_chunked_in_order(self, chunker):
        docs = [_make_doc("a" * 300), _make_doc("b" * 300)]
        chunks = chunker.chunk(docs)
        texts = [c.text[0] for c in chunks]
        assert "a" in texts
        assert "b" in texts
        # 'a' chunks come before 'b' chunks.
        first_b = next(i for i, t in enumerate(texts) if t == "b")
        last_a = max(i for i, t in enumerate(texts) if t == "a")
        assert last_a < first_b


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class TestEmbedder:
    def test_embedding_count_matches_chunk_count(self, tmp_path):
        chunks = [_make_chunk(f"text {i}", idx=i) for i in range(7)]
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            MockST.return_value = _mock_st_model()
            embedder = Embedder(cache_dir=tmp_path)
            embeddings = embedder.embed(chunks)

        assert len(embeddings) == 7

    def test_embedding_dimension_consistent(self, tmp_path):
        chunks = [_make_chunk(f"text {i}", idx=i) for i in range(3)]
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            MockST.return_value = _mock_st_model(_EMBED_DIM)
            embedder = Embedder(cache_dir=tmp_path)
            embeddings = embedder.embed(chunks)

        dims = {e.shape[0] for e in embeddings}
        assert len(dims) == 1, f"Inconsistent dimensions: {dims}"

    def test_embeddings_are_numpy_arrays(self, tmp_path):
        chunks = [_make_chunk("hello world", idx=0)]
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            MockST.return_value = _mock_st_model()
            embedder = Embedder(cache_dir=tmp_path)
            embeddings = embedder.embed(chunks)

        assert isinstance(embeddings[0], np.ndarray)
        assert embeddings[0].dtype == np.float32

    def test_embeddings_are_deterministic_via_cache(self, tmp_path):
        """Second embed() for the same chunk must return the cached vector."""
        chunk = _make_chunk("cached text", idx=0)
        fixed_vec = np.ones((_EMBED_DIM,), dtype=np.float32) / (_EMBED_DIM ** 0.5)

        # First call: model encodes and caches.
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = fixed_vec.reshape(1, -1)
            MockST.return_value = mock_model
            embedder1 = Embedder(cache_dir=tmp_path)
            vecs1 = embedder1.embed([chunk])

        # Second call: different mock returns zeros — cache is hit instead.
        with patch("sentence_transformers.SentenceTransformer") as MockST2:
            mock_model2 = MagicMock()
            mock_model2.encode.return_value = np.zeros((1, _EMBED_DIM), dtype=np.float32)
            MockST2.return_value = mock_model2
            embedder2 = Embedder(cache_dir=tmp_path)
            vecs2 = embedder2.embed([chunk])

        np.testing.assert_array_equal(vecs1[0], vecs2[0])

    def test_embed_query_returns_1d_array(self, tmp_path):
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            MockST.return_value = _mock_st_model()
            embedder = Embedder(cache_dir=tmp_path)
            vec = embedder.embed_query("what changed?")

        assert vec.ndim == 1
        assert vec.dtype == np.float32


# ---------------------------------------------------------------------------
# VectorStoreManager (FAISS backend)
# ---------------------------------------------------------------------------


class TestVectorStoreManager:
    """Tests use the FAISS backend with a temporary data directory."""

    @pytest.fixture()
    def embedder_mock(self):
        """A fake Embedder whose embed_query returns a fixed unit vector."""
        mock = MagicMock(spec=Embedder)
        q_vec = np.ones(_EMBED_DIM, dtype=np.float32) / (_EMBED_DIM ** 0.5)
        mock.embed_query.return_value = q_vec
        return mock

    @pytest.fixture()
    def vsm(self, tmp_path, embedder_mock, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE", "faiss")
        return VectorStoreManager(embedder_mock, data_dir=tmp_path)

    @staticmethod
    def _make_chunks(n: int, version: str = "v1") -> list[Chunk]:
        return [
            Chunk(
                text=f"chunk {i} from {version}",
                metadata={
                    "source_file": "doc.pdf",
                    "version_label": version,
                    "page_number": 1,
                    "section_title": "Test",
                    "file_type": "pdf",
                    "chunk_index": i,
                    "total_chunks": n,
                },
            )
            for i in range(n)
        ]

    # ── index ────────────────────────────────────────────────────────────────

    def test_index_does_not_raise(self, vsm):
        chunks = self._make_chunks(5, "v1")
        embeddings = _unit_rand(5)
        vsm.index("v1", chunks, embeddings)  # no exception

    def test_index_mismatched_lengths_raises(self, vsm):
        chunks = self._make_chunks(3, "v1")
        embeddings = _unit_rand(5)
        with pytest.raises(ValueError):
            vsm.index("v1", chunks, embeddings)

    # ── search ───────────────────────────────────────────────────────────────

    def test_search_returns_top_k_results(self, vsm):
        chunks = self._make_chunks(10, "v1")
        embeddings = _unit_rand(10)
        vsm.index("v1", chunks, embeddings)

        results = vsm.search("query", "v1", top_k=5)
        assert len(results) == 5

    def test_search_results_have_text_and_metadata(self, vsm):
        chunks = self._make_chunks(4, "v1")
        embeddings = _unit_rand(4)
        vsm.index("v1", chunks, embeddings)

        results = vsm.search("query", "v1", top_k=2)
        for r in results:
            assert isinstance(r.text, str) and r.text
            assert "version_label" in r.metadata
            assert "source_file" in r.metadata

    def test_search_result_scores_are_floats(self, vsm):
        chunks = self._make_chunks(3, "v1")
        embeddings = _unit_rand(3)
        vsm.index("v1", chunks, embeddings)

        results = vsm.search("query", "v1", top_k=3)
        for r in results:
            assert isinstance(r.score, float)

    def test_search_missing_index_raises(self, vsm):
        with pytest.raises(FileNotFoundError):
            vsm.search("query", "nonexistent-version", top_k=5)

    def test_top_k_capped_at_index_size(self, vsm):
        chunks = self._make_chunks(3, "v1")
        embeddings = _unit_rand(3)
        vsm.index("v1", chunks, embeddings)

        results = vsm.search("query", "v1", top_k=100)
        assert len(results) == 3  # only 3 docs in index

    # ── search_all_versions ──────────────────────────────────────────────────

    def test_search_all_versions_returns_each_version(self, vsm):
        for ver in ("alpha", "beta"):
            chunks = self._make_chunks(4, ver)
            embeddings = _unit_rand(4)
            vsm.index(ver, chunks, embeddings)

        results = vsm.search_all_versions("query", top_k_per_version=3)
        assert "alpha" in results
        assert "beta" in results

    def test_search_all_versions_per_version_count(self, vsm):
        for ver in ("v1", "v2"):
            chunks = self._make_chunks(6, ver)
            embeddings = _unit_rand(6)
            vsm.index(ver, chunks, embeddings)

        results = vsm.search_all_versions("query", top_k_per_version=2)
        for ver, chunks in results.items():
            assert len(chunks) <= 2, f"Expected ≤2 results for {ver}, got {len(chunks)}"
