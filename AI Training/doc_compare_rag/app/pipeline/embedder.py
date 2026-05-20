"""
embedder.py – Generate and cache embeddings for Chunk objects.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from app.pipeline.chunker import Chunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_CACHE_DIR = Path(".cache/embeddings")


# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------


class Embedder:
    """Generate unit-normalised embedding vectors for text chunks.

    The backend is selected automatically from the model name:

    - ``sentence-transformers/*`` → local :class:`SentenceTransformer` model.
    - ``text-embedding-*``        → OpenAI embeddings API.

    Results are cached to disk keyed by ``(source_file, version_label,
    chunk_index)``, so unchanged chunks are not re-embedded across runs.

    Parameters
    ----------
    model_name:
        Embedding model identifier.  Reads the ``EMBEDDING_MODEL`` environment
        variable; defaults to ``sentence-transformers/all-MiniLM-L6-v2``.
    cache_dir:
        Directory for the ``.npy`` embedding cache files.  Reads the
        ``EMBEDDING_CACHE_DIR`` environment variable; defaults to
        ``.cache/embeddings``.

    Raises
    ------
    ValueError
        If *model_name* does not match a known backend prefix.
    EnvironmentError
        If the OpenAI backend is selected but ``OPENAI_API_KEY`` is not set.
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.model_name: str = model_name or os.environ.get(
            "EMBEDDING_MODEL", _DEFAULT_MODEL
        )
        self.cache_dir = Path(
            cache_dir
            if cache_dir is not None
            else os.environ.get("EMBEDDING_CACHE_DIR", _DEFAULT_CACHE_DIR)
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Sentence-transformer model is lazy-loaded on first use.
        self._st_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, chunks: List["Chunk"]) -> List[np.ndarray]:
        """Return one unit-normalised embedding vector per chunk.

        Checks the disk cache first.  Only chunks that are not yet cached are
        sent to the embedding backend.  All results are stored to cache before
        returning.

        Parameters
        ----------
        chunks:
            List of :class:`~app.pipeline.chunker.Chunk` objects.

        Returns
        -------
        List[np.ndarray]
            One 1-D ``float32`` ndarray per chunk, unit-normalised for cosine
            similarity via inner product.
        """
        result: List[np.ndarray | None] = [None] * len(chunks)
        uncached_positions: List[int] = []
        uncached_texts: List[str] = []

        for i, chunk in enumerate(chunks):
            key = self._cache_key(chunk)
            vec = self._load_cache(key)
            if vec is not None:
                result[i] = vec
            else:
                uncached_positions.append(i)
                uncached_texts.append(chunk.text)

        if uncached_texts:
            fresh_vecs = self._encode(uncached_texts)
            for list_pos, chunk_idx in enumerate(uncached_positions):
                vec = fresh_vecs[list_pos]
                self._save_cache(self._cache_key(chunks[chunk_idx]), vec)
                result[chunk_idx] = vec

        return result  # type: ignore[return-value]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string, bypassing the chunk cache.

        Parameters
        ----------
        text:
            Raw query string to embed.

        Returns
        -------
        np.ndarray
            Unit-normalised 1-D ``float32`` embedding vector.
        """
        return self._encode([text])[0]

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> List[np.ndarray]:
        """Dispatch *texts* to the appropriate backend.

        Returns a list of unit-normalised ``float32`` ndarrays.
        """
        if self.model_name.startswith("sentence-transformers/"):
            return self._encode_sentence_transformers(texts)
        if self.model_name.startswith("text-embedding-"):
            return self._encode_openai(texts)
        raise ValueError(
            f"Unsupported embedding model: {self.model_name!r}. "
            "Model name must start with 'sentence-transformers/' or 'text-embedding-'."
        )

    def _encode_sentence_transformers(self, texts: List[str]) -> List[np.ndarray]:
        """Encode *texts* with a locally loaded SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            ) from exc

        if self._st_model is None:
            self._st_model = SentenceTransformer(self.model_name)

        # normalize_embeddings=True returns unit-L2 vectors suitable for
        # cosine similarity via inner product.
        vecs: np.ndarray = self._st_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [vecs[i].astype(np.float32) for i in range(len(texts))]

    def _encode_openai(self, texts: List[str]) -> List[np.ndarray]:
        """Encode *texts* via the OpenAI embeddings API."""
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install with: pip install openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Export it or add it to your .env file."
            )

        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(input=texts, model=self.model_name)
        raw_vecs = [
            np.array(item.embedding, dtype=np.float32) for item in response.data
        ]
        # Normalise to unit L2 norm for consistent cosine similarity via inner product.
        return [v / max(float(np.linalg.norm(v)), 1e-10) for v in raw_vecs]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(chunk: "Chunk") -> str:
        """Stable SHA-256 cache key derived from the chunk's provenance."""
        m = chunk.metadata
        raw = (
            f"{m.get('source_file', '')}"
            f"|{m.get('version_label', '')}"
            f"|{m.get('chunk_index', 0)}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    def _load_cache(self, key: str) -> np.ndarray | None:
        path = self._cache_path(key)
        return np.load(str(path)) if path.exists() else None

    def _save_cache(self, key: str, vec: np.ndarray) -> None:
        np.save(str(self._cache_path(key)), vec)


# ---------------------------------------------------------------------------
# Legacy functional API (used by streamlit_app.py)
# ---------------------------------------------------------------------------


def get_embeddings(chunks: List[str]) -> List[List[float]]:
    """Generate an embedding vector for each text string.

    .. deprecated::
        Prefer :class:`Embedder` which works with typed ``Chunk`` objects,
        supports multiple backends, and caches results to disk.

    Parameters
    ----------
    chunks:
        List of text strings to embed.

    Returns
    -------
    List[List[float]]
        List of embedding vectors as plain Python float lists.
    """
    embedder = Embedder()
    from app.pipeline.chunker import Chunk  # lazy import

    pseudo_chunks = [
        Chunk(text=t, metadata={"source_file": "_legacy", "version_label": "_legacy", "chunk_index": i})
        for i, t in enumerate(chunks)
    ]
    return [vec.tolist() for vec in embedder.embed(pseudo_chunks)]
