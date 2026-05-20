"""
vector_store.py – Per-document-version FAISS or Chroma vector store management.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from app.pipeline.chunker import Chunk
    from app.pipeline.embedder import Embedder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path("data")
_DEFAULT_CHROMA_DIR = "chroma_db"
_CHROMA_COLLECTION_PREFIX = "rag_"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(version_label: str) -> str:
    """Return a filesystem- and collection-safe version of *version_label*."""
    from app.utils.helpers import safe_filename

    return safe_filename(version_label)


def _sanitize_metadata(meta: dict) -> dict:
    """Return a copy of *meta* whose values are Chroma-compatible primitives.

    Chroma metadata values must be ``str``, ``int``, or ``float``.
    Booleans are coerced to int; everything else is stringified.
    """
    result: dict = {}
    for k, v in meta.items():
        key = str(k)
        if isinstance(v, bool):
            result[key] = int(v)
        elif isinstance(v, (int, float, str)):
            result[key] = v
        else:
            result[key] = str(v)
    return result


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Return *vectors* L2-normalised to unit norm (in-place safe copy)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# VectorStoreManager
# ---------------------------------------------------------------------------


class VectorStoreManager:
    """Manage per-version vector indexes backed by FAISS or Chroma.

    The active backend is controlled by the ``VECTOR_STORE`` environment
    variable (``"faiss"`` or ``"chroma"``; default ``"faiss"``).

    Each document version is stored in its own namespace:

    - **FAISS**: ``{data_dir}/{safe_version}.faiss`` (index) +
      ``{data_dir}/{safe_version}.chunks.pkl`` (serialised chunks).
    - **Chroma**: a dedicated collection named ``rag_{safe_version}``.

    Embedding vectors are stored unit-normalised so that inner-product search
    equals cosine similarity.

    Parameters
    ----------
    embedder:
        An :class:`~app.pipeline.embedder.Embedder` instance used to embed
        query strings during :meth:`search`.
    data_dir:
        Root directory for FAISS index files.  Reads ``DATA_DIR`` env var;
        defaults to ``data/``.
    """

    def __init__(
        self,
        embedder: "Embedder",
        data_dir: Path | str | None = None,
    ) -> None:
        self.embedder = embedder
        self.data_dir = Path(
            data_dir if data_dir is not None else os.environ.get("DATA_DIR", _DEFAULT_DATA_DIR)
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backend = os.environ.get("VECTOR_STORE", "faiss").lower()
        if self.backend not in {"faiss", "chroma"}:
            raise ValueError(
                f"Unknown VECTOR_STORE backend: {self.backend!r}. "
                "Set VECTOR_STORE to 'faiss' or 'chroma'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(
        self,
        version_label: str,
        chunks: List["Chunk"],
        embeddings: List[np.ndarray],
    ) -> None:
        """Build and persist an index for *version_label*.

        If an index for this version already exists it is overwritten.

        Parameters
        ----------
        version_label:
            Caller-supplied version identifier (e.g. ``"v1"``).
        chunks:
            Chunk objects to store (used for result reconstruction).
        embeddings:
            Parallel list of embedding vectors from
            :meth:`~app.pipeline.embedder.Embedder.embed`.

        Raises
        ------
        ValueError
            If *chunks* and *embeddings* have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )
        if not chunks:
            return

        if self.backend == "faiss":
            self._faiss_index(version_label, chunks, embeddings)
        else:
            self._chroma_index(version_label, chunks, embeddings)

    def search(
        self,
        query: str,
        version_label: str,
        top_k: int = 5,
    ) -> List["Chunk"]:
        """Search *version_label*'s index for the closest chunks to *query*.

        Parameters
        ----------
        query:
            Raw query string; embedded internally using :attr:`embedder`.
        version_label:
            Version to search.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        List[Chunk]
            Results ordered by descending cosine similarity, each with
            ``score`` set to the cosine similarity value (``[-1, 1]`` for
            FAISS; ``[−1, 1]`` for Chroma).

        Raises
        ------
        FileNotFoundError
            If no index exists for *version_label*.
        """
        query_vec = self.embedder.embed_query(query)
        if self.backend == "faiss":
            return self._faiss_search(query_vec, version_label, top_k)
        return self._chroma_search(query_vec, version_label, top_k)

    def search_all_versions(
        self,
        query: str,
        top_k_per_version: int = 3,
    ) -> Dict[str, List["Chunk"]]:
        """Search every indexed version and return top results per version.

        Parameters
        ----------
        query:
            Raw query string.
        top_k_per_version:
            Maximum results returned for each version.

        Returns
        -------
        Dict[str, List[Chunk]]
            Keys are version labels (as stored), values are lists of Chunk
            objects ordered by descending cosine similarity.  Versions that
            produce no results are omitted.
        """
        query_vec = self.embedder.embed_query(query)
        if self.backend == "faiss":
            return self._faiss_search_all(query_vec, top_k_per_version)
        return self._chroma_search_all(query_vec, top_k_per_version)

    # ------------------------------------------------------------------
    # FAISS backend
    # ------------------------------------------------------------------

    def _faiss_index_path(self, version_label: str) -> Path:
        return self.data_dir / f"{_safe(version_label)}.faiss"

    def _faiss_chunks_path(self, version_label: str) -> Path:
        return self.data_dir / f"{_safe(version_label)}.chunks.pkl"

    def _faiss_index(
        self,
        version_label: str,
        chunks: List["Chunk"],
        embeddings: List[np.ndarray],
    ) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for FAISS indexing. "
                "Install with: pip install faiss-cpu"
            ) from exc

        vectors = _normalize(np.stack(embeddings, axis=0))
        dim = vectors.shape[1]
        # IndexFlatIP: exact inner-product search.
        # With unit-normalised vectors this equals cosine similarity.
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(vectors)

        faiss.write_index(faiss_index, str(self._faiss_index_path(version_label)))
        with open(self._faiss_chunks_path(version_label), "wb") as fh:
            pickle.dump(chunks, fh)

    def _faiss_search(
        self,
        query_vec: np.ndarray,
        version_label: str,
        top_k: int,
    ) -> List["Chunk"]:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            ) from exc

        index_path = self._faiss_index_path(version_label)
        if not index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index found for version {version_label!r}. "
                f"Expected: {index_path}"
            )

        faiss_index = faiss.read_index(str(index_path))
        with open(self._faiss_chunks_path(version_label), "rb") as fh:
            stored_chunks: List["Chunk"] = pickle.load(fh)

        q = _normalize(query_vec.reshape(1, -1))
        scores, indices = faiss_index.search(q, min(top_k, faiss_index.ntotal))

        from app.pipeline.chunker import Chunk

        results: List[Chunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(stored_chunks):
                continue
            c = stored_chunks[idx]
            results.append(Chunk(text=c.text, metadata=c.metadata, score=float(score)))
        return results

    def _faiss_search_all(
        self,
        query_vec: np.ndarray,
        top_k_per_version: int,
    ) -> Dict[str, List["Chunk"]]:
        results: Dict[str, List["Chunk"]] = {}
        for faiss_file in sorted(self.data_dir.glob("*.faiss")):
            # Recover the original version_label from the stored chunks so the
            # returned key matches what the caller passed to index().
            chunks_path = faiss_file.with_suffix(".chunks.pkl")
            try:
                with open(chunks_path, "rb") as fh:
                    stored: list = pickle.load(fh)
                version_label = (
                    stored[0].metadata.get("version_label", faiss_file.stem)
                    if stored
                    else faiss_file.stem
                )
                chunks = self._faiss_search(query_vec, version_label, top_k_per_version)
                if chunks:
                    results[version_label] = chunks
            except Exception:  # noqa: BLE001
                continue
        return results

    # ------------------------------------------------------------------
    # Chroma backend
    # ------------------------------------------------------------------

    def _chroma_client(self):
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            ) from exc
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", _DEFAULT_CHROMA_DIR)
        return chromadb.PersistentClient(path=persist_dir)

    def _collection_name(self, version_label: str) -> str:
        return f"{_CHROMA_COLLECTION_PREFIX}{_safe(version_label)}"

    def _chroma_index(
        self,
        version_label: str,
        chunks: List["Chunk"],
        embeddings: List[np.ndarray],
    ) -> None:
        client = self._chroma_client()
        name = self._collection_name(version_label)

        # Overwrite any existing collection for this version.
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001
            pass

        # cosine space: Chroma returns distances as (1 - cosine_similarity),
        # so we convert back to similarity in _chroma_search.
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        ids = [f"{name}_{i}" for i in range(len(chunks))]
        collection.add(
            ids=ids,
            documents=[c.text for c in chunks],
            metadatas=[_sanitize_metadata(c.metadata) for c in chunks],
            embeddings=[e.tolist() for e in embeddings],
        )

    def _chroma_search(
        self,
        query_vec: np.ndarray,
        version_label: str,
        top_k: int,
    ) -> List["Chunk"]:
        client = self._chroma_client()
        name = self._collection_name(version_label)

        try:
            collection = client.get_collection(name)
        except Exception as exc:
            raise FileNotFoundError(
                f"No Chroma collection found for version {version_label!r}. "
                f"Expected collection: {name}"
            ) from exc

        result = collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        from app.pipeline.chunker import Chunk

        chunks: List[Chunk] = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            # Chroma cosine distance = 1 - cosine_similarity
            score = 1.0 - float(dist)
            chunks.append(Chunk(text=doc, metadata=meta, score=score))
        return chunks

    def _chroma_search_all(
        self,
        query_vec: np.ndarray,
        top_k_per_version: int,
    ) -> Dict[str, List["Chunk"]]:
        client = self._chroma_client()
        results: Dict[str, List["Chunk"]] = {}
        for collection in client.list_collections():
            name: str = collection.name
            if not name.startswith(_CHROMA_COLLECTION_PREFIX):
                continue
            # Recover original version_label from stored metadata.
            safe_version = name[len(_CHROMA_COLLECTION_PREFIX):]
            try:
                chunks = self._chroma_search(query_vec, safe_version, top_k_per_version)
                if not chunks:
                    continue
                # Use the original version_label from stored metadata when available.
                version_label = chunks[0].metadata.get("version_label", safe_version)
                results[version_label] = chunks
            except Exception:  # noqa: BLE001
                continue
        return results


# ---------------------------------------------------------------------------
# Legacy functional API (used by streamlit_app.py)
# ---------------------------------------------------------------------------


def build_faiss_index(embeddings: List[List[float]]):
    """Build an in-memory FAISS flat (L2) index.

    .. deprecated::
        Prefer :class:`VectorStoreManager` which handles persistence,
        cosine similarity, and multi-version namespacing.
    """
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu and numpy are required. "
            "Install with: pip install faiss-cpu numpy"
        ) from exc

    vectors = np.array(embeddings, dtype="float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def search_faiss(index, query_embedding: List[float], top_k: int = 5) -> Tuple:
    """Search a FAISS index for the top-k nearest neighbours.

    .. deprecated::
        Prefer :meth:`VectorStoreManager.search`.

    Returns
    -------
    Tuple
        ``(distances array, indices array)`` as returned by FAISS.
    """
    query = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query, top_k)
    return distances, indices


def build_chroma_collection(
    chunks: List[str],
    embeddings: List[List[float]],
    collection_name: str,
    persist_directory: str = "./chroma_db",
):
    """Create (or overwrite) a Chroma collection.

    .. deprecated::
        Prefer :class:`VectorStoreManager`.
    """
    try:
        import chromadb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "chromadb is required. Install with: pip install chromadb"
        ) from exc

    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection(collection_name)
    except Exception:  # noqa: BLE001
        pass

    collection = client.create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection


def query_chroma(collection, query_embedding: List[float], top_k: int = 5):
    """Query a Chroma collection for the top-k most similar chunks.

    .. deprecated::
        Prefer :meth:`VectorStoreManager.search`.

    Returns
    -------
    The raw chromadb ``QueryResult`` dict.
    """
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
