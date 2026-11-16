"""Dense (semantic) channel and visual channel.

Default dense backend is LSA (TF-IDF + truncated SVD): zero downloads, deterministic, offline. The `EmbeddingBackend`
protocol lets you plug in a neural model (e.g. sentence-transformers) without touching retrieval code; see README.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional, Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ..schemas import Chunk, Document
from ..utils.text import tokenize


class EmbeddingBackend(Protocol):
    name: str

    def fit(self, texts: list[str]) -> np.ndarray: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


class LSABackend:
    name = "lsa"

    def __init__(self, dims: int = 160):
        self.dims = dims
        self.vec: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None

    def fit(self, texts: list[str]) -> np.ndarray:
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        X = self.vec.fit_transform(texts)
        n_comp = max(2, min(self.dims, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=7, n_iter=7)
        Z = self.svd.fit_transform(X)
        return _l2(Z)

    def encode(self, texts: list[str]) -> np.ndarray:
        assert self.vec is not None and self.svd is not None, "backend not fitted"
        return _l2(self.svd.transform(self.vec.transform(texts)))


class SentenceTransformerBackend:  # pragma: no cover - optional dependency, exercised only when installed
    name = "sentence-transformers"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def make_backend() -> EmbeddingBackend:
    choice = os.environ.get("MERIDIAN_EMBEDDINGS", "lsa").lower()
    if choice in {"st", "sentence-transformers"}:
        try:
            return SentenceTransformerBackend(os.environ.get("MERIDIAN_ST_MODEL", "BAAI/bge-small-en-v1.5"))
        except Exception:
            pass  # fall back to the offline backend instead of failing to start
    return LSABackend()


class DenseIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        docs: Optional[dict[str, Document]] = None,
        backend: Optional[EmbeddingBackend] = None,
        vector_store: Optional[str] = None,
        qdrant_client=None,
        collection_name: Optional[str] = None,
    ):
        self.backend = backend or make_backend()
        self.n = len(chunks)
        self.vector_store = (vector_store or os.environ.get("MERIDIAN_VECTOR_STORE", "memory")).lower()
        self.collection_name = collection_name or os.environ.get("MERIDIAN_QDRANT_COLLECTION", "meridian_chunks")
        self._client = qdrant_client
        texts = [c.text for c in chunks]
        vectors = self.backend.fit(texts) if texts else np.zeros((0, 1))
        self.M: Optional[np.ndarray] = vectors if self.vector_store == "memory" else None
        self._point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)) for c in chunks]
        if self.vector_store == "qdrant" and chunks:
            self._index_qdrant(chunks, docs or {}, vectors)
        elif self.vector_store not in {"memory", "qdrant"}:
            raise ValueError(f"Unsupported vector store '{self.vector_store}'. Use 'memory' or 'qdrant'.")

    def _qdrant(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("Qdrant is selected but qdrant-client is not installed") from exc
            url = os.environ.get("MERIDIAN_QDRANT_URL", "http://localhost:6333")
            api_key = os.environ.get("MERIDIAN_QDRANT_API_KEY") or None
            self._client = QdrantClient(url=url, api_key=api_key, timeout=10)
        return self._client

    def _index_qdrant(self, chunks: list[Chunk], docs: dict[str, Document], vectors: np.ndarray) -> None:
        try:
            from qdrant_client.http import models
        except ImportError as exc:
            raise RuntimeError("Qdrant is selected but qdrant-client is not installed") from exc
        client = self._qdrant()
        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
        )
        batch_size = 256
        for start in range(0, len(chunks), batch_size):
            points = []
            for i in range(start, min(start + batch_size, len(chunks))):
                doc = docs.get(chunks[i].doc_id)
                point_models = [m.upper() for m in doc.models] if doc and doc.models else ["__generic__"]
                points.append(models.PointStruct(
                    id=self._point_ids[i],
                    vector=vectors[i].tolist(),
                    payload={"position": i, "models": point_models},
                ))
            client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, query: str, limit: int, models: Optional[list[str]] = None) -> list[tuple[int, float]]:
        if self.n == 0:
            return []
        q = self.backend.encode([query])[0]
        if self.vector_store == "memory":
            assert self.M is not None
            scores = self.M @ q
            indices = np.argsort(scores)[::-1][:limit]
            return [(int(i), float(scores[i])) for i in indices]

        query_filter = None
        if models:
            from qdrant_client.http import models as qmodels
            query_filter = qmodels.Filter(must=[
                qmodels.FieldCondition(
                    key="models",
                    match=qmodels.MatchAny(any=[m.upper() for m in models] + ["__generic__"]),
                )
            ])
        response = self._qdrant().query_points(
            collection_name=self.collection_name,
            query=q.tolist(),
            query_filter=query_filter,
            limit=min(limit, self.n),
            with_payload=True,
        )
        return [
            (int(point.payload["position"]), float(point.score))
            for point in response.points
            if point.payload and "position" in point.payload
        ]

    def vectors(self, indices: list[int]) -> dict[int, np.ndarray]:
        if not indices:
            return {}
        if self.vector_store == "memory":
            assert self.M is not None
            return {i: self.M[i] for i in indices}
        points = self._qdrant().retrieve(
            collection_name=self.collection_name,
            ids=[self._point_ids[i] for i in indices],
            with_vectors=True,
        )
        return {
            int(point.payload["position"]): np.asarray(point.vector, dtype=np.float32)
            for point in points
            if point.payload and "position" in point.payload and point.vector is not None
        }


class VisualIndex:
    """Global image descriptors (colour + gradient layout) for image->figure retrieval."""

    def __init__(self, chunks: list[Chunk]):
        self.positions = [i for i, c in enumerate(chunks) if c.visual_vec]
        self.n = len(chunks)
        self.M = _l2(np.array([chunks[i].visual_vec for i in self.positions], dtype=np.float32)) if self.positions else np.zeros((0, 1), np.float32)

    def score(self, vec: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        if not self.positions:
            return out
        v = vec.astype(np.float32)
        v = v / max(np.linalg.norm(v), 1e-9)
        sims = self.M @ v
        for pos, s in zip(self.positions, sims):
            out[pos] = float(s)
        return out
