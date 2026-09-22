"""Shared online retrieval core for all four workload cells.

The index is deliberately scoped by conversation namespace.  LoCoMo queries
have a known scope, so a large multi-tenant corpus must not become an artificial
global document scan. Sparse snapshots are rebuilt copy-on-write only for
the affected 369--689-document scope; dense vectors are updated independently.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def render_rawerk(raw_text: str, entities: str = "", relations: str = "", keywords: str = "") -> str:
    pieces = [str(raw_text or "").strip()]
    for label, value in (("E:", entities), ("R:", relations), ("K:", keywords)):
        value = str(value or "").strip()
        if value:
            pieces.append(f"{label} {value}")
    return "\n".join(piece for piece in pieces if piece)


@dataclass(frozen=True)
class SparseSnapshot:
    memory_ids: tuple[str, ...]
    documents: tuple[str, ...]
    vectorizer: CountVectorizer
    doc_tf: object
    doc_len: np.ndarray
    avg_dl: float
    idf: np.ndarray
    len_norm: np.ndarray


@dataclass(frozen=True)
class DenseSnapshot:
    memory_ids: tuple[str, ...]
    normalized_embeddings: np.ndarray


@dataclass(frozen=True)
class SearchResult:
    top10: tuple[tuple[str, float], ...]
    dense_top50: tuple[tuple[str, float], ...]
    sparse_top50: tuple[tuple[str, float], ...]
    candidate_count: int


def _build_sparse(documents: Mapping[str, str], k1: float = 1.5, b: float = 0.75) -> SparseSnapshot:
    if not documents:
        raise ValueError("Cannot build an empty sparse scope")
    memory_ids = tuple(sorted(documents))
    texts = tuple(documents[memory_id] for memory_id in memory_ids)
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50_000,
    )
    doc_tf = vectorizer.fit_transform(texts).tocsc()
    doc_len = np.asarray(doc_tf.sum(axis=1)).flatten().astype(np.float64)
    avg_dl = float(doc_len.mean())
    document_frequency = np.asarray((doc_tf > 0).sum(axis=0)).flatten()
    idf = np.log((len(memory_ids) - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0)
    len_norm = 1.0 - b + b * doc_len / max(avg_dl, 1e-12)
    return SparseSnapshot(memory_ids, texts, vectorizer, doc_tf, doc_len, avg_dl, idf, len_norm)


def _build_dense(embeddings: Mapping[str, Sequence[float]]) -> DenseSnapshot:
    if not embeddings:
        raise ValueError("Cannot build an empty dense scope")
    memory_ids = tuple(sorted(embeddings))
    matrix = np.asarray([embeddings[memory_id] for memory_id in memory_ids], dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Dense embeddings must be a 2-D matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return DenseSnapshot(memory_ids, matrix)


def _bm25_search(snapshot: SparseSnapshot, query: str, top_k: int = 50, k1: float = 1.5) -> list[tuple[str, float]]:
    query_vector = snapshot.vectorizer.transform([query]).tocsc()
    _, query_columns = query_vector.nonzero()
    scores = np.zeros(len(snapshot.memory_ids), dtype=np.float64)
    for column in query_columns:
        start, end = snapshot.doc_tf.indptr[column], snapshot.doc_tf.indptr[column + 1]
        rows = snapshot.doc_tf.indices[start:end]
        data = snapshot.doc_tf.data[start:end]
        tf = np.zeros(len(snapshot.memory_ids), dtype=np.float64)
        tf[rows] = data
        scores += (tf * (k1 + 1.0) / (tf + k1 * snapshot.len_norm + 1e-9)) * snapshot.idf[column]
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), snapshot.memory_ids[index]))
    return [(snapshot.memory_ids[index], float(scores[index])) for index in order[:top_k]]


def _dense_search(snapshot: DenseSnapshot, query_embedding: Sequence[float], top_k: int = 50) -> list[tuple[str, float]]:
    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim != 1 or query.shape[0] != snapshot.normalized_embeddings.shape[1]:
        raise ValueError("Query embedding dimension differs from memory embedding dimension")
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    scores = snapshot.normalized_embeddings @ query
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), snapshot.memory_ids[index]))
    return [(snapshot.memory_ids[index], float(scores[index])) for index in order[:top_k]]


def _zscore(items: Sequence[tuple[str, float]]) -> tuple[dict[str, float], float]:
    if not items:
        return {}, 0.0
    values = np.asarray([score for _, score in items], dtype=np.float64)
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        std = 1.0
    scores = {memory_id: (float(score) - float(values.mean())) / std for memory_id, score in items}
    return scores, min(scores.values())


def zscore_fuse(
    dense: Sequence[tuple[str, float]],
    sparse: Sequence[tuple[str, float]],
    alpha_dense: float = 0.6,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    dense_z, dense_min = _zscore(dense)
    sparse_z, sparse_min = _zscore(sparse)
    candidate_order: list[str] = []
    for memory_id, _ in list(dense) + list(sparse):
        if memory_id not in candidate_order:
            candidate_order.append(memory_id)
    order_index = {memory_id: index for index, memory_id in enumerate(candidate_order)}
    fused = [
        (
            memory_id,
            alpha_dense * dense_z.get(memory_id, dense_min)
            + (1.0 - alpha_dense) * sparse_z.get(memory_id, sparse_min),
        )
        for memory_id in candidate_order
    ]
    fused.sort(key=lambda item: (-item[1], order_index[item[0]]))
    return fused[:top_k]


class ScopedOnlineRetrievalIndex:
    """Thread-safe copy-on-write sparse/dense snapshots keyed by scope."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._scope_locks: dict[str, threading.RLock] = {}
        self._documents: dict[str, dict[str, str]] = {}
        self._embeddings: dict[str, dict[str, np.ndarray]] = {}
        self._sparse: dict[str, SparseSnapshot] = {}
        self._dense: dict[str, DenseSnapshot] = {}
        self._sparse_versions: dict[tuple[str, str], int] = {}
        self._dense_versions: dict[tuple[str, str], int] = {}
        self.duplicate_sparse_updates = 0
        self.duplicate_dense_updates = 0

    def load_scope(
        self,
        scope_id: str,
        documents: Mapping[str, str],
        embeddings: Mapping[str, Sequence[float]],
        version: int = 1,
    ) -> None:
        if set(documents) != set(embeddings):
            raise ValueError("Sparse and dense scope IDs differ")
        with self._lock:
            self._scope_locks.setdefault(scope_id, threading.RLock())
        with self._scope_locks[scope_id]:
            self._documents[scope_id] = dict(documents)
            self._embeddings[scope_id] = {
                key: np.asarray(value, dtype=np.float32) for key, value in embeddings.items()
            }
            self._sparse[scope_id] = _build_sparse(self._documents[scope_id])
            self._dense[scope_id] = _build_dense(self._embeddings[scope_id])
            for memory_id in documents:
                self._sparse_versions[(scope_id, memory_id)] = version
                self._dense_versions[(scope_id, memory_id)] = version

    def upsert_sparse(self, scope_id: str, memory_id: str, document: str, version: int) -> bool:
        with self._lock:
            self._scope_locks.setdefault(scope_id, threading.RLock())
        with self._scope_locks[scope_id]:
            key = (scope_id, memory_id)
            current = self._sparse_versions.get(key, -1)
            if version <= current:
                self.duplicate_sparse_updates += 1
                return False
            documents = dict(self._documents.get(scope_id, {}))
            documents[memory_id] = document
            snapshot = _build_sparse(documents)
            self._documents[scope_id] = documents
            self._sparse[scope_id] = snapshot
            self._sparse_versions[key] = version
            return True

    def upsert_dense(self, scope_id: str, memory_id: str, embedding: Sequence[float], version: int) -> bool:
        with self._lock:
            self._scope_locks.setdefault(scope_id, threading.RLock())
        with self._scope_locks[scope_id]:
            key = (scope_id, memory_id)
            current = self._dense_versions.get(key, -1)
            if version <= current:
                self.duplicate_dense_updates += 1
                return False
            embeddings = dict(self._embeddings.get(scope_id, {}))
            embeddings[memory_id] = np.asarray(embedding, dtype=np.float32)
            snapshot = _build_dense(embeddings)
            self._embeddings[scope_id] = embeddings
            self._dense[scope_id] = snapshot
            self._dense_versions[key] = version
            return True

    def is_visible(self, scope_id: str, memory_id: str, version: int) -> tuple[bool, bool]:
        with self._scope_locks[scope_id]:
            return (
                self._sparse_versions.get((scope_id, memory_id), -1) >= version,
                self._dense_versions.get((scope_id, memory_id), -1) >= version,
            )

    def search(
        self,
        scope_id: str,
        query: str,
        query_embedding: Sequence[float],
        candidate_ids: Iterable[str] | None = None,
        candidate_depth: int = 50,
    ) -> SearchResult:
        with self._scope_locks[scope_id]:
            sparse = self._sparse[scope_id]
            dense = self._dense[scope_id]
        # The snapshots are immutable, so scoring proceeds outside the write lock.
        sparse_ranked = _bm25_search(sparse, query, max(candidate_depth, len(sparse.memory_ids)))
        dense_ranked = _dense_search(dense, query_embedding, max(candidate_depth, len(dense.memory_ids)))
        if candidate_ids is not None:
            allowed = set(candidate_ids)
            sparse_ranked = [item for item in sparse_ranked if item[0] in allowed]
            dense_ranked = [item for item in dense_ranked if item[0] in allowed]
        sparse_top = sparse_ranked[:candidate_depth]
        dense_top = dense_ranked[:candidate_depth]
        top10 = zscore_fuse(dense_top, sparse_top, alpha_dense=0.6, top_k=10)
        return SearchResult(tuple(top10), tuple(dense_top), tuple(sparse_top), len(set(x for x, _ in dense_top + sparse_top)))

    def search_with_timings(
        self,
        scope_id: str,
        query: str,
        query_embedding: Sequence[float],
        candidate_ids: Iterable[str] | None = None,
        candidate_depth: int = 50,
    ) -> tuple[SearchResult, dict[str, float]]:
        """Identical retrieval with non-overlapping scorer/fusion clocks."""
        with self._scope_locks[scope_id]:
            sparse = self._sparse[scope_id]
            dense = self._dense[scope_id]
        allowed = set(candidate_ids) if candidate_ids is not None else None
        start = time.perf_counter_ns()
        sparse_ranked = _bm25_search(sparse, query, max(candidate_depth, len(sparse.memory_ids)))
        if allowed is not None:
            sparse_ranked = [item for item in sparse_ranked if item[0] in allowed]
        sparse_top = sparse_ranked[:candidate_depth]
        after_sparse = time.perf_counter_ns()
        dense_ranked = _dense_search(dense, query_embedding, max(candidate_depth, len(dense.memory_ids)))
        if allowed is not None:
            dense_ranked = [item for item in dense_ranked if item[0] in allowed]
        dense_top = dense_ranked[:candidate_depth]
        after_dense = time.perf_counter_ns()
        top10 = zscore_fuse(dense_top, sparse_top, alpha_dense=0.6, top_k=10)
        after_fusion = time.perf_counter_ns()
        result = SearchResult(tuple(top10), tuple(dense_top), tuple(sparse_top), len(set(x for x, _ in dense_top + sparse_top)))
        return result, {
            "sparse_scoring_ms": (after_sparse - start) / 1e6,
            "dense_scoring_ms": (after_dense - after_sparse) / 1e6,
            "zscore_fusion_ms": (after_fusion - after_dense) / 1e6,
        }
