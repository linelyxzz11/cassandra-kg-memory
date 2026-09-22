#!/usr/bin/env python3
"""Run the canonical LoCoMo Cat1-4 retrievers over one backend projection.

The backend supplies memory fields and the corrected spaCy KG edge view.  Frozen
BGE arrays and all scoring code are deliberately shared, so the experiment
isolates storage/projection equivalence rather than vector-index differences.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
QUESTIONS = ROOT / "data/retrieval_gold/locomo_cat1_4_gold_memory.csv"
MEMORIES = DATA / "locomo_memory_records.csv"
FEATURES = ROOT / "data/frozen_retrieval/p3_memory_features.csv"
EDGES = ROOT / "data/frozen_retrieval/locomo_kg_edges_spacy.csv"
MEM_EMB = DATA / "locomo_memory_bge_large.npy"
QA_EMB = DATA / "locomo_qa_bge_large.npy"
MEM_IDS = DATA / "locomo_memory_ids_bge.txt"
QA_IDS = DATA / "locomo_qa_ids_bge.txt"
DENSE_SCORES = ROOT / "data/frozen_retrieval/frozen_dense_scores_long.csv"
OUTPUT_ROOT = ROOT / "results/backend_equivalence/backend_equivalence_v2/runs"

METHODS = (
    "BM25",
    "Dense-bge",
    "Dense+GlobalKG",
    "RRF_compact",
    "ZScore-Raw",
    "ZScore-RawERK",
)
TOP_K = 10
CANDIDATE_DEPTH = 50
ALPHA_DENSE = 0.6
RRF_K = 10
KG_LAMBDA = 0.2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_rows(rows: Iterable[Any]) -> str:
    encoded = json.dumps(list(rows), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_ids(value: str) -> list[str]:
    return list(dict.fromkeys(part.strip() for part in value.split(";") if part.strip()))


def normalize_node(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def evidence_from_source(value: Any) -> str:
    text = str(value or "").strip()
    return text.rsplit("|", 1)[-1].strip() if "|" in text else ""


@dataclass(frozen=True)
class MemoryProjection:
    memory_id: str
    sample_id: str
    dia_id: str
    text: str
    timestamp: str
    entities: str
    relations: str
    keywords: str

    def rawerk(self) -> str:
        pieces = [self.text]
        for label, value in (
            ("E:", self.entities),
            ("R:", self.relations),
            ("K:", self.keywords),
        ):
            if value.strip():
                pieces.append(f"{label} {value.strip()}")
        return "\n".join(pieces)

    def digest_tuple(self) -> tuple[str, ...]:
        return (
            self.memory_id,
            self.sample_id,
            self.dia_id,
            self.text,
            self.timestamp,
            self.entities,
            self.relations,
            self.keywords,
        )


@dataclass(frozen=True)
class EdgeProjection:
    graph_id: str
    src_id: str
    relation: str
    dst_id: str
    evidence: str

    def digest_tuple(self) -> tuple[str, ...]:
        return (self.graph_id, self.src_id, self.relation, self.dst_id, self.evidence)


class Backend:
    name = "abstract"

    def fetch_memories(self, scope_id: str) -> list[MemoryProjection]:
        raise NotImplementedError

    def fetch_edges(self, scope_id: str) -> list[EdgeProjection]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def config(self) -> dict[str, Any]:
        return {}


class CSVBackend(Backend):
    name = "csv"

    def __init__(self) -> None:
        feature_by_id = {row["memory_id"].strip(): row for row in read_csv(FEATURES)}
        self.by_scope: dict[str, list[MemoryProjection]] = defaultdict(list)
        for row in read_csv(MEMORIES):
            memory_id = row["memory_id"].strip()
            feature = feature_by_id.get(memory_id, {})
            projection = MemoryProjection(
                memory_id=memory_id,
                sample_id=row["sample_id"].strip(),
                dia_id=row["dia_id"].strip(),
                text=row.get("text", "").strip(),
                timestamp=row.get("timestamp", "").strip(),
                entities=feature.get("entities", "").strip(),
                relations=feature.get("relations", "").strip(),
                keywords=feature.get("keywords", "").strip(),
            )
            self.by_scope[projection.sample_id].append(projection)
        self.edges_by_scope: dict[str, list[EdgeProjection]] = defaultdict(list)
        for row in read_csv(EDGES):
            edge = EdgeProjection(
                row["graph_id"].strip(), normalize_node(row["src_id"]),
                row["relation"].strip(), normalize_node(row["dst_id"]),
                row["evidence"].strip(),
            )
            self.edges_by_scope[edge.graph_id].append(edge)

    def fetch_memories(self, scope_id: str) -> list[MemoryProjection]:
        return list(self.by_scope.get(scope_id, []))

    def fetch_edges(self, scope_id: str) -> list[EdgeProjection]:
        return list(self.edges_by_scope.get(scope_id, []))

    def config(self) -> dict[str, Any]:
        return {"memory_source": str(MEMORIES), "feature_source": str(FEATURES), "edge_source": str(EDGES)}


class CassandraBackend(Backend):
    name = "cassandra"

    def __init__(self, host: str, port: int, memory_keyspace: str) -> None:
        from cassandra.cluster import Cluster
        from cassandra.query import dict_factory

        self.host, self.port = host, port
        self.memory_keyspace = memory_keyspace
        self.cluster = Cluster([host], port=port)
        self.memory_session = self.cluster.connect(memory_keyspace)
        self.memory_session.row_factory = dict_factory

    def fetch_memories(self, scope_id: str) -> list[MemoryProjection]:
        query = (
            "SELECT memory_id,dia_id,raw_text,entities,relations,keywords,timestamp "
            "FROM bridge_v2_memory_by_scope WHERE scope_id=%s"
        )
        rows = self.memory_session.execute(query, (scope_id,))
        return [
            MemoryProjection(
                memory_id=str(row["memory_id"]), sample_id=scope_id,
                dia_id=str(row.get("dia_id") or ""),
                text=str(row.get("raw_text") or ""), timestamp=str(row.get("timestamp") or ""),
                entities=str(row.get("entities") or ""), relations=str(row.get("relations") or ""),
                keywords=str(row.get("keywords") or ""),
            )
            for row in rows
        ]

    def fetch_edges(self, scope_id: str) -> list[EdgeProjection]:
        query = (
            "SELECT scope_id,src_id,relation,dst_id,evidence "
            "FROM bridge_v2_edges_by_scope WHERE scope_id=%s"
        )
        rows = self.memory_session.execute(query, (scope_id,))
        return [
            EdgeProjection(
                str(row["scope_id"]), str(row["src_id"]), str(row["relation"]),
                str(row["dst_id"]), str(row.get("evidence") or ""),
            )
            for row in rows
        ]

    def close(self) -> None:
        self.cluster.shutdown()

    def config(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "memory_keyspace": self.memory_keyspace,
                "memory_table": "bridge_v2_memory_by_scope", "edge_table": "bridge_v2_edges_by_scope"}


class Neo4jBackend(Backend):
    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str) -> None:
        from neo4j import GraphDatabase

        if not password:
            raise RuntimeError("Neo4j password is empty; pass --neo4j-password or set NEO4J_PASSWORD")
        self.uri, self.user, self.database = uri, user, database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def fetch_memories(self, scope_id: str) -> list[MemoryProjection]:
        query = """
        MATCH (m:BridgeV2Memory {scope_id:$scope})
        RETURN m.memory_id AS memory_id,m.dia_id AS dia_id,m.raw_text AS raw_text,m.timestamp AS timestamp,
               m.entities AS entities,m.relations AS relations,m.keywords AS keywords
        """
        with self.driver.session(database=self.database) as session:
            rows = list(session.run(query, scope=scope_id))
        return [
            MemoryProjection(
                memory_id=str(row["memory_id"]), sample_id=scope_id,
                dia_id=str(row.get("dia_id") or ""),
                text=str(row.get("raw_text") or ""), timestamp=str(row.get("timestamp") or ""),
                entities=str(row.get("entities") or ""), relations=str(row.get("relations") or ""),
                keywords=str(row.get("keywords") or ""),
            )
            for row in rows
        ]

    def fetch_edges(self, scope_id: str) -> list[EdgeProjection]:
        query = """
        MATCH (s:BridgeV2Entity)-[r:BRIDGE_V2_EDGE {scope_id:$scope}]->(d:BridgeV2Entity)
        RETURN r.scope_id AS graph_id,r.src_id AS src_id,r.relation AS relation,
               r.dst_id AS dst_id,r.evidence AS evidence
        """
        with self.driver.session(database=self.database) as session:
            rows = list(session.run(query, scope=scope_id))
        return [
            EdgeProjection(
                str(row["graph_id"]), str(row["src_id"]), str(row["relation"]),
                str(row["dst_id"]), str(row.get("evidence") or ""),
            )
            for row in rows
        ]

    def close(self) -> None:
        self.driver.close()

    def config(self) -> dict[str, Any]:
        return {"uri": self.uri, "user": self.user, "database": self.database,
                "memory_view": "BridgeV2Memory", "edge_view": "BridgeV2Entity-BRIDGE_V2_EDGE-BridgeV2Entity"}


def memory_dia_id(memory_id: str) -> str:
    marker = "_session_"
    if marker not in memory_id:
        raise ValueError(f"Cannot reconstruct dia_id from memory_id: {memory_id}")
    tail = memory_id.split(marker, 1)[1]
    if "_" not in tail:
        raise ValueError(f"Cannot reconstruct dia_id from memory_id: {memory_id}")
    return tail.split("_", 1)[1]


class BM25:
    def __init__(self) -> None:
        self.vectorizer = CountVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=50000
        )

    def fit(self, documents: list[str]) -> None:
        self.tf = self.vectorizer.fit_transform(documents).tocsc()
        self.n_docs = self.tf.shape[0]
        doc_len = np.asarray(self.tf.sum(axis=1)).ravel()
        avg_dl = float(doc_len.mean())
        df = np.asarray((self.tf > 0).sum(axis=0)).ravel()
        self.idf = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
        self.len_norm = 1.0 - 0.75 + 0.75 * doc_len / max(avg_dl, 1e-12)

    def score(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query]).tocsc()
        scores = np.zeros(self.n_docs, dtype=np.float64)
        _, columns = q.nonzero()
        for column in columns:
            start, end = self.tf.indptr[column], self.tf.indptr[column + 1]
            rows = self.tf.indices[start:end]
            freq = self.tf.data[start:end]
            scores[rows] += self.idf[column] * (
                freq * 2.5 / (freq + 1.5 * self.len_norm[rows])
            )
        return scores


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        std = 1.0
    return (values - float(values.mean())) / std


def top_indices(scores: np.ndarray, memory_ids: list[str], depth: int, ordinal: dict[str, int]) -> list[int]:
    return sorted(
        range(len(memory_ids)), key=lambda index: (-float(scores[index]), ordinal[memory_ids[index]])
    )[:depth]


def top_pairs(scores: np.ndarray, memory_ids: list[str], depth: int, ordinal: dict[str, int]) -> list[tuple[str, float]]:
    return [(memory_ids[index], float(scores[index])) for index in top_indices(scores, memory_ids, depth, ordinal)]


def top_pairs_memory_id(scores: np.ndarray, memory_ids: list[str], depth: int) -> list[tuple[str, float]]:
    """Corrected DenseKG's frozen tie-break is lexical memory_id."""
    indices = sorted(range(len(memory_ids)), key=lambda index: (-float(scores[index]), memory_ids[index]))[:depth]
    return [(memory_ids[index], float(scores[index])) for index in indices]


def zscore_fuse(
    dense: list[tuple[str, float]], bm25: list[tuple[str, float]], ordinal: dict[str, int]
) -> list[tuple[str, float]]:
    del ordinal
    def branch(items: list[tuple[str, float]]) -> tuple[dict[str, float], float]:
        values = np.asarray([score for _, score in items], dtype=np.float64)
        zs = zscore(values)
        mapping = {memory_id: float(zs[index]) for index, (memory_id, _) in enumerate(items)}
        return mapping, min(mapping.values()) if mapping else 0.0

    dense_z, dense_min = branch(dense)
    bm25_z, bm25_min = branch(bm25)
    candidates = list(dict.fromkeys([mid for mid, _ in dense] + [mid for mid, _ in bm25]))
    candidate_order = {memory_id: index for index, memory_id in enumerate(candidates)}
    fused = [
        (mid, ALPHA_DENSE * dense_z.get(mid, dense_min) + (1.0 - ALPHA_DENSE) * bm25_z.get(mid, bm25_min))
        for mid in candidates
    ]
    return sorted(fused, key=lambda item: (-item[1], candidate_order[item[0]]))[:CANDIDATE_DEPTH]


def rrf_fuse(
    dense: list[tuple[str, float]], sparse: list[tuple[str, float]]
) -> list[tuple[str, float]]:
    dense_rank = {mid: rank for rank, (mid, _) in enumerate(dense, 1)}
    sparse_rank = {mid: rank for rank, (mid, _) in enumerate(sparse, 1)}
    rows = []
    for mid in set(dense_rank) | set(sparse_rank):
        score = 0.0
        best = 10**9
        if mid in dense_rank:
            score += ALPHA_DENSE / (RRF_K + dense_rank[mid])
            best = min(best, dense_rank[mid])
        if mid in sparse_rank:
            score += (1.0 - ALPHA_DENSE) / (RRF_K + sparse_rank[mid])
            best = min(best, sparse_rank[mid])
        rows.append((mid, score, best))
    rows.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [(mid, score) for mid, score, _ in rows[:CANDIDATE_DEPTH]]


def build_degree_prior(memories: list[MemoryProjection], edges: list[EdgeProjection]) -> tuple[np.ndarray, int]:
    by_turn: dict[str, list[int]] = defaultdict(list)
    for index, memory in enumerate(memories):
        by_turn[memory.dia_id].append(index)
    mapped: dict[int, list[tuple[str, str]]] = defaultdict(list)
    degree: Counter[str] = Counter()
    assignments = 0
    for edge in edges:
        indices = by_turn.get(edge.evidence, [])
        if not indices:
            continue
        src, dst = normalize_node(edge.src_id), normalize_node(edge.dst_id)
        if not src or not dst:
            continue
        degree[src] += 1
        degree[dst] += 1
        for index in indices:
            mapped[index].append((src, dst))
            assignments += 1
    prior = np.zeros(len(memories), dtype=np.float64)
    for index, assigned in mapped.items():
        salience = [math.log1p(degree[src] + degree[dst]) for src, dst in assigned]
        prior[index] = math.log1p(len(assigned)) + sum(salience) / len(salience)
    return prior, assignments


def metrics(gold_ids: list[str], ranking: list[str]) -> dict[str, float | None]:
    if not gold_ids:
        return {name: None for name in ("mrr_at_10", "recall_at_10", "ndcg_at_10", "hit_at_1", "hit_at_5", "hit_at_10")}
    gold = set(gold_ids)
    relevant = [rank for rank, mid in enumerate(ranking[:10], 1) if mid in gold]
    first = min(relevant) if relevant else None
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), 10) + 1))
    return {
        "mrr_at_10": 0.0 if first is None else 1.0 / first,
        "recall_at_10": len(set(ranking[:10]) & gold) / len(gold),
        "ndcg_at_10": dcg / idcg,
        "hit_at_1": float(first is not None and first <= 1),
        "hit_at_5": float(first is not None and first <= 5),
        "hit_at_10": float(first is not None and first <= 10),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "csv":
        return CSVBackend()
    if args.backend == "cassandra":
        return CassandraBackend(args.cassandra_host, args.cassandra_port, args.cassandra_memory_keyspace)
    return Neo4jBackend(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("csv", "cassandra", "neo4j"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cassandra-host", default="127.0.0.1")
    parser.add_argument("--cassandra-port", type=int, default=9042)
    parser.add_argument("--cassandra-memory-keyspace", default="kg_memory")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    parser.add_argument("--neo4j-database", default="neo4j")
    args = parser.parse_args()

    # Local credentials are deliberately loaded at runtime and never emitted.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    if not args.neo4j_password:
        args.neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    question_rows = read_csv(QUESTIONS)
    if len(question_rows) != 1540:
        raise RuntimeError(f"Expected 1540 canonical Cat1-4 questions, got {len(question_rows)}")
    scope_ids = list(dict.fromkeys(row["conversation_id"].strip() for row in question_rows))
    frozen_memory_ids = [line.strip() for line in MEM_IDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    frozen_query_ids = [line.strip() for line in QA_IDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Ranking ties in the canonical runs use memory_records.csv corpus order.
    # The embedding ID order is an independent lookup order and must not leak
    # into tie-breaking.
    corpus_ids = [row["memory_id"].strip() for row in read_csv(MEMORIES)]
    ordinal = {memory_id: index for index, memory_id in enumerate(corpus_ids)}
    memory_index = {memory_id: index for index, memory_id in enumerate(frozen_memory_ids)}
    query_index = {query_id: index for index, query_id in enumerate(frozen_query_ids)}
    mem_emb = np.load(MEM_EMB).astype(np.float64)
    qa_emb = np.load(QA_EMB).astype(np.float64)
    mem_emb /= np.maximum(np.linalg.norm(mem_emb, axis=1, keepdims=True), 1e-12)
    qa_emb /= np.maximum(np.linalg.norm(qa_emb, axis=1, keepdims=True), 1e-12)

    # The published Dense/RRF/ZScore artifacts were frozen from this score
    # cache (six-decimal scores).  Recomputing from the NPY arrays changes a
    # handful of near-ties, so use the cache for those canonical methods.
    # Corrected Dense+GlobalKG intentionally keeps its separately frozen
    # full-precision NPY scoring path.
    wanted_queries = {row["query_id"].strip() for row in question_rows}
    frozen_dense_scores: dict[str, dict[str, float]] = defaultdict(dict)
    with DENSE_SCORES.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            query_id = row["query_id"].strip()
            if query_id in wanted_queries:
                frozen_dense_scores[query_id][row["memory_id"].strip()] = float(row["score"])

    backend = make_backend(args)
    out_dir = args.output_root / backend.name
    top10_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    digest_output: list[dict[str, Any]] = []
    try:
        memories_by_scope: dict[str, list[MemoryProjection]] = {}
        priors_by_scope: dict[str, np.ndarray] = {}
        for scope_id in scope_ids:
            fetched = backend.fetch_memories(scope_id)
            if not fetched:
                raise RuntimeError(f"Backend {backend.name} returned no memories for {scope_id}")
            if len({memory.memory_id for memory in fetched}) != len(fetched):
                raise RuntimeError(f"Duplicate memory_id in backend projection for {scope_id}")
            unknown = sorted(memory.memory_id for memory in fetched if memory.memory_id not in ordinal)
            if unknown:
                raise RuntimeError(f"Backend {backend.name} has IDs outside frozen corpus for {scope_id}: {unknown[:5]}")
            memories = sorted(fetched, key=lambda memory: ordinal[memory.memory_id])
            edges = sorted(backend.fetch_edges(scope_id), key=lambda edge: edge.digest_tuple())
            if not edges:
                raise RuntimeError(f"Backend {backend.name} returned no corrected KG edges for {scope_id}")
            prior, assignments = build_degree_prior(memories, edges)
            if assignments == 0:
                raise RuntimeError(f"Backend {backend.name} corrected KG edges map to no memories for {scope_id}")
            memories_by_scope[scope_id] = memories
            priors_by_scope[scope_id] = prior
            digest_output.append({
                "backend": backend.name, "scope_id": scope_id, "candidate_count": len(memories),
                "candidate_sha256": digest_rows(sorted(memory.memory_id for memory in memories)),
                "projection_sha256": digest_rows(memory.digest_tuple() for memory in memories),
                "corpus_order_sha256": digest_rows(memory.memory_id for memory in memories),
                "edge_row_count": len(edges), "edge_assignment_count": assignments,
                "edge_sha256": digest_rows(edge.digest_tuple() for edge in edges),
            })

        query_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in question_rows:
            query_groups[row["conversation_id"].strip()].append(row)

        for scope_id in scope_ids:
            memories = memories_by_scope[scope_id]
            memory_ids = [memory.memory_id for memory in memories]
            indices = np.asarray([memory_index[mid] for mid in memory_ids], dtype=np.int64)
            raw_bm25, erk_bm25 = BM25(), BM25()
            raw_bm25.fit([memory.text for memory in memories])
            erk_bm25.fit([memory.rawerk() for memory in memories])
            prior_z = zscore(priors_by_scope[scope_id])

            for question in query_groups[scope_id]:
                query_id = question["query_id"].strip()
                if query_id not in query_index:
                    raise RuntimeError(f"Missing frozen query embedding: {query_id}")
                densekg_raw_scores = mem_emb[indices] @ qa_emb[query_index[query_id]]
                score_map = frozen_dense_scores.get(query_id, {})
                missing_dense = [memory_id for memory_id in memory_ids if memory_id not in score_map]
                if missing_dense:
                    raise RuntimeError(
                        f"Frozen dense cache misses {len(missing_dense)} backend candidates for {query_id}"
                    )
                dense_scores = np.asarray([score_map[memory_id] for memory_id in memory_ids], dtype=np.float64)
                raw_scores = raw_bm25.score(question["question"])
                erk_scores = erk_bm25.score(question["question"])
                dense50 = top_pairs(dense_scores, memory_ids, CANDIDATE_DEPTH, ordinal)
                raw50 = top_pairs(raw_scores, memory_ids, CANDIDATE_DEPTH, ordinal)
                erk50 = top_pairs(erk_scores, memory_ids, CANDIDATE_DEPTH, ordinal)
                raw100 = top_pairs(raw_scores, memory_ids, 100, ordinal)
                dense100 = top_pairs(dense_scores, memory_ids, 100, ordinal)
                densekg_scores = zscore(densekg_raw_scores) + KG_LAMBDA * prior_z
                densekg100 = top_pairs_memory_id(densekg_scores, memory_ids, 100)
                rrf_pool = list(dict.fromkeys(
                    [memory_id for memory_id, _ in dense50]
                    + [memory_id for memory_id, _ in erk50[:TOP_K]]
                ))
                zraw_pool = list(dict.fromkeys(
                    [memory_id for memory_id, _ in dense50]
                    + [memory_id for memory_id, _ in raw50]
                ))
                zerk_pool = list(dict.fromkeys(
                    [memory_id for memory_id, _ in dense50]
                    + [memory_id for memory_id, _ in erk50]
                ))
                rankings = {
                    "BM25": top_pairs(raw_scores, memory_ids, TOP_K, ordinal),
                    "Dense-bge": top_pairs(dense_scores, memory_ids, TOP_K, ordinal),
                    "Dense+GlobalKG": top_pairs_memory_id(densekg_scores, memory_ids, TOP_K),
                    # The frozen BM25(compact) artifact retained Top-10 only;
                    # canonical RRF therefore fuses Dense@50 with compact@10.
                    "RRF_compact": rrf_fuse(dense50, erk50[:TOP_K])[:TOP_K],
                    "ZScore-Raw": zscore_fuse(dense50, raw50, ordinal)[:TOP_K],
                    "ZScore-RawERK": zscore_fuse(dense50, erk50, ordinal)[:TOP_K],
                }
                gold_ids = parse_ids(question.get("gold_memory_ids", ""))
                diagnostic_pools = {
                    "BM25": [memory_id for memory_id, _ in raw100],
                    "Dense-bge": [memory_id for memory_id, _ in dense100],
                    "Dense+GlobalKG": [memory_id for memory_id, _ in densekg100],
                    "RRF_compact": rrf_pool,
                    "ZScore-Raw": zraw_pool,
                    "ZScore-RawERK": zerk_pool,
                }
                for method in METHODS:
                    ranking = rankings[method]
                    ids = [memory_id for memory_id, _ in ranking]
                    for rank, (memory_id, score) in enumerate(ranking, 1):
                        top10_rows.append({
                            "backend": backend.name, "method": method, "query_id": query_id,
                            "category": question["category"], "split": question["split"],
                            "rank": rank, "memory_id": memory_id, "score": repr(float(score)),
                        })
                    metric_rows.append({
                        "backend": backend.name, "method": method, "query_id": query_id,
                        "category": question["category"], "split": question["split"],
                        "candidate_count": len(memories), "gold_count": len(gold_ids),
                        **metrics(gold_ids, ids), "top10_ids": ";".join(ids),
                    })
                    pool_rows.append({
                        "backend": backend.name,
                        "method": method,
                        "query_id": query_id,
                        "category": question["category"],
                        "pool_definition": (
                            "full_rank_top100" if method in {"BM25", "Dense-bge", "Dense+GlobalKG"}
                            else "fusion_input_union"
                        ),
                        "pool_size": len(diagnostic_pools[method]),
                        "pool_ids": ";".join(diagnostic_pools[method]),
                    })
    finally:
        backend.close()

    metric_names = ("mrr_at_10", "recall_at_10", "ndcg_at_10", "hit_at_1", "hit_at_5", "hit_at_10")
    summary_rows = []
    for method in METHODS:
        selected = [row for row in metric_rows if row["method"] == method]
        evaluable = [row for row in selected if row["gold_count"] > 0]
        summary_rows.append({
            "backend": args.backend, "method": method, "n_queries": len(selected), "n_evaluable": len(evaluable),
            **{name: sum(float(row[name]) for row in evaluable) / len(evaluable) for name in metric_names},
        })

    expected_metric_rows = len(question_rows) * len(METHODS)
    expected_top10_rows = expected_metric_rows * TOP_K
    if len(metric_rows) != expected_metric_rows or len(top10_rows) != expected_top10_rows:
        raise RuntimeError(
            f"Incomplete output: metrics={len(metric_rows)}/{expected_metric_rows}, "
            f"top10={len(top10_rows)}/{expected_top10_rows}"
        )

    write_csv(out_dir / "top10.csv", top10_rows,
              ["backend", "method", "query_id", "category", "split", "rank", "memory_id", "score"])
    write_csv(out_dir / "per_query_metrics.csv", metric_rows,
              ["backend", "method", "query_id", "category", "split", "candidate_count", "gold_count", *metric_names, "top10_ids"])
    write_csv(out_dir / "summary.csv", summary_rows,
              ["backend", "method", "n_queries", "n_evaluable", *metric_names])
    write_csv(out_dir / "input_scope_digests.csv", digest_output,
              ["backend", "scope_id", "candidate_count", "candidate_sha256", "projection_sha256", "corpus_order_sha256", "edge_row_count", "edge_assignment_count", "edge_sha256"])
    write_csv(out_dir / "diagnostic_pools.csv", pool_rows,
              ["backend", "method", "query_id", "category", "pool_definition", "pool_size", "pool_ids"])
    manifest = {
        "experiment": "P7-A backend bridge v2", "backend": backend.name,
        "scope": {"name": "canonical_cat1_4", "n_queries": len(question_rows), "n_scopes": len(scope_ids)},
        "methods": list(METHODS),
        "protocol": {"top_k": TOP_K, "candidate_depth": CANDIDATE_DEPTH, "alpha_dense": ALPHA_DENSE,
                     "rrf_k": RRF_K, "rrf_sparse_branch": "BM25-RawERK", "rrf_depths": "Dense@50+BM25-RawERK@10",
                     "zscore_missing": "branch_min",
                     "densekg_prior": "degree_centrality", "densekg_lambda": KG_LAMBDA,
                     "tie_breaks": {"base": "frozen corpus ordinal", "rrf": "score,best_branch_rank,memory_id"}},
        "backend_config": backend.config(),
        "inputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in (QUESTIONS, MEMORIES, FEATURES, EDGES, MEM_EMB, QA_EMB, MEM_IDS, QA_IDS, DENSE_SCORES)},
        "outputs": ["top10.csv", "per_query_metrics.csv", "summary.csv", "input_scope_digests.csv", "diagnostic_pools.csv"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"backend": backend.name, "output": str(out_dir), "queries": len(question_rows), "methods": len(METHODS)}))


if __name__ == "__main__":
    main()
