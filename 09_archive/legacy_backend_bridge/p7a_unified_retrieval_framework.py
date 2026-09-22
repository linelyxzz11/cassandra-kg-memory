#!/usr/bin/env python3
"""
P7-A Backend Retrieval Equivalence Validation — unified framework.
One scorer, three backends (CSV / Cassandra / Neo4j).
"""
from __future__ import annotations
import csv, hashlib, json, logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────
# 1. LogicalMemoryRecord — frozen canonical representation
# ──────────────────────────────────────────────────────────────────

@dataclass
class LogicalMemoryRecord:
    memory_id: str
    conversation_id: str
    raw_text: str
    erk_text: str          # text + entity + relation + keyword concatenated
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    timestamp: str = ""
    embedding_id: Optional[int] = None
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    # (src_entity, relation, tgt_memory_id or tgt_entity)

# ──────────────────────────────────────────────────────────────────
# 2. Unified Backend Adapter Interface
# ──────────────────────────────────────────────────────────────────

class BackendAdapter(ABC):
    """All retrieval/scoring code calls this interface. Only one copy."""
    name: str = "abstract"

    @abstractmethod
    def list_memories(self, scope_id: str) -> list[str]:
        """Return all memory_ids for a given conversation/sample scope."""
        ...

    @abstractmethod
    def get_raw_records(self, scope_id: str) -> dict[str, str]:
        """{memory_id: raw_text}"""
        ...

    @abstractmethod
    def get_erk_records(self, scope_id: str) -> dict[str, str]:
        """{memory_id: erk_text}"""
        ...

    @abstractmethod
    def get_memory_ids(self, scope_id: str) -> list[str]:
        """Ordered memory_id list for the scope."""
        ...

    @abstractmethod
    def get_embeddings(self, memory_ids: list[str]) -> dict[str, list[float]]:
        """Precomputed frozen embeddings keyed by memory_id."""
        ...

    @abstractmethod
    def get_triples(self, scope_id: str) -> dict[str, list[tuple[str, str, str]]]:
        """{memory_id: [(entity, relation, target_memory_id)]}"""
        ...

    @abstractmethod
    def get_edges_by_src(self, scope_id: str, src_entity: str) -> list[tuple[str, str, str]]:
        """All KG edges from this entity within the scope."""
        ...

    @abstractmethod
    def get_edges_by_src_relation(self, scope_id: str, src_entity: str, relation: str) -> list[tuple[str, str, str]]:
        """Filtered KG edges."""
        ...

# ──────────────────────────────────────────────────────────────────
# 3. CSV Backend (canonical reference)
# ──────────────────────────────────────────────────────────────────

class CSVBackend(BackendAdapter):
    name = "CSV"

    def __init__(self, root: Path):
        self.root = root
        self.memories: dict[str, dict] = {}
        self.features: dict[str, dict] = {}
        self.embeddings_cache: dict[str, list[float]] = {}
        self._load()

    def _load(self):
        # memory_records.csv
        records_paths = [
            self.root / "01_data" / "locomo_memory_records.csv",
            # fallback: project root relative
        ]
        for p in records_paths:
            if p.exists():
                with open(p, encoding="utf-8-sig") as f:
                    for r in csv.DictReader(f):
                        mid = r["memory_id"].strip()
                        self.memories[mid] = {
                            "text": r.get("text", "").strip(),
                            "speaker": r.get("speaker", "").strip(),
                            "timestamp": r.get("timestamp", "").strip(),
                            "session_id": r.get("session_id", "").strip(),
                            "sample_id": mid.split("_session_")[0] if "_session_" in mid else "",
                        }
                break

        # memory_features.csv
        feat_paths = [
            self.root / "scripts" / "experiments" / "artifacts" / "p3_memory_features.csv",
        ]
        for p in feat_paths:
            if p.exists():
                with open(p, encoding="utf-8-sig") as f:
                    for r in csv.DictReader(f):
                        mid = r["memory_id"].strip()
                        self.features[mid] = {
                            "entity": r.get("entity", ""),
                            "relation": r.get("relation", ""),
                            "keyword": r.get("keyword", ""),
                        }
                break

    def list_memories(self, scope_id: str) -> list[str]:
        return [mid for mid, m in self.memories.items() if m.get("sample_id", "") == scope_id]

    def get_memory_ids(self, scope_id: str) -> list[str]:
        return sorted(self.list_memories(scope_id))

    def get_raw_records(self, scope_id: str) -> dict[str, str]:
        return {mid: self.memories[mid]["text"] for mid in self.list_memories(scope_id)}

    def get_erk_records(self, scope_id: str) -> dict[str, str]:
        out = {}
        for mid in self.list_memories(scope_id):
            raw = self.memories.get(mid, {}).get("text", "")
            feats = self.features.get(mid, {})
            erk = " ".join(filter(None, [
                raw,
                feats.get("entity", ""),
                feats.get("relation", ""),
                feats.get("keyword", ""),
            ]))
            out[mid] = erk
        return out

    def get_embeddings(self, memory_ids: list[str]) -> dict[str, list[float]]:
        # Load from frozen embeddings file
        # Placeholder — requires loading .npy or pickle
        if not self.embeddings_cache:
            emb_path = self.root / "scripts" / "experiments" / "artifacts" / "frozen_memory_embeddings.npy"
            if emb_path.exists():
                import numpy as np
                data = np.load(str(emb_path), allow_pickle=True)
                # Assumes data is dict or structured array — adapt to actual format
        return {mid: self.embeddings_cache.get(mid, []) for mid in memory_ids}

    def get_triples(self, scope_id: str) -> dict[str, list[tuple[str, str, str]]]:
        # Build triples from entity/relation features
        # This is a simplification — real triples need KG graph traversal
        out = defaultdict(list)
        for mid, feats in self.features.items():
            ent = feats.get("entity", "")
            rel = feats.get("relation", "")
            if ent and rel:
                out[mid].append((ent, rel, mid))
        return dict(out)

    def get_edges_by_src(self, scope_id: str, src_entity: str) -> list[tuple[str, str, str]]:
        all_triples = self.get_triples(scope_id)
        edges = []
        for mid, triples in all_triples.items():
            for e, r, t in triples:
                if e.lower() == src_entity.lower():
                    edges.append((e, r, t))
        return edges

    def get_edges_by_src_relation(self, scope_id: str, src_entity: str, relation: str) -> list[tuple[str, str, str]]:
        return [(e, r, t) for e, r, t in self.get_edges_by_src(scope_id, src_entity)
                if r.lower() == relation.lower()]

# ──────────────────────────────────────────────────────────────────
# 4. Cassandra Backend (placeholder)
# ──────────────────────────────────────────────────────────────────

class CassandraBackend(BackendAdapter):
    name = "Cassandra"
    def __init__(self, contact_points: list[str], keyspace: str):
        # from cassandra.cluster import Cluster
        # self.session = Cluster(contact_points).connect(keyspace)
        ...

    def list_memories(self, scope_id: str) -> list[str]:
        # SELECT memory_id FROM kg_memory_table WHERE conversation_id = %s
        ...

    def get_raw_records(self, scope_id: str) -> dict[str, str]:
        # SELECT memory_id, raw_text FROM kg_memory_table WHERE conversation_id = %s
        ...

    def get_erk_records(self, scope_id: str) -> dict[str, str]:
        # SELECT memory_id, raw_text, entities, relations, keywords FROM kg_memory_table WHERE conversation_id = %s
        ...

    def get_memory_ids(self, scope_id: str) -> list[str]:
        ...

    def get_embeddings(self, memory_ids: list[str]) -> dict[str, list[float]]:
        # SELECT memory_id, embedding FROM memory_embeddings WHERE memory_id IN (...)
        ...

    def get_triples(self, scope_id: str) -> dict[str, list[tuple[str, str, str]]]:
        # SELECT src_entity, relation, tgt_memory_id FROM kg_triples WHERE scope_id = %s
        ...

    def get_edges_by_src(self, scope_id: str, src_entity: str) -> list[tuple[str, str, str]]:
        # SELECT src, rel, tgt FROM kg_triples WHERE scope_id = %s AND src = %s
        ...

    def get_edges_by_src_relation(self, scope_id: str, src_entity: str, relation: str) -> list[tuple[str, str, str]]:
        # SELECT src, rel, tgt FROM kg_triples WHERE scope_id = %s AND src = %s AND rel = %s
        ...

# ──────────────────────────────────────────────────────────────────
# 5. Neo4j Backend (placeholder)
# ──────────────────────────────────────────────────────────────────

class Neo4jBackend(BackendAdapter):
    name = "Neo4j"
    def __init__(self, uri: str, user: str, password: str):
        # from neo4j import GraphDatabase
        # self.driver = GraphDatabase.driver(uri, auth=(user, password))
        ...

    def list_memories(self, scope_id: str) -> list[str]:
        # MATCH (m:Memory {scope_id: $s}) RETURN m.memory_id
        ...

    def get_raw_records(self, scope_id: str) -> dict[str, str]:
        # MATCH (m:Memory {scope_id: $s}) RETURN m.memory_id, m.raw_text
        ...

    def get_erk_records(self, scope_id: str) -> dict[str, str]:
        ...

    def get_memory_ids(self, scope_id: str) -> list[str]:
        ...

    def get_embeddings(self, memory_ids: list[str]) -> dict[str, list[float]]:
        ...

    def get_triples(self, scope_id: str) -> dict[str, list[tuple[str, str, str]]]:
        # MATCH (m:Memory {scope_id: $s})-[r:RELATES]->(t:Memory) RETURN m.memory_id, type(r), t.memory_id
        ...

    def get_edges_by_src(self, scope_id: str, src_entity: str) -> list[tuple[str, str, str]]:
        ...

    def get_edges_by_src_relation(self, scope_id: str, src_entity: str, relation: str) -> list[tuple[str, str, str]]:
        ...

# ──────────────────────────────────────────────────────────────────
# 6. Unified Retrieval Scorer (single copy, all backends)
# ──────────────────────────────────────────────────────────────────

class UnifiedRetrievalScorer:
    """Calls only BackendAdapter methods. Platform-agnostic."""

    def __init__(self, backend: BackendAdapter, params: dict):
        self.backend = backend
        self.params = params
        self.alpha = params.get("alpha", 0.6)
        self.top_k = params.get("top_k", 10)

    # ---- BM25 ----
    def bm25_rank(self, scope_id: str, query_text: str, variant: str = "Raw") -> dict[str, float]:
        from sklearn.feature_extraction.text import CountVectorizer
        records = self.backend.get_raw_records(scope_id) if variant == "Raw" else self.backend.get_erk_records(scope_id)
        memory_ids = list(records.keys())
        if len(memory_ids) == 0:
            return {}
        texts = [records[mid] for mid in memory_ids]
        vec = CountVectorizer(min_df=1)
        dtm = vec.fit_transform(texts)
        q_vec = vec.transform([query_text])
        scores = (dtm @ q_vec.T).toarray().flatten()
        return {mid: float(scores[i]) for i, mid in enumerate(memory_ids)}

    # ---- Dense ----
    def dense_rank(self, scope_id: str, query_embedding: list[float]) -> dict[str, float]:
        import numpy as np
        memory_ids = self.backend.list_memories(scope_id)
        if len(memory_ids) == 0:
            return {}
        embeddings = self.backend.get_embeddings(memory_ids)
        q = np.array(query_embedding)
        scores = {}
        for mid in memory_ids:
            e = np.array(embeddings.get(mid, []))
            if len(e) > 0:
                scores[mid] = float(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-12))
        return scores

    # ---- GlobalKG re-rank ----
    def global_kg_boost(self, scope_id: str, query_text: str, dense_scores: dict[str, float],
                        w_kg: float = 0.1) -> dict[str, float]:
        # Simplified: extract entity from query → find KG edges → boost connected memories
        triples = self.backend.get_triples(scope_id)
        connected = set()
        for mid, tlist in triples.items():
            for src, rel, tgt in tlist:
                if src.lower() in query_text.lower():
                    connected.add(mid)
                    connected.add(tgt)
        result = {}
        for mid, score in dense_scores.items():
            boost = 1.0 if mid in connected else 0.0
            result[mid] = score * (1 - w_kg) + boost * w_kg
        return result

    # ---- WRRF fusion ----
    def wrrf_fusion(self, rankings_a: dict[str, float], rankings_b: dict[str, float],
                    alpha: float = 0.6, k: int = 10) -> dict[str, float]:
        ra = {mid: i + 1 for i, (mid, _) in enumerate(
            sorted(rankings_a.items(), key=lambda x: -x[1]))}
        rb = {mid: i + 1 for i, (mid, _) in enumerate(
            sorted(rankings_b.items(), key=lambda x: -x[1]))}
        result = {}
        for mid in set(ra) | set(rb):
            result[mid] = alpha / (k + ra.get(mid, 999)) + (1 - alpha) / (k + rb.get(mid, 999))
        return result

    # ---- ZScore fusion ----
    def zscore_fusion(self, scores_a: dict[str, float], scores_b: dict[str, float],
                      alpha: float = 0.6) -> dict[str, float]:
        import numpy as np
        def z_norm(d):
            vals = list(d.values())
            mu = np.mean(vals) if vals else 0
            std = np.std(vals) if vals else 1.0
            if std == 0:
                std = 1.0
            return {k: (v - mu) / std for k, v in d.items()}
        za = z_norm(scores_a)
        zb = z_norm(scores_b)
        result = {}
        for mid in set(za) | set(zb):
            result[mid] = alpha * za.get(mid, -3.0) + (1 - alpha) * zb.get(mid, -3.0)
        return result

    # ---- Top-K extraction ----
    @staticmethod
    def top_k(scores: dict[str, float], k: int = 10) -> list[tuple[str, float]]:
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]

# ──────────────────────────────────────────────────────────────────
# 7. Parity Comparator
# ──────────────────────────────────────────────────────────────────

@dataclass
class PerQueryParity:
    query_id: str
    category: str
    method: str
    backend: str
    candidate_count: int
    top10_exact_match: bool
    top10_jaccard: float
    rank_biased_overlap: float
    max_abs_diff: float
    mrr: float
    mrr_diff: float
    r10: float
    r10_diff: float
    first_diff_rank: int
    diagnosis: str

class ParityComparator:
    @staticmethod
    def compare(csv_result: dict, backend_result: dict, gold_evidence: set[str] = None) -> PerQueryParity:
        csv_top10 = csv_result.get("top10", [])
        be_top10 = backend_result.get("top10", [])

        # Top-10 exact match (order-sensitive)
        exact = csv_top10 == be_top10

        # Jaccard
        cs = set(csv_top10)
        bs = set(be_top10)
        jaccard = len(cs & bs) / max(len(cs | bs), 1)

        # Rank-biased overlap
        rbo = ParityComparator._rbo(csv_top10, be_top10)

        # Score differences
        csv_scores = csv_result.get("scores", {})
        be_scores = backend_result.get("scores", {})
        max_diff = max(abs(csv_scores.get(mid, 0) - be_scores.get(mid, 0))
                       for mid in set(csv_scores) | set(be_scores)) if csv_scores or be_scores else 0.0

        # First differing rank
        first_diff = 0
        for i in range(min(len(csv_top10), len(be_top10))):
            if csv_top10[i] != be_top10[i]:
                first_diff = i + 1
                break

        # Diagnosis
        if exact:
            diag = "exact_match"
        elif jaccard == 1.0:
            diag = "order_diff_same_set"
        elif jaccard > 0.9:
            diag = "near_match"
        else:
            diag = "significant_divergence"

        return PerQueryParity(
            query_id=csv_result.get("query_id", ""),
            category=csv_result.get("category", ""),
            method=csv_result.get("method", ""),
            backend=backend_result.get("backend", ""),
            candidate_count=csv_result.get("candidate_count", 0),
            top10_exact_match=exact,
            top10_jaccard=jaccard,
            rank_biased_overlap=rbo,
            max_abs_diff=max_diff,
            mrr=0.0, mrr_diff=0.0,
            r10=0.0, r10_diff=0.0,
            first_diff_rank=first_diff,
            diagnosis=diag,
        )

    @staticmethod
    def _rbo(a: list, b: list, p: float = 0.98) -> float:
        """Rank-Biased Overlap (simplified)."""
        sa, sb = set(), set()
        overlap_sum = 0.0
        weight_sum = 0.0
        for d in range(1, min(len(a), len(b)) + 1):
            sa.add(a[d - 1])
            sb.add(b[d - 1])
            overlap = len(sa & sb) / d
            w = (1 - p) * (p ** (d - 1))
            overlap_sum += overlap * w
            weight_sum += w
        return overlap_sum / weight_sum if weight_sum > 0 else 0.0

# ──────────────────────────────────────────────────────────────────
# 8. Utility functions
# ──────────────────────────────────────────────────────────────────

def sha256_of_list(items: list[str]) -> str:
    return hashlib.sha256("|".join(sorted(str(i) for i in items)).encode()).hexdigest()

def sha256_of_dict(d: dict, sort_keys: bool = True) -> str:
    items = sorted(d.items()) if sort_keys else list(d.items())
    return hashlib.sha256(json.dumps({str(k): str(v)[:200] for k, v in items}).encode()).hexdigest()

# ──────────────────────────────────────────────────────────────────
# 9. Main execution skeleton
# ──────────────────────────────────────────────────────────────────

def run_equivalence_experiment(
    csv_backend: CSVBackend,
    cassandra_backend: Optional[CassandraBackend],
    neo4j_backend: Optional[Neo4jBackend],
    questions: list[dict],
    methods: list[str],
    output_dir: Path,
):
    scorer = UnifiedRetrievalScorer(csv_backend, {"alpha": 0.6, "top_k": 10})
    results = []

    for q in questions:
        qid = q["query_id"]
        scope = q["sample_id"]  # conversation scope
        q_text = q["question"]

        # ---- 1. Verify input data parity ----
        for be_name, be in [("Cassandra", cassandra_backend), ("Neo4j", neo4j_backend)]:
            if be is None:
                continue
            csv_mids = set(csv_backend.list_memories(scope))
            be_mids = set(be.list_memories(scope))
            # Gate check
            if csv_mids != be_mids:
                logging.warning(f"Input mismatch: {qid} {be_name} memory_id sets differ")

        # ---- 2. Run each method through unified scorer ----
        for method in methods:
            # Placeholder: run actual method scoring
            pass

    return results

if __name__ == "__main__":
    print("P7-A Backend Equivalence Framework Loaded.")
    print("Requires: Cassandra instance, Neo4j instance, frozen embeddings.")
    print("See: BACKEND_EQUIVALENCE_REPORT.md for usage.")
