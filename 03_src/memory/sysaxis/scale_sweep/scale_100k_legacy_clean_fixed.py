"""
100K legacy clean scale point for Cassandra-KG vs Neo4j.

Pipeline:
  1) CSV stats
  2) Idempotent clean read-graph import to Cassandra + Neo4j
  3) Import guards against CSV
  4) Manifest rewrite + expected hash recompute
  5) 256-query semantic gate
  6) 2-trial smoke with real 10% writes
  7) Optional formal 20 trials with --formal
  8) After guards + summary

Default behavior is SAFE: run through smoke and stop.
Run formal trials only after smoke passes:
  python scale_100k_legacy_clean_fixed.py --formal

Assumptions:
  - Cassandra at 127.0.0.1:9042, keyspace ai_memory
  - Neo4j at bolt://127.0.0.1:7687, auth neo4j/password123
  - Project root: D:/memorytable/cassandra-kg-memory unless overridden
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from cassandra.cluster import Cluster
from cassandra.query import PreparedStatement
from neo4j import GraphDatabase

# ----------------------------
# Configuration
# ----------------------------

READ_GRAPH_ID = "sysaxis_100K_legacy_clean_20260709"
WRITE_GRAPH_BASE = "sysaxis_100K_legacy_clean_write_20260709"
SCALE_LABEL = "100K_legacy_clean"
GRAPH_TYPE = "legacy_synthetic_rebuilt_from_c1_source_100k"
CREATED_AT = datetime(2026, 7, 9, 0, 0, 0, tzinfo=timezone.utc)

CLIENTS = 32
HOP = 2
FANOUT = 20
WRITE_RATIO = 0.10
FRONTIER_WORKERS = 16
IMPORT_WORKERS = 64
GUARD_WORKERS = 128
FETCH_SIZE = 1000

SMOKE_SECONDS = 20
MEASURE_SECONDS = 45
WARMUP_SECONDS = 15
REPEATS = 5

CASSANDRA_HOSTS = ["127.0.0.1"]
CASSANDRA_PORT = 9042
CASSANDRA_KEYSPACE = "ai_memory"
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "password123")

# UUID v1 timestamp is 100ns intervals since 1582-10-15.
_UUID_EPOCH_100NS = 0x01B21DD213814000
_FIXED_UUID_TIME_100NS = int(CREATED_AT.timestamp() * 10_000_000) + _UUID_EPOCH_100NS

LogicalEdge = Tuple[str, str, str]
LogicalPath = Tuple[LogicalEdge, ...]


# ----------------------------
# Utility functions
# ----------------------------

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def percentile(sorted_vals: Sequence[float], pct: float) -> Optional[float]:
    if not sorted_vals:
        return None
    idx = int((len(sorted_vals) - 1) * pct / 100.0)
    return sorted_vals[idx]


def median(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return statistics.median(vals)


def iqr(vals: Sequence[float]) -> Optional[float]:
    if len(vals) < 4:
        return 0.0 if vals else None
    s = sorted(vals)
    # For n=5 formal repeats, this matches our prior summary convention.
    return s[3] - s[1]


def deterministic_timeuuid(graph_id: str, src: str, rel: str, dst: str, salt: str = "") -> uuid.UUID:
    """Return a deterministic UUID version 1 value acceptable for Cassandra timeuuid.

    It uses a fixed UUID timestamp and derives node/clock sequence from SHA-256.
    This is deterministic and idempotent, while still having UUID version=1.
    """
    material = f"{graph_id}|{src}|{rel}|{dst}|{salt}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    node = int.from_bytes(digest[:6], "big") & ((1 << 48) - 1)
    # Set multicast bit so this is clearly not a real MAC address.
    node |= 0x010000000000
    clock_seq = int.from_bytes(digest[6:8], "big") & 0x3FFF
    fields = (
        _FIXED_UUID_TIME_100NS & 0xFFFFFFFF,
        (_FIXED_UUID_TIME_100NS >> 32) & 0xFFFF,
        ((_FIXED_UUID_TIME_100NS >> 48) & 0x0FFF) | (1 << 12),
        (clock_seq >> 8) & 0x3F,
        clock_seq & 0xFF,
        node,
    )
    return uuid.UUID(fields=fields, version=1)


def path_hash(paths: Set[LogicalPath]) -> str:
    # Normalize as list of paths; each path is list of [src, rel, dst].
    norm = [list(map(list, p)) for p in sorted(paths)]
    return hashlib.sha256(json.dumps(norm, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def csv_logical_edge(row: Dict[str, str]) -> LogicalEdge:
    return (str(row["src_id"]), str(row["relation"]), str(row["dst_id"]))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ----------------------------
# Cassandra / Neo4j wrappers
# ----------------------------

class CassandraKG:
    def __init__(self) -> None:
        self.cluster = Cluster(CASSANDRA_HOSTS, port=CASSANDRA_PORT)
        self.session = self.cluster.connect(CASSANDRA_KEYSPACE)
        self.session.default_fetch_size = FETCH_SIZE
        self.ps: Dict[str, PreparedStatement] = {}
        self._prepare()

    def _prepare(self) -> None:
        stmts = {
            "ins_src": """
                INSERT INTO kg_edges_by_src
                (graph_id, src_id, relation, dst_id, edge_id, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            "ins_dst": """
                INSERT INTO kg_edges_by_dst
                (graph_id, dst_id, relation, src_id, edge_id, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            "ins_bucket": """
                INSERT INTO kg_edges_by_relation_bucket
                (graph_id, relation, bucket, dst_id, src_id, edge_id, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            "ins_src_rel": """
                INSERT INTO kg_edges_by_src_relation
                (graph_id, src_id, relation, dst_id, edge_id, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            "sel_src": """
                SELECT relation, dst_id
                FROM kg_edges_by_src
                WHERE graph_id=? AND src_id=?
            """,
            "sel_src_full": """
                SELECT relation, dst_id, edge_id
                FROM kg_edges_by_src
                WHERE graph_id=? AND src_id=?
            """,
        }
        for k, v in stmts.items():
            self.ps[k] = self.session.prepare(" ".join(v.split()))

    def close(self) -> None:
        try:
            self.session.shutdown()
        finally:
            self.cluster.shutdown()

    def insert_edge(self, graph_id: str, src: str, rel: str, dst: str, source: str, salt: str = "") -> None:
        eid = deterministic_timeuuid(graph_id, src, rel, dst, salt)
        # relation_bucket currently mirrors relation for this schema.
        self.session.execute(self.ps["ins_src"], (graph_id, src, rel, dst, eid, source, CREATED_AT))
        self.session.execute(self.ps["ins_dst"], (graph_id, dst, rel, src, eid, source, CREATED_AT))
        self.session.execute(self.ps["ins_bucket"], (graph_id, rel, abs(hash(rel)) % 10, dst, src, eid, source, CREATED_AT))
        self.session.execute(self.ps["ins_src_rel"], (graph_id, src, rel, dst, eid, source, CREATED_AT))

    def fetch_by_src(self, graph_id: str, src: str) -> List[LogicalEdge]:
        rows = self.session.execute(self.ps["sel_src"], (graph_id, src))
        edges = [(src, str(r.relation), str(r.dst_id)) for r in rows]
        edges.sort(key=lambda e: (e[1], e[2]))
        return edges

    def full_by_src(self, graph_id: str, src: str) -> List[LogicalEdge]:
        rows = self.session.execute(self.ps["sel_src_full"], (graph_id, src))
        edges = [(src, str(r.relation), str(r.dst_id)) for r in rows]
        edges.sort(key=lambda e: (e[1], e[2]))
        return edges

    def validate_edge_exists(self, graph_id: str, src: str, rel: str, dst: str) -> bool:
        return any(e[1] == rel and e[2] == dst for e in self.fetch_by_src(graph_id, src))


class Neo4jKG:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    def close(self) -> None:
        self.driver.close()

    def insert_edge(self, graph_id: str, src: str, rel: str, dst: str, source: str, salt: str = "") -> None:
        eid = str(deterministic_timeuuid(graph_id, src, rel, dst, salt))
        with self.driver.session() as session:
            session.run(
                """
                MERGE (s:C3KGNode {graph_id:$g, node_id:$s})
                MERGE (d:C3KGNode {graph_id:$g, node_id:$d})
                MERGE (s)-[r:C3KG_EDGE {graph_id:$g, relation:$rel, edge_id:$eid}]->(d)
                SET r.source = $source
                """,
                g=graph_id,
                s=src,
                d=dst,
                rel=rel,
                eid=eid,
                source=source,
            )

    def fetch_by_src_rel(self, graph_id: str, src: str, rel: str) -> List[LogicalEdge]:
        with self.driver.session() as session:
            rows = list(
                session.run(
                    """
                    MATCH (n:C3KGNode {graph_id:$g, node_id:$src})-[r:C3KG_EDGE {graph_id:$g, relation:$rel}]->(m:C3KGNode {graph_id:$g})
                    RETURN n.node_id AS s, r.relation AS rel, m.node_id AS d
                    ORDER BY rel, d
                    """,
                    g=graph_id,
                    src=src,
                    rel=rel,
                )
            )
        return [(str(r["s"]), str(r["rel"]), str(r["d"])) for r in rows]

    def validate_edge_exists(self, graph_id: str, src: str, rel: str, dst: str) -> bool:
        with self.driver.session() as session:
            rec = session.run(
                """
                MATCH (:C3KGNode {graph_id:$g, node_id:$s})-[r:C3KG_EDGE {graph_id:$g, relation:$rel}]->(:C3KGNode {graph_id:$g, node_id:$d})
                RETURN count(r) AS c
                """,
                g=graph_id,
                s=src,
                rel=rel,
                d=dst,
            ).single()
        return bool(rec and rec["c"] > 0)


# ----------------------------
# Data loading / import / guards
# ----------------------------

def load_csv_edges(path: Path) -> Tuple[List[Dict[str, str]], Set[LogicalEdge], Set[str]]:
    rows: List[Dict[str, str]] = []
    edge_set: Set[LogicalEdge] = set()
    src_ids: Set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"src_id", "relation", "dst_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for r in reader:
            rows.append(r)
            e = csv_logical_edge(r)
            edge_set.add(e)
            src_ids.add(e[0])
    return rows, edge_set, src_ids


def csv_stats(rows: List[Dict[str, str]], edge_set: Set[LogicalEdge], src_ids: Set[str]) -> Dict[str, Any]:
    dst_ids = {e[2] for e in edge_set}
    return {
        "csv_rows": len(rows),
        "csv_distinct_logical_edges": len(edge_set),
        "distinct_src_ids": len(src_ids),
        "distinct_dst_ids": len(dst_ids),
        "duplicate_logical_edges": len(rows) - len(edge_set),
        "nominal_scale": "100K",
        "actual_csv_distinct_edges": len(edge_set),
        "scale_label": SCALE_LABEL,
        "graph_id": READ_GRAPH_ID,
    }


def import_cassandra(rows: List[Dict[str, str]], out: Path) -> Dict[str, Any]:
    unique: Dict[LogicalEdge, Dict[str, str]] = {}
    for r in rows:
        unique[csv_logical_edge(r)] = r
    unique_rows = list(unique.values())
    batches = [unique_rows[i : i + 500] for i in range(0, len(unique_rows), 500)]

    error_path = out / "cassandra_import_errors.jsonl"
    touch(error_path)
    lock = threading.Lock()

    def worker(batch: List[Dict[str, str]]) -> Dict[str, int]:
        kg = CassandraKG()
        counts = defaultdict(int)
        errors = 0
        try:
            for r in batch:
                src, rel, dst = csv_logical_edge(r)
                source = str(r.get("source") or "c1_source_100k.csv")
                try:
                    kg.insert_edge(READ_GRAPH_ID, src, rel, dst, source)
                    counts["kg_edges_by_src"] += 1
                    counts["kg_edges_by_dst"] += 1
                    counts["kg_edges_by_relation_bucket"] += 1
                    counts["kg_edges_by_src_relation"] += 1
                except Exception as exc:
                    errors += 1
                    with lock:
                        append_jsonl(error_path, {"src": src, "relation": rel, "dst": dst, "error": repr(exc)})
        finally:
            kg.close()
        counts["errors"] = errors
        return dict(counts)

    t0 = time.time()
    totals = defaultdict(int)
    with ThreadPoolExecutor(max_workers=IMPORT_WORKERS) as executor:
        futures = [executor.submit(worker, b) for b in batches]
        for future in as_completed(futures):
            result = future.result()
            for k, v in result.items():
                totals[k] += v
    elapsed = time.time() - t0
    summary = {
        "graph_id": READ_GRAPH_ID,
        "input_csv_rows": len(rows),
        "input_unique_edges": len(unique_rows),
        "insert_success_by_table": {k: totals[k] for k in [
            "kg_edges_by_src",
            "kg_edges_by_dst",
            "kg_edges_by_relation_bucket",
            "kg_edges_by_src_relation",
        ]},
        "errors": totals["errors"],
        "deterministic_timeuuid": True,
        "created_at": CREATED_AT.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "logical_edges_per_second": round(len(unique_rows) / max(elapsed, 0.001), 3),
        "physical_rows_per_second": round((len(unique_rows) * 4) / max(elapsed, 0.001), 3),
    }
    write_json(out / "cassandra_import_summary.json", summary)
    if summary["errors"] != 0:
        raise RuntimeError(f"Cassandra import had {summary['errors']} errors. See {error_path}")
    return summary


def cassandra_guard(edge_set: Set[LogicalEdge], src_ids: Set[str], out_path: Path, graph_id: str = READ_GRAPH_ID) -> Dict[str, Any]:
    expected = edge_set
    src_list = sorted(src_ids)
    result_set: Set[LogicalEdge] = set()
    raw = 0
    empty = 0
    lock = threading.Lock()

    def worker(src_subset: List[str]) -> Tuple[int, int, Set[LogicalEdge]]:
        kg = CassandraKG()
        local_raw = 0
        local_empty = 0
        local_set: Set[LogicalEdge] = set()
        try:
            for src in src_subset:
                edges = kg.full_by_src(graph_id, src)
                if not edges:
                    local_empty += 1
                local_raw += len(edges)
                local_set.update(edges)
        finally:
            kg.close()
        return local_raw, local_empty, local_set

    chunks = [src_list[i : i + 50] for i in range(0, len(src_list), 50)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=GUARD_WORKERS) as executor:
        futures = [executor.submit(worker, c) for c in chunks]
        for future in as_completed(futures):
            local_raw, local_empty, local_set = future.result()
            raw += local_raw
            empty += local_empty
            result_set.update(local_set)
    missing = expected - result_set
    extra = result_set - expected
    guard = {
        "graph_id": graph_id,
        "actual_raw_rows": raw,
        "actual_distinct_logical_edges": len(result_set),
        "duplicates": raw - len(result_set),
        "missing_vs_csv": len(missing),
        "extra_vs_csv": len(extra),
        "partition_count_checked": len(src_list),
        "empty_partitions": empty,
        "elapsed_seconds": round(time.time() - t0, 3),
        "all_pass": raw == len(result_set) and not missing and not extra,
    }
    write_json(out_path, guard)
    return guard


def import_neo4j(rows: List[Dict[str, str]], out: Path) -> Dict[str, Any]:
    unique: Dict[LogicalEdge, Dict[str, str]] = {}
    for r in rows:
        unique[csv_logical_edge(r)] = r
    unique_rows = list(unique.values())
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    t0 = time.time()
    batch_size = 500
    try:
        with driver.session() as session:
            for i in range(0, len(unique_rows), batch_size):
                batch = []
                for r in unique_rows[i : i + batch_size]:
                    src, rel, dst = csv_logical_edge(r)
                    batch.append(
                        {
                            "g": READ_GRAPH_ID,
                            "s": src,
                            "d": dst,
                            "rel": rel,
                            "eid": str(deterministic_timeuuid(READ_GRAPH_ID, src, rel, dst)),
                            "source": str(r.get("source") or "c1_source_100k.csv"),
                        }
                    )
                session.run(
                    """
                    UNWIND $batch AS p
                    MERGE (s:C3KGNode {graph_id:p.g, node_id:p.s})
                    MERGE (d:C3KGNode {graph_id:p.g, node_id:p.d})
                    MERGE (s)-[r:C3KG_EDGE {graph_id:p.g, relation:p.rel, edge_id:p.eid}]->(d)
                    SET r.source = p.source
                    """,
                    batch=batch,
                )
    finally:
        driver.close()
    summary = {
        "graph_id": READ_GRAPH_ID,
        "input_csv_rows": len(rows),
        "input_unique_edges": len(unique_rows),
        "batch_size": batch_size,
        "deterministic_edge_id": True,
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    write_json(out / "neo4j_import_summary.json", summary)
    return summary


def neo4j_guard(edge_set: Set[LogicalEdge], out_path: Path, graph_id: str = READ_GRAPH_ID) -> Dict[str, Any]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    neo_set: Set[LogicalEdge] = set()
    try:
        with driver.session() as session:
            nc = session.run("MATCH (n:C3KGNode {graph_id:$g}) RETURN count(n) AS c", g=graph_id).single()["c"]
            ec = session.run("MATCH ()-[r:C3KG_EDGE {graph_id:$g}]->() RETURN count(r) AS c", g=graph_id).single()["c"]
            rows = session.run(
                """
                MATCH (n:C3KGNode {graph_id:$g})-[r:C3KG_EDGE {graph_id:$g}]->(m:C3KGNode {graph_id:$g})
                RETURN n.node_id AS s, r.relation AS rel, m.node_id AS d
                """,
                g=graph_id,
            )
            for rec in rows:
                neo_set.add((str(rec["s"]), str(rec["rel"]), str(rec["d"])))
    finally:
        driver.close()
    missing = edge_set - neo_set
    extra = neo_set - edge_set
    guard = {
        "graph_id": graph_id,
        "node_count": nc,
        "edge_count": ec,
        "distinct_logical_edges": len(neo_set),
        "duplicate_logical_edges": ec - len(neo_set),
        "missing_vs_csv": len(missing),
        "extra_vs_csv": len(extra),
        "all_pass": ec == len(neo_set) and not missing and not extra,
    }
    write_json(out_path, guard)
    return guard


# ----------------------------
# Traversal / semantic gate
# ----------------------------

def cassandra_traverse(q: Dict[str, Any], executor: ThreadPoolExecutor, kg: CassandraKG) -> Tuple[Set[LogicalPath], int, int]:
    relation_path = [str(x) for x in q["relation_path"]]
    frontier: Set[Tuple[str, Tuple[str, ...], LogicalPath]] = {(str(q["seed_id"]), (str(q["seed_id"]),), tuple())}
    round_trips = 0
    raw_rows = 0
    for rel in relation_path:
        sources = sorted({node for node, _, _ in frontier})
        future_to_src = {executor.submit(kg.fetch_by_src, READ_GRAPH_ID, src): src for src in sources}
        by_src: Dict[str, List[LogicalEdge]] = {}
        for future in as_completed(future_to_src):
            src = future_to_src[future]
            edges = future.result()
            round_trips += 1
            raw_rows += len(edges)
            by_src[src] = [e for e in edges if e[1] == rel]
        nxt: Set[Tuple[str, Tuple[str, ...], LogicalPath]] = set()
        for src, node_path, edge_path in frontier:
            for e in by_src.get(src, [])[:FANOUT]:
                dst = e[2]
                if dst not in node_path:
                    nxt.add((dst, node_path + (dst,), edge_path + (e,)))
        frontier = nxt
        if not frontier:
            break
    return {edge_path for _, _, edge_path in frontier}, round_trips, raw_rows


def neo4j_traverse(q: Dict[str, Any], neo: Neo4jKG) -> Tuple[Set[LogicalPath], int, int]:
    relation_path = [str(x) for x in q["relation_path"]]
    frontier: Set[Tuple[str, Tuple[str, ...], LogicalPath]] = {(str(q["seed_id"]), (str(q["seed_id"]),), tuple())}
    round_trips = 0
    raw_rows = 0
    for rel in relation_path:
        sources = sorted({node for node, _, _ in frontier})
        by_src: Dict[str, List[LogicalEdge]] = {}
        for src in sources:
            edges = neo.fetch_by_src_rel(READ_GRAPH_ID, src, rel)
            round_trips += 1
            raw_rows += len(edges)
            by_src[src] = edges
        nxt: Set[Tuple[str, Tuple[str, ...], LogicalPath]] = set()
        for src, node_path, edge_path in frontier:
            for e in by_src.get(src, [])[:FANOUT]:
                dst = e[2]
                if dst not in node_path:
                    nxt.add((dst, node_path + (dst,), edge_path + (e,)))
        frontier = nxt
        if not frontier:
            break
    return {edge_path for _, _, edge_path in frontier}, round_trips, raw_rows


def load_and_rewrite_manifest(src_path: Path, dst_path: Path) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                q["graph_id"] = READ_GRAPH_ID
                q["hop"] = HOP
                q["fanout"] = FANOUT
                q["cycle_policy"] = "path"
                queries.append(q)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return queries


def manifest_summary(queries: List[Dict[str, Any]], out: Path) -> Dict[str, Any]:
    rel_counts = defaultdict(int)
    counts: List[int] = []
    for q in queries:
        rel_counts["->".join(map(str, q.get("relation_path", [])))] += 1
        if "expected_path_count" in q:
            try:
                counts.append(int(q["expected_path_count"]))
            except Exception:
                pass
    summary = {
        "query_count": len(queries),
        "distinct_seeds": len({str(q.get("seed_id")) for q in queries}),
        "hop": HOP,
        "fanout": FANOUT,
        "cycle_policy": "path",
        "graph_id": READ_GRAPH_ID,
        "relation_path_distribution_top10": sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)[:10],
    }
    if counts:
        s = sorted(counts)
        summary.update(
            {
                "existing_expected_path_count_min": s[0],
                "existing_expected_path_count_p50": percentile(s, 50),
                "existing_expected_path_count_p95": percentile(s, 95),
                "existing_expected_path_count_max": s[-1],
            }
        )
    write_json(out / "manifest_summary.json", summary)
    return summary


def semantic_gate_and_rewrite_hashes(queries: List[Dict[str, Any]], manifest_dst: Path, out: Path) -> Dict[str, Any]:
    cass = CassandraKG()
    neo = Neo4jKG()
    executor = ThreadPoolExecutor(max_workers=FRONTIER_WORKERS)
    cass_hg = {"checked": len(queries), "empty": 0, "mismatch": 0, "all_pass": False}
    neo_hg = {"checked": len(queries), "empty": 0, "mismatch": 0, "all_pass": False}
    examples: List[Dict[str, Any]] = []
    try:
        for q in queries:
            cp, _, _ = cassandra_traverse(q, executor, cass)
            np, _, _ = neo4j_traverse(q, neo)
            ch = path_hash(cp)
            nh = path_hash(np)
            if not cp:
                cass_hg["empty"] += 1
            if not np:
                neo_hg["empty"] += 1
            if ch != nh:
                cass_hg["mismatch"] += 1
                neo_hg["mismatch"] += 1
                if len(examples) < 5:
                    examples.append({"query_id": q.get("query_id"), "cass_hash": ch, "neo_hash": nh, "cass_count": len(cp), "neo_count": len(np)})
            q["expected_path_hash"] = ch
            q["expected_path_count"] = len(cp)
        cass_hg["all_pass"] = cass_hg["empty"] == 0 and cass_hg["mismatch"] == 0
        neo_hg["all_pass"] = neo_hg["empty"] == 0 and neo_hg["mismatch"] == 0
        sg = {"cass_hg": cass_hg, "neo_hg": neo_hg, "mismatch_examples": examples, "all_pass": cass_hg["all_pass"] and neo_hg["all_pass"]}
        write_json(out / "semantic_gate_cassandra_neo4j.json", sg)
        write_json(out / "hash_gate_before_cassandra.json", cass_hg)
        write_json(out / "hash_gate_before_neo4j.json", neo_hg)
        with manifest_dst.open("w", encoding="utf-8") as f:
            for q in queries:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        return sg
    finally:
        executor.shutdown(wait=True)
        cass.close()
        neo.close()


# ----------------------------
# Mixed workload
# ----------------------------

@dataclass(frozen=True)
class WriteRecord:
    graph_id: str
    src: str
    rel: str
    dst: str
    source: str
    salt: str


class WorkloadRunner:
    def __init__(self, system: str, queries: List[Dict[str, Any]], trial_graph_id: str, seed: int) -> None:
        self.system = system
        self.queries = queries
        self.trial_graph_id = trial_graph_id
        self.seed = seed
        self.counter = 0
        self.counter_lock = threading.Lock()
        self.writes_done: List[WriteRecord] = []
        self.writes_lock = threading.Lock()
        self.cass: Optional[CassandraKG] = None
        self.neo: Optional[Neo4jKG] = None
        self.executor: Optional[ThreadPoolExecutor] = None

    def __enter__(self) -> "WorkloadRunner":
        if self.system == "cassandra_opt":
            self.cass = CassandraKG()
            self.executor = ThreadPoolExecutor(max_workers=FRONTIER_WORKERS)
        elif self.system == "neo4j":
            self.neo = Neo4jKG()
        else:
            raise ValueError(f"Unknown system: {self.system}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.cass:
            self.cass.close()
        if self.neo:
            self.neo.close()

    def _next_write(self, client_id: int) -> WriteRecord:
        with self.counter_lock:
            idx = self.counter
            self.counter += 1
        src = f"w_src_{self.system}_{client_id}_{idx}"
        dst = f"w_dst_{self.system}_{client_id}_{idx}"
        rel = "scale_write"
        source = f"scale_100k_legacy_clean_trial:{self.trial_graph_id}"
        salt = str(idx)
        return WriteRecord(self.trial_graph_id, src, rel, dst, source, salt)

    def do_read(self, q: Dict[str, Any]) -> Tuple[Set[LogicalPath], int, int]:
        if self.system == "cassandra_opt":
            assert self.cass and self.executor
            return cassandra_traverse(q, self.executor, self.cass)
        assert self.neo
        return neo4j_traverse(q, self.neo)

    def do_write(self, client_id: int) -> WriteRecord:
        wr = self._next_write(client_id)
        if self.system == "cassandra_opt":
            assert self.cass
            self.cass.insert_edge(wr.graph_id, wr.src, wr.rel, wr.dst, wr.source, wr.salt)
        else:
            assert self.neo
            self.neo.insert_edge(wr.graph_id, wr.src, wr.rel, wr.dst, wr.source, wr.salt)
        with self.writes_lock:
            self.writes_done.append(wr)
        return wr

    def validate_writes(self, k: int = 20) -> Dict[str, Any]:
        with self.writes_lock:
            sample = list(self.writes_done)
        if not sample:
            return {"checked": 0, "passed": 0, "all_pass": False}
        rng = random.Random(123)
        chosen = rng.sample(sample, min(k, len(sample)))
        passed = 0
        if self.system == "cassandra_opt":
            kg = CassandraKG()
            try:
                for wr in chosen:
                    if kg.validate_edge_exists(wr.graph_id, wr.src, wr.rel, wr.dst):
                        passed += 1
            finally:
                kg.close()
        else:
            neo = Neo4jKG()
            try:
                for wr in chosen:
                    if neo.validate_edge_exists(wr.graph_id, wr.src, wr.rel, wr.dst):
                        passed += 1
            finally:
                neo.close()
        return {"checked": len(chosen), "passed": passed, "all_pass": passed == len(chosen)}


def spotcheck(system: str, queries: List[Dict[str, Any]], k: int = 10) -> Dict[str, Any]:
    rng = random.Random(42)
    chosen = rng.sample(queries, min(k, len(queries)))
    cass = CassandraKG() if system == "cassandra_opt" else None
    neo = Neo4jKG() if system == "neo4j" else None
    executor = ThreadPoolExecutor(max_workers=FRONTIER_WORKERS) if system == "cassandra_opt" else None
    passed = 0
    failures = []
    try:
        for q in chosen:
            if system == "cassandra_opt":
                assert cass and executor
                paths, _, _ = cassandra_traverse(q, executor, cass)
            else:
                assert neo
                paths, _, _ = neo4j_traverse(q, neo)
            h = path_hash(paths)
            ok = h == q.get("expected_path_hash")
            if ok:
                passed += 1
            else:
                failures.append({"query_id": q.get("query_id"), "expected": q.get("expected_path_hash"), "actual": h, "count": len(paths)})
    finally:
        if executor:
            executor.shutdown(wait=True)
        if cass:
            cass.close()
        if neo:
            neo.close()
    return {"checked": len(chosen), "passed": passed, "all_pass": passed == len(chosen), "failures": failures[:5]}


def run_mixed_trial(system: str, mode: str, repeat: int, queries: List[Dict[str, Any]], seconds: int, warmup_seconds: int, out: Path, run_id: str, is_smoke: bool = False) -> Dict[str, Any]:
    trial_graph = f"{WRITE_GRAPH_BASE}_{system}_{mode}_r{repeat}_{'smoke' if is_smoke else 'formal'}_{run_id}"
    seed = abs(hash((system, mode, repeat, run_id))) % (2**31)

    def run_phase(duration: int, collect: bool, runner: WorkloadRunner) -> Dict[str, Any]:
        stop = threading.Event()
        read_lat: List[List[float]] = [[] for _ in range(CLIENTS)]
        write_lat: List[List[float]] = [[] for _ in range(CLIENTS)]
        read_err = [0 for _ in range(CLIENTS)]
        write_err = [0 for _ in range(CLIENTS)]
        rt_vals: List[List[int]] = [[] for _ in range(CLIENTS)]
        raw_vals: List[List[int]] = [[] for _ in range(CLIENTS)]
        read_counts = [0 for _ in range(CLIENTS)]
        write_counts = [0 for _ in range(CLIENTS)]

        def client_loop(client_id: int) -> None:
            rng = random.Random(seed + client_id * 1009 + (0 if collect else 99991))
            q_idx = client_id % len(queries)
            while not stop.is_set():
                do_write = rng.random() < WRITE_RATIO
                t0 = time.perf_counter()
                if do_write:
                    try:
                        runner.do_write(client_id)
                        if collect:
                            write_lat[client_id].append((time.perf_counter() - t0) * 1000)
                            write_counts[client_id] += 1
                    except Exception:
                        if collect:
                            write_err[client_id] += 1
                else:
                    q = queries[q_idx]
                    q_idx = (q_idx + 1) % len(queries)
                    try:
                        _, rt, raw = runner.do_read(q)
                        if collect:
                            read_lat[client_id].append((time.perf_counter() - t0) * 1000)
                            rt_vals[client_id].append(rt)
                            raw_vals[client_id].append(raw)
                            read_counts[client_id] += 1
                    except Exception:
                        if collect:
                            read_err[client_id] += 1

        threads = [threading.Thread(target=client_loop, args=(i,), daemon=True) for i in range(CLIENTS)]
        for t in threads:
            t.start()
        t_start = time.perf_counter()
        time.sleep(duration)
        t_end = time.perf_counter()
        stop.set()
        for t in threads:
            t.join(timeout=10)
        return {
            "elapsed": t_end - t_start,
            "read_lat": [x for sub in read_lat for x in sub],
            "write_lat": [x for sub in write_lat for x in sub],
            "rt_vals": [x for sub in rt_vals for x in sub],
            "raw_vals": [x for sub in raw_vals for x in sub],
            "read_errors": sum(read_err),
            "write_errors": sum(write_err),
            "reads": sum(read_counts),
            "writes": sum(write_counts),
        }

    with WorkloadRunner(system, queries, trial_graph, seed) as runner:
        if mode == "warm" and warmup_seconds > 0:
            _ = run_phase(warmup_seconds, collect=False, runner=runner)
        result = run_phase(seconds, collect=True, runner=runner)
        write_val = runner.validate_writes(20)

    read_lat = sorted(result["read_lat"])
    write_lat = sorted(result["write_lat"])
    rt_vals = sorted(result["rt_vals"])
    raw_vals = sorted(result["raw_vals"])
    elapsed = result["elapsed"]
    spot = spotcheck(system, queries, 10)

    def metric(vals: Sequence[float], func: str) -> Optional[float]:
        if not vals:
            return None
        if func == "mean":
            return round(statistics.mean(vals), 3)
        pct = float(func[1:])
        v = percentile(vals, pct)
        return round(v, 3) if v is not None else None

    row: Dict[str, Any] = {
        "run_id": run_id,
        "scale_label": SCALE_LABEL,
        "graph_id": READ_GRAPH_ID,
        "write_graph_id": trial_graph,
        "system": system,
        "mode": mode,
        "cold_mode": "process_cold" if mode == "cold" else "warm",
        "clients": CLIENTS,
        "hop": HOP,
        "fanout": FANOUT,
        "cycle_policy": "path",
        "write_ratio_target": WRITE_RATIO,
        "write_ratio_actual": round(result["writes"] / max(result["reads"] + result["writes"], 1), 4),
        "repeat": repeat,
        "warmup_seconds": warmup_seconds if mode == "warm" else 0,
        "measurement_seconds": round(elapsed, 3),
        "completed_reads": result["reads"],
        "completed_writes": result["writes"],
        "read_QPS": round(result["reads"] / max(elapsed, 0.001), 3),
        "write_QPS": round(result["writes"] / max(elapsed, 0.001), 3),
        "read_mean_ms": metric(read_lat, "mean"),
        "read_p50_ms": metric(read_lat, "p50"),
        "read_p95_ms": metric(read_lat, "p95"),
        "read_p99_ms": metric(read_lat, "p99"),
        "write_mean_ms": metric(write_lat, "mean"),
        "write_p50_ms": metric(write_lat, "p50"),
        "write_p95_ms": metric(write_lat, "p95"),
        "write_p99_ms": metric(write_lat, "p99"),
        "round_trips_mean": round(statistics.mean(rt_vals), 3) if rt_vals else None,
        "round_trips_p95": percentile(rt_vals, 95) if rt_vals else None,
        "raw_rows_mean": round(statistics.mean(raw_vals), 3) if raw_vals else None,
        "raw_rows_p95": percentile(raw_vals, 95) if raw_vals else None,
        "read_error_count": result["read_errors"],
        "write_error_count": result["write_errors"],
        "error_rate": round((result["read_errors"] + result["write_errors"]) / max(result["reads"] + result["writes"], 1), 6),
        "spotcheck_10_10": spot["all_pass"],
        "spotcheck_passed": spot["passed"],
        "spotcheck_checked": spot["checked"],
        "write_validation_20_20": write_val["all_pass"],
        "write_validation_passed": write_val["passed"],
        "write_validation_checked": write_val["checked"],
        "cache_enabled": False,
        "cache_hit_rate": 0,
        "effective_latency_ms": metric(read_lat, "mean"),
        "frontier_workers": FRONTIER_WORKERS if system == "cassandra_opt" else None,
        "relation_index_enabled": False,
        "backend_state": "process_cold" if mode == "cold" else "warm",
    }

    append_jsonl(out / ("correctness_spotcheck_smoke.jsonl" if is_smoke else f"correctness_spotcheck_{mode}.jsonl"), {"system": system, "mode": mode, "repeat": repeat, "spotcheck": spot, "write_validation": write_val})
    return row


def run_smoke(queries: List[Dict[str, Any]], out: Path, run_id: str) -> List[Dict[str, Any]]:
    rows = []
    failures = out / "failures_smoke.jsonl"
    touch(failures)
    for system in ["cassandra_opt", "neo4j"]:
        print(f"[smoke] {system}", flush=True)
        row = run_mixed_trial(system, "cold", 1, queries, SMOKE_SECONDS, 0, out, run_id, is_smoke=True)
        rows.append(row)
        print(
            f"  reads={row['completed_reads']} writes={row['completed_writes']} "
            f"rQPS={row['read_QPS']:.1f} wQPS={row['write_QPS']:.1f} "
            f"rmean={row['read_mean_ms']} rp95={row['read_p95_ms']} "
            f"wmean={row['write_mean_ms']} spot={row['spotcheck_10_10']} writeval={row['write_validation_20_20']}",
            flush=True,
        )
        if not row["spotcheck_10_10"] or not row["write_validation_20_20"] or row["read_error_count"] or row["write_error_count"]:
            append_jsonl(failures, {"system": system, "row": row, "reason": "smoke validation/error failure"})
            raise RuntimeError("Smoke failed")
    write_csv(out / "smoke_summary.csv", rows)
    with (out / "smoke_summary.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return rows


def run_formal(queries: List[Dict[str, Any]], out: Path, run_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    touch(out / "failures_cold.jsonl")
    touch(out / "failures_warm.jsonl")
    cold_rows: List[Dict[str, Any]] = []
    warm_rows: List[Dict[str, Any]] = []
    run_order: List[Dict[str, Any]] = []
    rng = random.Random(42)
    for mode in ["cold", "warm"]:
        entries = []
        for repeat in range(1, REPEATS + 1):
            for system in ["cassandra_opt", "neo4j"]:
                entries.append({"system": system, "mode": mode, "repeat": repeat})
        rng.shuffle(entries)
        for e in entries:
            run_order.append(e)
    with (out / "run_order.jsonl").open("w", encoding="utf-8") as f:
        for e in run_order:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    fieldnames: Optional[List[str]] = None
    for idx, entry in enumerate(run_order, 1):
        system, mode, repeat = entry["system"], entry["mode"], entry["repeat"]
        print(f"[formal {idx}/{len(run_order)}] {system} {mode} repeat={repeat}", flush=True)
        try:
            row = run_mixed_trial(system, mode, repeat, queries, MEASURE_SECONDS, WARMUP_SECONDS, out, run_id, is_smoke=False)
            print(
                f"  reads={row['completed_reads']} writes={row['completed_writes']} "
                f"rQPS={row['read_QPS']:.1f} wQPS={row['write_QPS']:.1f} "
                f"rmean={row['read_mean_ms']} rp95={row['read_p95_ms']} rp99={row['read_p99_ms']} "
                f"wmean={row['write_mean_ms']} spot={row['spotcheck_10_10']} writeval={row['write_validation_20_20']}",
                flush=True,
            )
            if not row["spotcheck_10_10"] or not row["write_validation_20_20"] or row["read_error_count"] or row["write_error_count"]:
                append_jsonl(out / f"failures_{mode}.jsonl", {"entry": entry, "row": row, "reason": "validation/error failure"})
                raise RuntimeError(f"Formal trial failed: {entry}")
            if fieldnames is None:
                fieldnames = list(row.keys())
            append_csv(out / f"trial_summary_{mode}.csv", row, fieldnames)
            append_jsonl(out / f"trial_summary_{mode}.jsonl", row)
            if mode == "cold":
                cold_rows.append(row)
            else:
                warm_rows.append(row)
        except Exception as exc:
            append_jsonl(out / f"failures_{mode}.jsonl", {"entry": entry, "error": repr(exc)})
            raise
    return cold_rows, warm_rows


def build_summary(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["system"]].append(r)
    final = []
    for system, grp in sorted(groups.items()):
        def med_field(k: str) -> Optional[float]:
            vals = [float(r[k]) for r in grp if r.get(k) is not None]
            m = median(vals)
            return round(m, 3) if m is not None else None
        rqps = [float(r["read_QPS"]) for r in grp]
        wqps = [float(r["write_QPS"]) for r in grp]
        final.append(
            {
                "scale_label": SCALE_LABEL,
                "graph_id": READ_GRAPH_ID,
                "system": system,
                "mode": mode,
                "n": len(grp),
                "median_read_QPS": med_field("read_QPS"),
                "median_write_QPS": med_field("write_QPS"),
                "median_read_mean_ms": med_field("read_mean_ms"),
                "median_read_p50_ms": med_field("read_p50_ms"),
                "median_read_p95_ms": med_field("read_p95_ms"),
                "median_read_p99_ms": med_field("read_p99_ms"),
                "median_write_mean_ms": med_field("write_mean_ms"),
                "median_write_p50_ms": med_field("write_p50_ms"),
                "median_write_p95_ms": med_field("write_p95_ms"),
                "median_write_p99_ms": med_field("write_p99_ms"),
                "min_read_QPS": round(min(rqps), 3) if rqps else None,
                "max_read_QPS": round(max(rqps), 3) if rqps else None,
                "IQR_read_QPS": round(iqr(rqps) or 0.0, 3),
                "min_write_QPS": round(min(wqps), 3) if wqps else None,
                "max_write_QPS": round(max(wqps), 3) if wqps else None,
                "median_cache_hit_rate": 0,
                "median_effective_latency_ms": med_field("effective_latency_ms"),
                "median_error_rate": med_field("error_rate"),
            }
        )
    return final


def write_summaries(cold_rows: List[Dict[str, Any]], warm_rows: List[Dict[str, Any]], out: Path) -> None:
    sc = build_summary(cold_rows, "cold")
    sw = build_summary(warm_rows, "warm")
    combined = sc + sw
    fields = list(combined[0].keys()) if combined else []
    write_csv(out / "final_scale_100k_legacy_clean_summary_cold.csv", sc, fields)
    write_csv(out / "final_scale_100k_legacy_clean_summary_warm.csv", sw, fields)
    write_csv(out / "final_scale_100k_legacy_clean_summary_cold_warm.csv", combined, fields)
    write_json(out / "final_scale_100k_legacy_clean_summary.json", {"cold": sc, "warm": sw, "combined": combined})


def after_hash_gate(queries: List[Dict[str, Any]], out: Path) -> Dict[str, Any]:
    cass = CassandraKG()
    neo = Neo4jKG()
    executor = ThreadPoolExecutor(max_workers=FRONTIER_WORKERS)
    hc = {"checked": len(queries), "empty": 0, "mismatch": 0, "all_pass": False}
    hn = {"checked": len(queries), "empty": 0, "mismatch": 0, "all_pass": False}
    try:
        for q in queries:
            cp, _, _ = cassandra_traverse(q, executor, cass)
            np, _, _ = neo4j_traverse(q, neo)
            ch = path_hash(cp)
            nh = path_hash(np)
            if not cp:
                hc["empty"] += 1
            if not np:
                hn["empty"] += 1
            if ch != q.get("expected_path_hash"):
                hc["mismatch"] += 1
            if nh != q.get("expected_path_hash"):
                hn["mismatch"] += 1
        hc["all_pass"] = hc["empty"] == 0 and hc["mismatch"] == 0
        hn["all_pass"] = hn["empty"] == 0 and hn["mismatch"] == 0
        write_json(out / "hash_gate_after_cassandra.json", hc)
        write_json(out / "hash_gate_after_neo4j.json", hn)
        return {"cass_hg": hc, "neo_hg": hn, "all_pass": hc["all_pass"] and hn["all_pass"]}
    finally:
        executor.shutdown(wait=True)
        cass.close()
        neo.close()


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="D:/memorytable/cassandra-kg-memory", help="Project root")
    parser.add_argument("--formal", action="store_true", help="Run formal 20 trials after smoke")
    parser.add_argument("--skip-import", action="store_true", help="Skip imports and go directly to guards/gates")
    args = parser.parse_args()

    project = Path(args.project)
    out = project / "reports/sysaxis_scale_sweep_final/scale_100k_legacy_clean"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = project / "results/c1_source_100k.csv"
    manifest_src = project / "results/c1_manifest_100k_h2.jsonl"
    manifest_dst = project / "results/sysaxis_scale_100k_legacy_clean_manifest_h2.jsonl"
    run_id = now_tag()

    ensure_file(csv_path, "100K source CSV")
    ensure_file(manifest_src, "100K h2 manifest")

    # Empty failure files should exist even if nothing fails.
    for p in ["failures_smoke.jsonl", "failures_cold.jsonl", "failures_warm.jsonl", "cassandra_import_errors.jsonl"]:
        touch(out / p)

    write_json(
        out / "environment.json",
        {
            "run_id": run_id,
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "project": str(project),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(
        out / "run_config.json",
        {
            "scale_label": SCALE_LABEL,
            "graph_id": READ_GRAPH_ID,
            "graph_type": GRAPH_TYPE,
            "not_scale_controlled": True,
            "source_csv": str(csv_path),
            "manifest_src": str(manifest_src),
            "manifest_dst": str(manifest_dst),
            "systems": ["cassandra_opt", "neo4j"],
            "clients": CLIENTS,
            "hop": HOP,
            "fanout": FANOUT,
            "write_ratio": WRITE_RATIO,
            "modes": ["cold", "warm"],
            "repeats": REPEATS,
            "cold_mode": "process_cold",
            "cache_enabled": False,
            "relation_index_enabled": False,
            "frontier_workers": FRONTIER_WORKERS,
            "write_graph_base": WRITE_GRAPH_BASE,
            "created_at": CREATED_AT.isoformat(),
            "run_formal": bool(args.formal),
            "skip_import": bool(args.skip_import),
        },
    )

    print("=== CSV stats ===", flush=True)
    rows, edge_set, src_ids = load_csv_edges(csv_path)
    stats = csv_stats(rows, edge_set, src_ids)
    write_json(out / "csv_stats.json", stats)
    print(f"CSV rows={stats['csv_rows']} distinct={stats['csv_distinct_logical_edges']} dup={stats['duplicate_logical_edges']} src={stats['distinct_src_ids']}", flush=True)

    if not args.skip_import:
        print("\n=== Cassandra idempotent import ===", flush=True)
        cass_summary = import_cassandra(rows, out)
        print(f"Cassandra import unique={cass_summary['input_unique_edges']} errors={cass_summary['errors']} elapsed={cass_summary['elapsed_seconds']}s", flush=True)

        print("\n=== Neo4j idempotent import ===", flush=True)
        neo_summary = import_neo4j(rows, out)
        print(f"Neo4j import unique={neo_summary['input_unique_edges']} elapsed={neo_summary['elapsed_seconds']}s", flush=True)

    print("\n=== Cassandra guard after import ===", flush=True)
    cass_guard = cassandra_guard(edge_set, src_ids, out / "cassandra_guard_after_import.json")
    print(f"Cass raw={cass_guard['actual_raw_rows']} distinct={cass_guard['actual_distinct_logical_edges']} dup={cass_guard['duplicates']} miss={cass_guard['missing_vs_csv']} extra={cass_guard['extra_vs_csv']} pass={cass_guard['all_pass']}", flush=True)
    if not cass_guard["all_pass"]:
        raise RuntimeError("Cassandra import guard failed")

    print("\n=== Neo4j guard after import ===", flush=True)
    neo_guard = neo4j_guard(edge_set, out / "neo4j_guard_after_import.json")
    print(f"Neo edges={neo_guard['edge_count']} distinct={neo_guard['distinct_logical_edges']} dup={neo_guard['duplicate_logical_edges']} miss={neo_guard['missing_vs_csv']} extra={neo_guard['extra_vs_csv']} pass={neo_guard['all_pass']}", flush=True)
    if not neo_guard["all_pass"]:
        raise RuntimeError("Neo4j import guard failed")

    print("\n=== Manifest rewrite ===", flush=True)
    queries = load_and_rewrite_manifest(manifest_src, manifest_dst)
    manifest_summary(queries, out)
    if len(queries) != 256:
        raise RuntimeError(f"Manifest query_count is {len(queries)}, expected 256")
    print(f"Manifest rewritten: {manifest_dst} queries={len(queries)}", flush=True)

    print("\n=== Semantic gate + expected hash rewrite ===", flush=True)
    sg = semantic_gate_and_rewrite_hashes(queries, manifest_dst, out)
    print(f"Semantic gate pass={sg['all_pass']} cass={sg['cass_hg']} neo={sg['neo_hg']}", flush=True)
    if not sg["all_pass"]:
        raise RuntimeError("Semantic gate failed")

    # Reload queries with expected hash from rewritten manifest.
    queries = [json.loads(line) for line in manifest_dst.open("r", encoding="utf-8") if line.strip()]

    print("\n=== Smoke: real 10% mixed read/write ===", flush=True)
    smoke_rows = run_smoke(queries, out, run_id)
    print("Smoke PASS", flush=True)

    if not args.formal:
        print("\nSTOP AFTER SMOKE: rerun with --formal to execute 20 formal trials.", flush=True)
        return 0

    print("\n=== Formal 20 trials ===", flush=True)
    cold_rows, warm_rows = run_formal(queries, out, run_id)

    print("\n=== After read-graph guards ===", flush=True)
    cass_after = cassandra_guard(edge_set, src_ids, out / "cassandra_guard_after_trials.json")
    neo_after = neo4j_guard(edge_set, out / "neo4j_guard_after_trials.json")
    print(f"Cass after pass={cass_after['all_pass']} Neo after pass={neo_after['all_pass']}", flush=True)
    if not cass_after["all_pass"] or not neo_after["all_pass"]:
        raise RuntimeError("After read-graph guard failed")

    print("\n=== After hash gates ===", flush=True)
    ah = after_hash_gate(queries, out)
    print(f"After hash gate pass={ah['all_pass']} cass={ah['cass_hg']} neo={ah['neo_hg']}", flush=True)
    if not ah["all_pass"]:
        raise RuntimeError("After hash gate failed")

    print("\n=== Summary ===", flush=True)
    write_summaries(cold_rows, warm_rows, out)
    print(f"Completed. Outputs: {out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
