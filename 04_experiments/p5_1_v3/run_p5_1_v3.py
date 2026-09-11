"""P5-1 v3: logical-memory update to final retrieval visibility.

Unlike legacy P5-1, success requires all logical event components to be visible
and the newly inserted memory to occur in the shared RawERK BM25 Top-K computed
from a fresh backend read.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from p5_1_protocol import (
    PROTOCOL_VERSION,
    RETRIEVER_ID,
    LogicalEvent,
    bm25_rank,
    file_sha256,
    load_source_rows,
    logical_event_digest,
    materialize_event,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "02_artifacts" / "frozen_update_events.jsonl"
DEFAULT_OUTPUT = ROOT / "05_reports" / "p5_1_retrieval_visibility_v3"


def load_local_env(path: Path) -> None:
    """Load missing variables from a git-ignored local .env without logging values."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class BackendAdapter(Protocol):
    name: str

    def write_event(self, event: LogicalEvent) -> None: ...
    def fetch_logical_view(self, event: LogicalEvent) -> dict | None: ...
    def fetch_candidates(self, run_scope: str, memory_ids: list[str]) -> list[tuple[str, str]]: ...
    def close(self) -> None: ...


class CassandraAdapter:
    name = "cassandra"

    def __init__(self, host: str, keyspace: str = "p5_1_v3") -> None:
        from cassandra.cluster import Cluster

        self.cluster = Cluster([host], protocol_version=4)
        self.session = self.cluster.connect()
        self.session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
            "WITH replication={'class':'SimpleStrategy','replication_factor':1}"
        )
        self.session.set_keyspace(keyspace)
        self.session.execute(
            "CREATE TABLE IF NOT EXISTS memory_by_scope ("
            "run_scope text, memory_id text, update_id text, version int, raw_text text, "
            "raw_erk_text text, src_id text, dst_id text, relation text, edge_id text, "
            "probe_token text, PRIMARY KEY ((run_scope), memory_id))"
        )
        self.session.execute(
            "CREATE TABLE IF NOT EXISTS entity_by_scope ("
            "run_scope text, entity_id text, PRIMARY KEY ((run_scope), entity_id))"
        )
        self.session.execute(
            "CREATE TABLE IF NOT EXISTS edge_by_scope_src ("
            "run_scope text, src_id text, edge_id text, dst_id text, relation text, memory_id text, version int, "
            "PRIMARY KEY ((run_scope, src_id), edge_id))"
        )
        self.p_memory = self.session.prepare(
            "INSERT INTO memory_by_scope (run_scope,memory_id,update_id,version,raw_text,raw_erk_text,src_id,dst_id,relation,edge_id,probe_token) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)"
        )
        self.p_entity = self.session.prepare(
            "INSERT INTO entity_by_scope (run_scope,entity_id) VALUES (?,?)"
        )
        self.p_edge = self.session.prepare(
            "INSERT INTO edge_by_scope_src (run_scope,src_id,edge_id,dst_id,relation,memory_id,version) VALUES (?,?,?,?,?,?,?)"
        )
        self.p_get_memory = self.session.prepare(
            "SELECT * FROM memory_by_scope WHERE run_scope=? AND memory_id=?"
        )
        self.p_get_entity = self.session.prepare(
            "SELECT entity_id FROM entity_by_scope WHERE run_scope=? AND entity_id=?"
        )
        self.p_get_edge = self.session.prepare(
            "SELECT edge_id,dst_id,relation,memory_id,version FROM edge_by_scope_src WHERE run_scope=? AND src_id=? AND edge_id=?"
        )
        self.p_candidates = self.session.prepare(
            "SELECT memory_id,raw_erk_text FROM memory_by_scope WHERE run_scope=? AND memory_id IN ?"
        )

    def write_event(self, event: LogicalEvent) -> None:
        from cassandra.query import BatchStatement, BatchType

        batch = BatchStatement(batch_type=BatchType.LOGGED)
        batch.add(self.p_memory, (
            event.run_scope, event.memory_id, event.update_id, event.version,
            event.raw_text, event.raw_erk_text, event.src_id, event.dst_id,
            event.relation, event.edge_id, event.probe_token,
        ))
        batch.add(self.p_entity, (event.run_scope, event.src_id))
        batch.add(self.p_entity, (event.run_scope, event.dst_id))
        batch.add(self.p_edge, (
            event.run_scope, event.src_id, event.edge_id, event.dst_id,
            event.relation, event.memory_id, event.version,
        ))
        self.session.execute(batch)

    def fetch_logical_view(self, event: LogicalEvent) -> dict | None:
        memory = self.session.execute(self.p_get_memory, (event.run_scope, event.memory_id)).one()
        src = self.session.execute(self.p_get_entity, (event.run_scope, event.src_id)).one()
        dst = self.session.execute(self.p_get_entity, (event.run_scope, event.dst_id)).one()
        edge = self.session.execute(self.p_get_edge, (event.run_scope, event.src_id, event.edge_id)).one()
        if not (memory and src and dst and edge):
            return None
        return {
            "update_id": memory.update_id,
            "run_scope": memory.run_scope,
            "memory_id": memory.memory_id,
            "version": int(memory.version),
            "raw_text": memory.raw_text,
            "raw_erk_text": memory.raw_erk_text,
            "src_id": memory.src_id,
            "dst_id": edge.dst_id,
            "relation": edge.relation,
            "edge_id": edge.edge_id,
            "probe_token": memory.probe_token,
        }

    def fetch_candidates(self, run_scope: str, memory_ids: list[str]) -> list[tuple[str, str]]:
        rows = self.session.execute(self.p_candidates, (run_scope, memory_ids))
        return sorted((row.memory_id, row.raw_erk_text) for row in rows)

    def close(self) -> None:
        self.session.shutdown()
        self.cluster.shutdown()


class Neo4jAdapter:
    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        from neo4j import GraphDatabase

        if not password:
            raise RuntimeError("NEO4J_PASSWORD is required for the Neo4j backend")
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_pool_size=128)
        self.driver.verify_connectivity()
        self.database = database
        constraints = [
            "CREATE CONSTRAINT p5v3_memory_key IF NOT EXISTS FOR (m:P5V3Memory) REQUIRE (m.run_scope,m.memory_id) IS UNIQUE",
            "CREATE CONSTRAINT p5v3_entity_key IF NOT EXISTS FOR (e:P5V3Entity) REQUIRE (e.run_scope,e.entity_id) IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for query in constraints:
                session.run(query).consume()

    def write_event(self, event: LogicalEvent) -> None:
        payload = event.canonical_view()
        query = """
        MERGE (m:P5V3Memory {run_scope:$run_scope, memory_id:$memory_id})
        SET m.update_id=$update_id, m.version=$version, m.raw_text=$raw_text,
            m.raw_erk_text=$raw_erk_text, m.src_id=$src_id, m.dst_id=$dst_id,
            m.relation=$relation, m.edge_id=$edge_id, m.probe_token=$probe_token
        MERGE (src:P5V3Entity {run_scope:$run_scope, entity_id:$src_id})
        MERGE (dst:P5V3Entity {run_scope:$run_scope, entity_id:$dst_id})
        MERGE (src)-[r:P5V3_KG_EDGE {run_scope:$run_scope, edge_id:$edge_id}]->(dst)
        SET r.relation=$relation, r.memory_id=$memory_id, r.version=$version
        RETURN m.memory_id AS memory_id
        """
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query, **payload).single(strict=True))

    def fetch_logical_view(self, event: LogicalEvent) -> dict | None:
        query = """
        MATCH (m:P5V3Memory {run_scope:$run_scope, memory_id:$memory_id})
        MATCH (src:P5V3Entity {run_scope:$run_scope, entity_id:$src_id})
              -[r:P5V3_KG_EDGE {run_scope:$run_scope, edge_id:$edge_id}]->
              (dst:P5V3Entity {run_scope:$run_scope, entity_id:$dst_id})
        RETURN m.update_id AS update_id, m.run_scope AS run_scope, m.memory_id AS memory_id,
               m.version AS version, m.raw_text AS raw_text, m.raw_erk_text AS raw_erk_text,
               src.entity_id AS src_id, dst.entity_id AS dst_id, r.relation AS relation,
               r.edge_id AS edge_id, m.probe_token AS probe_token
        """
        with self.driver.session(database=self.database) as session:
            row = session.run(query, **event.canonical_view()).single()
            return dict(row) if row else None

    def fetch_candidates(self, run_scope: str, memory_ids: list[str]) -> list[tuple[str, str]]:
        query = "MATCH (m:P5V3Memory {run_scope:$run_scope}) WHERE m.memory_id IN $memory_ids RETURN m.memory_id AS memory_id, m.raw_erk_text AS raw_erk_text ORDER BY memory_id"
        with self.driver.session(database=self.database) as session:
            return [(row["memory_id"], row["raw_erk_text"]) for row in session.run(query, run_scope=run_scope, memory_ids=memory_ids)]

    def close(self) -> None:
        self.driver.close()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return ordered[index]


def make_adapter(name: str, args: argparse.Namespace) -> BackendAdapter:
    if name == "cassandra":
        return CassandraAdapter(args.cassandra_host, args.cassandra_keyspace)
    if name == "neo4j":
        password = os.environ.get("NEO4J_PASSWORD", "")
        return Neo4jAdapter(args.neo4j_uri, args.neo4j_user, password, args.neo4j_database)
    raise ValueError(name)


def check_event_visible(adapter: BackendAdapter, event: LogicalEvent, baseline_ids: list[str], top_k: int, timeout_s: float, poll_s: float) -> dict:
    start = time.perf_counter()
    adapter.write_event(event)
    commit = time.perf_counter()
    deadline = start + timeout_s
    attempts = 0
    logical_visible_at = None
    topk_ids: list[str] = []
    candidate_count = 0
    target_rank = None
    mismatch = ""
    while time.perf_counter() <= deadline:
        attempts += 1
        view = adapter.fetch_logical_view(event)
        now = time.perf_counter()
        if view == event.canonical_view():
            if logical_visible_at is None:
                logical_visible_at = now
            candidate_ids = sorted(baseline_ids + [event.memory_id])
            candidates = adapter.fetch_candidates(event.run_scope, candidate_ids)
            candidate_count = len(candidates)
            if [memory_id for memory_id, _ in candidates] != candidate_ids:
                mismatch = "fixed_candidate_cohort_incomplete_or_extra"
                time.sleep(poll_s)
                continue
            ranked = bm25_rank(event.probe_query, candidates, top_k=top_k)
            topk_ids = [memory_id for memory_id, _ in ranked]
            if event.memory_id in topk_ids:
                visible = time.perf_counter()
                target_rank = topk_ids.index(event.memory_id) + 1
                return {
                    "status": "ok",
                    "update_to_commit_ms": (commit - start) * 1000,
                    "commit_to_logical_visible_ms": (logical_visible_at - commit) * 1000,
                    "update_to_topk_ms": (visible - start) * 1000,
                    "retrieval_after_commit_ms": (visible - commit) * 1000,
                    "probe_attempts": attempts,
                    "candidate_count": candidate_count,
                    "target_rank": target_rank,
                    "topk_ids": ";".join(topk_ids),
                    "mismatch": "",
                }
            mismatch = "logical_view_complete_but_target_not_in_topk"
        elif view is not None:
            mismatch = "logical_view_mismatch"
        else:
            mismatch = "logical_view_incomplete"
        time.sleep(poll_s)
    end = time.perf_counter()
    return {
        "status": "timeout",
        "update_to_commit_ms": (commit - start) * 1000,
        "commit_to_logical_visible_ms": (logical_visible_at - commit) * 1000 if logical_visible_at else -1.0,
        "update_to_topk_ms": (end - start) * 1000,
        "retrieval_after_commit_ms": (end - commit) * 1000,
        "probe_attempts": attempts,
        "candidate_count": candidate_count,
        "target_rank": target_rank or -1,
        "topk_ids": ";".join(topk_ids),
        "mismatch": mismatch,
    }


def event_sets(source_rows: list[dict], run_base: str, shards: int, baseline_per_shard: int, warmup_events: int, formal_events: int):
    needed = shards * baseline_per_shard + warmup_events + formal_events
    if len(source_rows) < needed:
        raise ValueError(f"Need {needed} source rows, found {len(source_rows)}")
    cursor = 0
    baseline: list[LogicalEvent] = []
    for shard in range(shards):
        scope = f"{run_base}_s{shard:03d}"
        for local in range(baseline_per_shard):
            baseline.append(materialize_event(source_rows[cursor], scope, "baseline", shard * baseline_per_shard + local))
            cursor += 1
    warmup: list[LogicalEvent] = []
    for index in range(warmup_events):
        scope = f"{run_base}_s{index % shards:03d}"
        warmup.append(materialize_event(source_rows[cursor], scope, "warmup", index))
        cursor += 1
    formal: list[LogicalEvent] = []
    for index in range(formal_events):
        scope = f"{run_base}_s{index % shards:03d}"
        formal.append(materialize_event(source_rows[cursor], scope, "formal", index))
        cursor += 1
    return baseline, warmup, formal


def seed_events(adapter: BackendAdapter, events: list[LogicalEvent], workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(adapter.write_event, event) for event in events]
        for future in as_completed(futures):
            future.result()


def run_backend_configuration(adapter: BackendAdapter, backend: str, concurrency: int, run_id: int, args: argparse.Namespace, source_rows: list[dict]) -> tuple[list[dict], dict]:
    run_base = f"{args.run_tag}_c{concurrency}_r{run_id}"
    baseline, warmup, formal = event_sets(
        source_rows, run_base, args.shards, args.baseline_per_shard,
        args.warmup_events, args.formal_events,
    )
    baseline_ids_by_scope: dict[str, list[str]] = {}
    for event in baseline:
        baseline_ids_by_scope.setdefault(event.run_scope, []).append(event.memory_id)
    for ids in baseline_ids_by_scope.values():
        ids.sort()
    seed_events(adapter, baseline, min(concurrency, 32))
    for event in warmup:
        result = check_event_visible(
            adapter, event, baseline_ids_by_scope[event.run_scope],
            args.top_k, args.timeout_s, args.poll_s,
        )
        if result["status"] != "ok":
            raise RuntimeError(f"Warmup failed for {backend}: {event.update_id} {result['mismatch']}")
    started = time.perf_counter()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_map = {
            pool.submit(
                check_event_visible, adapter, event,
                baseline_ids_by_scope[event.run_scope], args.top_k,
                args.timeout_s, args.poll_s,
            ): event
            for event in formal
        }
        for future in as_completed(future_map):
            event = future_map[future]
            result = future.result()
            rows.append({
                "protocol_version": PROTOCOL_VERSION,
                "retriever_id": RETRIEVER_ID,
                "backend": backend,
                "concurrency": concurrency,
                "run_id": run_id,
                "update_id": event.update_id,
                "run_scope": event.run_scope,
                "memory_id": event.memory_id,
                "expected_candidate_count": args.baseline_per_shard + 1,
                **result,
            })
    elapsed = time.perf_counter() - started
    rows.sort(key=lambda row: row["update_id"])
    ok = [row for row in rows if row["status"] == "ok"]
    summary = {
        "backend": backend,
        "concurrency": concurrency,
        "run_id": run_id,
        "formal_events": len(rows),
        "successful_events": len(ok),
        "timeouts": len(rows) - len(ok),
        "elapsed_s": elapsed,
        "throughput_events_s": len(rows) / elapsed if elapsed else 0.0,
        "p50_update_to_topk_ms": percentile([row["update_to_topk_ms"] for row in ok], 0.50),
        "p95_update_to_topk_ms": percentile([row["update_to_topk_ms"] for row in ok], 0.95),
        "p99_update_to_topk_ms": percentile([row["update_to_topk_ms"] for row in ok], 0.99),
        "mean_candidate_count": statistics.mean(row["candidate_count"] for row in ok) if ok else 0.0,
        "logical_event_sha256": logical_event_digest(formal),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_configurations(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["backend"]), int(row["concurrency"])), []).append(row)
    summaries: list[dict] = []
    for (backend, concurrency), group in sorted(grouped.items()):
        ok = [row for row in group if row["status"] == "ok"]
        latencies = [float(row["update_to_topk_ms"]) for row in ok]
        commit_latencies = [float(row["update_to_commit_ms"]) for row in ok]
        summaries.append({
            "protocol_version": PROTOCOL_VERSION,
            "retriever_id": RETRIEVER_ID,
            "backend": backend,
            "concurrency": concurrency,
            "formal_events": len(group),
            "successful_events": len(ok),
            "timeouts": len(group) - len(ok),
            "p50_update_to_topk_ms": percentile(latencies, 0.50),
            "p95_update_to_topk_ms": percentile(latencies, 0.95),
            "p99_update_to_topk_ms": percentile(latencies, 0.99),
            "p50_update_to_commit_ms": percentile(commit_latencies, 0.50),
            "p95_update_to_commit_ms": percentile(commit_latencies, 0.95),
            "p99_update_to_commit_ms": percentile(commit_latencies, 0.99),
            "mean_candidate_count": statistics.mean(float(row["candidate_count"]) for row in ok) if ok else 0.0,
            "mean_target_rank": statistics.mean(float(row["target_rank"]) for row in ok) if ok else 0.0,
        })
    return summaries


def run_cross_backend_parity(args: argparse.Namespace, source_rows: list[dict]) -> dict:
    """Sequential state and Top-K parity gate, independent of concurrency order."""

    if set(args.backends) != {"cassandra", "neo4j"}:
        return {"status": "NOT_RUN", "reason": "requires Cassandra and Neo4j"}
    run_base = f"{args.run_tag}_parity"
    baseline, _, formal = event_sets(source_rows, run_base, 2, 8, 0, 4)
    events = baseline + formal
    baseline_ids_by_scope: dict[str, list[str]] = {}
    for event in baseline:
        baseline_ids_by_scope.setdefault(event.run_scope, []).append(event.memory_id)
    for ids in baseline_ids_by_scope.values():
        ids.sort()
    adapters = {backend: make_adapter(backend, args) for backend in ("cassandra", "neo4j")}
    try:
        for adapter in adapters.values():
            for event in events:
                adapter.write_event(event)
        logical_mismatches: list[str] = []
        for event in events:
            expected = event.canonical_view()
            views = {name: adapter.fetch_logical_view(event) for name, adapter in adapters.items()}
            if any(view != expected for view in views.values()) or views["cassandra"] != views["neo4j"]:
                logical_mismatches.append(event.update_id)
        candidate_mismatches: list[str] = []
        topk_mismatches: list[str] = []
        for event in formal:
            scope = event.run_scope
            candidate_ids = sorted(baseline_ids_by_scope[scope] + [event.memory_id])
            candidates = {
                name: adapter.fetch_candidates(scope, candidate_ids)
                for name, adapter in adapters.items()
            }
            if candidates["cassandra"] != candidates["neo4j"] or [item[0] for item in candidates["cassandra"]] != candidate_ids:
                candidate_mismatches.append(event.update_id)
                continue
            cass_topk = bm25_rank(event.probe_query, candidates["cassandra"], top_k=args.top_k)
            neo_topk = bm25_rank(event.probe_query, candidates["neo4j"], top_k=args.top_k)
            if cass_topk != neo_topk or event.memory_id not in [item[0] for item in cass_topk]:
                topk_mismatches.append(event.update_id)
        passed = not logical_mismatches and not candidate_mismatches and not topk_mismatches
        return {
            "status": "PASS" if passed else "FAIL",
            "events_checked": len(events),
            "formal_queries_checked": len(formal),
            "logical_mismatches": logical_mismatches,
            "candidate_mismatches": candidate_mismatches,
            "topk_mismatches": topk_mismatches,
        }
    finally:
        for adapter in adapters.values():
            adapter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "benchmark"), default="preflight")
    parser.add_argument("--backends", nargs="+", choices=("cassandra", "neo4j"), default=["cassandra", "neo4j"])
    parser.add_argument("--source-events", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-tag", default=datetime.now(timezone.utc).strftime("p5v3_%Y%m%dT%H%M%SZ"))
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--formal-events", type=int, default=2000)
    parser.add_argument("--warmup-events", type=int, default=100)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--baseline-per-shard", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--poll-s", type=float, default=0.01)
    parser.add_argument("--cassandra-host", default="127.0.0.1")
    parser.add_argument("--cassandra-keyspace", default="p5_1_v3")
    parser.add_argument("--neo4j-uri", default="bolt://127.0.0.1:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-database", default="neo4j")
    args = parser.parse_args()
    if args.mode == "preflight":
        args.concurrencies = [2]
        args.runs = 1
        args.formal_events = min(args.formal_events, 8)
        args.warmup_events = min(args.warmup_events, 2)
        args.shards = min(args.shards, 2)
        args.baseline_per_shard = min(args.baseline_per_shard, 8)
    if any(value <= 0 for value in args.concurrencies):
        parser.error("concurrencies must be positive")
    for field in ("runs", "formal_events", "shards", "baseline_per_shard", "top_k"):
        if getattr(args, field) <= 0:
            parser.error(f"{field} must be positive")
    if args.warmup_events < 0:
        parser.error("warmup_events must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    load_local_env(ROOT / ".env")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = load_source_rows(args.source_events)
    expected_rows = len(args.backends) * len(args.concurrencies) * args.runs * args.formal_events
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "retriever_id": RETRIEVER_ID,
        "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "mode": args.mode,
        "source_events": str(args.source_events.resolve()),
        "source_events_sha256": file_sha256(args.source_events),
        "source_event_rows": len(source_rows),
        "logical_event_contract": {
            "memory_documents": 1,
            "entity_records": 2,
            "directed_kg_edges": 1,
            "terminal_condition": "complete logical view and memory_id present in shared RawERK BM25 Top-K from fresh backend candidate read",
            "candidate_cohort": "exact frozen shard baseline memory IDs plus the current target memory only",
        },
        "config": {
            "backends": args.backends,
            "concurrencies": args.concurrencies,
            "runs_per_backend_concurrency": args.runs,
            "formal_events_per_run": args.formal_events,
            "warmup_events_per_run": args.warmup_events,
            "shards": args.shards,
            "baseline_per_shard": args.baseline_per_shard,
            "top_k": args.top_k,
            "timeout_s": args.timeout_s,
            "poll_s": args.poll_s,
        },
        "expected_formal_rows": expected_rows,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    manifest_path = args.output_dir / f"{args.run_tag}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_rows: list[dict] = []
    run_summaries: list[dict] = []
    try:
        for concurrency in args.concurrencies:
            for run_id in range(1, args.runs + 1):
                backend_order = args.backends if run_id % 2 else list(reversed(args.backends))
                for backend in backend_order:
                    print(f"[{backend}] c={concurrency} run={run_id}", flush=True)
                    adapter = make_adapter(backend, args)
                    try:
                        rows, summary = run_backend_configuration(
                            adapter, backend, concurrency, run_id, args, source_rows
                        )
                    finally:
                        adapter.close()
                    all_rows.extend(rows)
                    run_summaries.append(summary)
                    print(json.dumps(summary, ensure_ascii=False), flush=True)
        if len(all_rows) != expected_rows:
            raise RuntimeError(f"Manifest count gate failed: expected {expected_rows}, got {len(all_rows)}")
        timeout_count = sum(row["status"] != "ok" for row in all_rows)
        if timeout_count:
            raise RuntimeError(f"Visibility gate failed: {timeout_count} events did not reach Top-K")
        bad_candidate_counts = [
            row["update_id"] for row in all_rows
            if int(row["candidate_count"]) != int(row["expected_candidate_count"])
        ]
        if bad_candidate_counts:
            raise RuntimeError(f"Fixed candidate cohort gate failed for {len(bad_candidate_counts)} events")
        digests: dict[tuple[int, int], set[str]] = {}
        for row in run_summaries:
            key = (int(row["concurrency"]), int(row["run_id"]))
            digests.setdefault(key, set()).add(str(row["logical_event_sha256"]))
        bad_digest_keys = [key for key, values in digests.items() if len(values) != 1]
        if bad_digest_keys:
            raise RuntimeError(f"Cross-backend logical event digest mismatch: {bad_digest_keys}")
        parity = run_cross_backend_parity(args, source_rows)
        if parity["status"] == "FAIL":
            raise RuntimeError(f"Cross-backend state/Top-K parity failed: {parity}")
        per_event_path = args.output_dir / f"{args.run_tag}_per_event.csv"
        per_run_path = args.output_dir / f"{args.run_tag}_per_run.csv"
        summary_path = args.output_dir / f"{args.run_tag}_summary.csv"
        write_csv(per_event_path, all_rows)
        write_csv(per_run_path, run_summaries)
        write_csv(summary_path, aggregate_configurations(all_rows))
        full_cross_backend = len(args.backends) == 2 and parity["status"] == "PASS"
        citation_ready = args.mode == "benchmark" and full_cross_backend
        manifest.update({
            "status": "PASS" if full_cross_backend else "PASS_PARTIAL",
            "citation_ready": citation_ready,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "actual_formal_rows": len(all_rows),
            "timeouts": timeout_count,
            "per_event_csv": per_event_path.name,
            "per_event_sha256": file_sha256(per_event_path),
            "per_run_csv": per_run_path.name,
            "per_run_sha256": file_sha256(per_run_path),
            "summary_csv": summary_path.name,
            "summary_sha256": file_sha256(summary_path),
            "gates": {
                "manifest_count_exact": True,
                "all_events_entered_topk": True,
                "fixed_candidate_count_exact": True,
                "cross_backend_logical_event_digest_equal": True if len(args.backends) == 2 else None,
                "cross_backend_state_and_topk_parity": parity,
            },
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{manifest['status']}: {len(all_rows)} rows -> {args.output_dir}", flush=True)
        return 0
    except Exception as exc:
        manifest.update({
            "status": "FAIL",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "actual_formal_rows": len(all_rows),
            "error": f"{type(exc).__name__}: {exc}",
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
