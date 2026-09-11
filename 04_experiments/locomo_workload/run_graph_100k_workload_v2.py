"""Run one graph-aware v2 100K 95:5 mixed-workload point."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from graph_event_v2 import digest
from live_cells_graph_v2 import CassandraGraphCells, Neo4jGraphCells
from online_retrieval import ScopedOnlineRetrievalIndex
from run_canonical_live_gate import ROOT, env
from run_graph_100k_load_gate_v2 import expand
from run_graph_canonical_gate_v2 import build_graph_records

OUT = ROOT / "05_reports" / "locomo_workload_graph_v2_100k" / "main_95_5_v2" / "runs"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=["cassandra-base", "cassandra-materialized", "neo4j-native", "neo4j-materialized"])
    parser.add_argument("--concurrency", required=True, type=int, choices=[1, 8, 16, 32, 64])
    parser.add_argument("--rep", required=True, type=int, choices=[0, 1, 2])
    parser.add_argument("--warmup-limit", type=int, default=500)
    parser.add_argument("--measured-limit", type=int, default=5000)
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args(); env()

    records = expand(build_graph_records()); by_id = {record.memory_id: record for record in records}; by_scope = {}
    for record in records:
        by_scope.setdefault(record.scope_id, []).append(record)
    memory_ids = (ROOT / "01_data" / "locomo_memory_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    memory_array = np.load(ROOT / "01_data" / "locomo_memory_bge_large.npy", mmap_mode="r")
    memory_vector = {memory_id: memory_array[index] for index, memory_id in enumerate(memory_ids)}
    qa_ids = (ROOT / "01_data" / "locomo_qa_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    qa_array = np.load(ROOT / "01_data" / "locomo_qa_bge_large.npy", mmap_mode="r")
    qa_vector = {qa_id: qa_array[index] for index, qa_id in enumerate(qa_ids)}
    trace_dir = ROOT / "05_reports" / "locomo_workload_100k" / "traces"
    warmup = load_jsonl(trace_dir / f"rep{args.rep}_warmup.jsonl")[:args.warmup_limit]
    measured = load_jsonl(trace_dir / f"rep{args.rep}_measured.jsonl")[:args.measured_limit]
    targets = {op["target_memory_id"] for op in warmup + measured if op["op_type"] == "update"}
    backend = CassandraGraphCells(os.getenv("CASSANDRA_HOST", "127.0.0.1")) if args.cell.startswith("cassandra") else Neo4jGraphCells(os.getenv("NEO4J_URI", "bolt://localhost:7687"), os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"), os.getenv("NEO4J_DATABASE", "neo4j"))
    index = ScopedOnlineRetrievalIndex(); print(f"building_index cell={args.cell}", flush=True)
    for scope, scope_records in by_scope.items():
        base = [record for record in scope_records if record.memory_id not in targets]
        index.load_scope(scope, {record.memory_id: record.rawerk for record in base}, {record.memory_id: memory_vector[record.memory_id.split("::", 1)[1]] for record in base})
    for target in targets:
        backend.delete(args.cell, by_id[target])

    def execute(operation, phase):
        started = time.perf_counter_ns(); scope = operation["scope_id"]
        if operation["op_type"] == "read":
            backend_start = time.perf_counter_ns(); candidate_ids = backend.ids(args.cell, scope); backend_end = time.perf_counter_ns()
            result, timings = index.search_with_timings(scope, operation["question"], qa_vector[operation["source_qa_id"]], candidate_ids)
            ended = time.perf_counter_ns()
            return {"op_id": operation["op_id"], "phase": phase, "op_type": "read", "status": "ok", "latency_ms": (ended-started)/1e6, "backend_ms": (backend_end-backend_start)/1e6, **timings, "fresh_hit_at_10": "", "graph_projection_ok": "", "relation_candidate_ok": "", "candidate_count": len(candidate_ids)}
        record = by_id[operation["target_memory_id"]]
        backend.insert(args.cell, record); committed = time.perf_counter_ns()
        projection = backend.fetch_graph(args.cell, scope, record.memory_id); graph_visible = time.perf_counter_ns()
        relation_ok = all(record.memory_id in backend.ids_by_relation(args.cell, scope, relation) for relation in sorted({edge["relation"] for edge in record.edges}))
        index.upsert_sparse(scope, record.memory_id, record.rawerk, 2)
        index.upsert_dense(scope, record.memory_id, memory_vector[operation["source_memory_id"]], 2)
        candidate_ids = backend.ids(args.cell, scope); backend_done = time.perf_counter_ns()
        result, timings = index.search_with_timings(scope, operation["question"], qa_vector[operation["source_qa_id"]], candidate_ids)
        ended = time.perf_counter_ns(); top10 = {memory_id for memory_id, _ in result.top10}
        return {"op_id": operation["op_id"], "phase": phase, "op_type": "update", "status": "ok", "latency_ms": (ended-started)/1e6, "backend_ms": (backend_done-started)/1e6, **timings, "fresh_hit_at_10": int(record.memory_id in top10), "graph_projection_ok": int(bool(projection) and digest(projection)==digest(record.graph_projection())), "relation_candidate_ok": int(relation_ok), "candidate_count": len(candidate_ids), "commit_ms": (committed-started)/1e6, "graph_visible_ms": (graph_visible-started)/1e6}

    def run_phase(operations, phase):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            rows = list(executor.map(lambda operation: execute(operation, phase), operations))
        return rows, time.perf_counter()-started

    try:
        _, warmup_seconds = run_phase(warmup, "warmup")
        rows, measured_seconds = run_phase(measured, "measured")
        output = __import__("pathlib").Path(args.output); output.mkdir(parents=True, exist_ok=True)
        suffix = "" if args.measured_limit == 5000 and args.warmup_limit == 500 else f"_smoke{args.measured_limit}"
        stem = f"{args.cell}_c{args.concurrency}_r{args.rep}{suffix}"
        with (output / f"{stem}_events.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = sorted({key for row in rows for key in row}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        reads = [row for row in rows if row["op_type"] == "read"]; updates = [row for row in rows if row["op_type"] == "update"]
        status = "PASS" if len(rows)==args.measured_limit and all(row["status"]=="ok" for row in rows) and all(row["graph_projection_ok"]==1 and row["relation_candidate_ok"]==1 for row in updates) else "FAIL"
        summary = {"status": status, "protocol_id": "graph-aware-logical-event-v2-100k-95r5u", "cell": args.cell, "concurrency": args.concurrency, "repetition": args.rep, "warmup_operations": len(warmup), "measured_operations": len(rows), "reads": len(reads), "updates": len(updates), "warmup_seconds": warmup_seconds, "measured_seconds": measured_seconds, "throughput_ops_s": len(rows)/measured_seconds, "latency_p50_ms": percentile([row["latency_ms"] for row in rows],50), "latency_p95_ms": percentile([row["latency_ms"] for row in rows],95), "latency_p99_ms": percentile([row["latency_ms"] for row in rows],99), "read_p50_ms": percentile([row["latency_ms"] for row in reads],50), "read_p95_ms": percentile([row["latency_ms"] for row in reads],95), "read_p99_ms": percentile([row["latency_ms"] for row in reads],99), "update_p50_ms": percentile([row["latency_ms"] for row in updates],50), "update_p95_ms": percentile([row["latency_ms"] for row in updates],95), "update_p99_ms": percentile([row["latency_ms"] for row in updates],99), "fresh_hit_at_10": sum(row["fresh_hit_at_10"] for row in updates)/len(updates), "error_rate": 0.0, "graph_projection_pass": sum(row["graph_projection_ok"] for row in updates), "relation_candidate_pass": sum(row["relation_candidate_ok"] for row in updates)}
        (output / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(summary), flush=True)
        if status != "PASS": raise SystemExit(2)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
