"""Replay frozen 95:5 traces with graph-aware v2 logical event timings."""
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

OUT = ROOT / "05_reports" / "locomo_workload_graph_v2_100k" / "stage_sidecar"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=["cassandra-base", "cassandra-materialized", "neo4j-native", "neo4j-materialized"])
    parser.add_argument("--concurrency", required=True, type=int, choices=[1, 8, 16, 32, 64])
    parser.add_argument("--rep", required=True, type=int, choices=[0, 1, 2])
    args = parser.parse_args(); env()
    records = expand(build_graph_records()); by_id = {record.memory_id: record for record in records}; by_scope = {}
    for record in records: by_scope.setdefault(record.scope_id, []).append(record)
    memory_ids = (ROOT / "01_data" / "locomo_memory_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    memory_array = np.load(ROOT / "01_data" / "locomo_memory_bge_large.npy", mmap_mode="r")
    memory_vector = {memory_id: memory_array[index] for index, memory_id in enumerate(memory_ids)}
    qa_ids = (ROOT / "01_data" / "locomo_qa_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    qa_array = np.load(ROOT / "01_data" / "locomo_qa_bge_large.npy", mmap_mode="r")
    qa_vector = {qa_id: qa_array[index] for index, qa_id in enumerate(qa_ids)}
    trace_dir = ROOT / "05_reports" / "locomo_workload_100k" / "traces"
    warmup = load_jsonl(trace_dir / f"rep{args.rep}_warmup.jsonl")
    measured = load_jsonl(trace_dir / f"rep{args.rep}_measured.jsonl")
    targets = {operation["target_memory_id"] for operation in warmup + measured if operation["op_type"] == "update"}
    backend = CassandraGraphCells(os.getenv("CASSANDRA_HOST", "127.0.0.1")) if args.cell.startswith("cassandra") else Neo4jGraphCells(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"), os.getenv("NEO4J_DATABASE", "neo4j"))
    index = ScopedOnlineRetrievalIndex(); print("building_index", flush=True)
    for scope, scope_records in by_scope.items():
        base = [record for record in scope_records if record.memory_id not in targets]
        index.load_scope(
            scope,
            {record.memory_id: record.rawerk for record in base},
            {record.memory_id: memory_vector[record.memory_id.split("::", 1)[1]] for record in base},
        )
    for target in targets: backend.delete(args.cell, by_id[target])

    def execute(operation, keep):
        scope = operation["scope_id"]
        if operation["op_type"] == "read":
            candidate_ids = backend.ids(args.cell, scope)
            index.search(scope, operation["question"], qa_vector[operation["source_qa_id"]], candidate_ids)
            return None
        record = by_id[operation["target_memory_id"]]
        t0 = time.perf_counter_ns(); backend.insert(args.cell, record); t_commit = time.perf_counter_ns()
        graph_projection = backend.fetch_graph(args.cell, scope, record.memory_id); t_graph = time.perf_counter_ns()
        relation_ok = True
        for relation in sorted({edge["relation"] for edge in record.edges}):
            relation_ok = relation_ok and record.memory_id in backend.ids_by_relation(args.cell, scope, relation)
        t_relation = time.perf_counter_ns()
        index.upsert_sparse(scope, record.memory_id, record.rawerk, 2); t_sparse = time.perf_counter_ns()
        index.upsert_dense(scope, record.memory_id, memory_vector[operation["source_memory_id"]], 2); t_dense = time.perf_counter_ns()
        candidate_ids = backend.ids(args.cell, scope); t_candidates = time.perf_counter_ns()
        result, timings = index.search_with_timings(scope, operation["question"], qa_vector[operation["source_qa_id"]], candidate_ids); t_top10 = time.perf_counter_ns()
        top = {memory_id for memory_id, _ in result.top10}; hit = record.memory_id in top
        if not keep: return None
        return {
            "op_id": operation["op_id"], "cell": args.cell, "concurrency": args.concurrency,
            "repetition": args.rep, "qa_id": operation["qa_id"], "scope_id": scope,
            "target_memory_id": record.memory_id, "edge_count": len(record.edges),
            "logical_mutations": record.expected_mutations(args.cell),
            "commit_ms": (t_commit - t0) / 1e6,
            "graph_visible_ms": (t_graph - t0) / 1e6,
            "relation_candidates_visible_ms": (t_relation - t0) / 1e6,
            "sparse_index_visible_ms": (t_sparse - t0) / 1e6,
            "dense_index_visible_ms": (t_dense - t0) / 1e6,
            "candidate_fetch_ms": (t_candidates - t_dense) / 1e6,
            **timings,
            "time_to_top10_ms": (t_top10 - t0) / 1e6 if hit else "",
            "fresh_hit_at_10": int(hit),
            "graph_projection_ok": int(bool(graph_projection) and digest(graph_projection) == digest(record.graph_projection())),
            "relation_candidate_ok": int(relation_ok),
            "candidate_count": len(candidate_ids),
        }

    def phase(operations, keep):
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            return [row for row in executor.map(lambda operation: execute(operation, keep), operations) if row]

    try:
        phase(warmup, False); rows = phase(measured, True)
        OUT.mkdir(parents=True, exist_ok=True); stem = f"{args.cell}_c{args.concurrency}_r{args.rep}"
        with (OUT / f"{stem}_updates.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        status = "PASS" if len(rows) == 250 and all(row["graph_projection_ok"] == 1 and row["relation_candidate_ok"] == 1 for row in rows) else "FAIL"
        summary = {"status": status, "cell": args.cell, "concurrency": args.concurrency, "repetition": args.rep, "update_events": len(rows), "fresh_hit_at_10": sum(row["fresh_hit_at_10"] for row in rows) / len(rows), "graph_projection_pass": sum(row["graph_projection_ok"] for row in rows), "relation_candidate_pass": sum(row["relation_candidate_ok"] for row in rows)}
        (OUT / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary))
        if status != "PASS": raise SystemExit(2)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
