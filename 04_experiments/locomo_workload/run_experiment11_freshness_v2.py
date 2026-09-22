"""Open-loop graph-aware online freshness benchmark for Experiment 11.

The producer replays a 95:5 read/update stream at controlled arrival rates.
Each update is split into an acknowledged raw-memory commit, structured-view
completion, sparse+dense index visibility, and a real ZScore-RawERK Top-10.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from graph_event_v2 import digest
from live_cells_graph_v2 import CassandraGraphCells, Neo4jGraphCells
from online_retrieval import ScopedOnlineRetrievalIndex
from run_canonical_live_gate import ROOT, env
from run_graph_100k_load_gate_v2 import expand
from run_graph_canonical_gate_v2 import build_graph_records


OUT = ROOT / "05_reports" / "experiment11_online_freshness_v2" / "runs"
DEADLINES_MS = (50, 100, 250, 500, 1000, 2000)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def load_vectors():
    memory_ids = (ROOT / "01_data" / "locomo_memory_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    memory_array = np.load(ROOT / "01_data" / "locomo_memory_bge_large.npy", mmap_mode="r")
    qa_ids = (ROOT / "01_data" / "locomo_qa_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    qa_array = np.load(ROOT / "01_data" / "locomo_qa_bge_large.npy", mmap_mode="r")
    return (
        {memory_id: memory_array[index] for index, memory_id in enumerate(memory_ids)},
        {qa_id: qa_array[index] for index, qa_id in enumerate(qa_ids)},
    )


def source_id(replica_id: str) -> str:
    return replica_id.split("::", 1)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=["cassandra-base", "cassandra-materialized", "neo4j-native", "neo4j-materialized"])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--rates", default="1,2,5,10", help="Update arrivals/s; reads arrive at 19x each rate")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--min-updates-per-scenario", type=int, default=40)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); env()
    rates = [float(value) for value in args.rates.split(",")]
    if args.smoke:
        rates, args.duration, args.repetitions, args.min_updates_per_scenario = [1.0], 2.0, 1, 2

    records = expand(build_graph_records())
    by_id = {record.memory_id: record for record in records}
    by_scope: dict[str, list] = {}
    for record in records:
        by_scope.setdefault(record.scope_id, []).append(record)
    memory_vectors, qa_vectors = load_vectors()

    trace_dir = ROOT / "05_reports" / "locomo_workload_100k" / "traces"
    all_measured = [operation for rep in range(3) for operation in load_jsonl(trace_dir / f"rep{rep}_measured.jsonl")]
    reads = [operation for operation in all_measured if operation["op_type"] == "read"]
    update_candidates = []
    seen_targets = set()
    for operation in all_measured:
        if operation["op_type"] == "update" and operation["target_memory_id"] not in seen_targets:
            update_candidates.append(operation)
            seen_targets.add(operation["target_memory_id"])
    needed = sum(max(args.min_updates_per_scenario, int(round(rate * args.duration))) for rate in rates) * args.repetitions
    if needed > len(update_candidates):
        raise RuntimeError(f"Need {needed} unique update targets but only {len(update_candidates)} are available")
    selected_updates = update_candidates[:needed]
    targets = {operation["target_memory_id"] for operation in selected_updates}

    backend = (
        CassandraGraphCells(os.getenv("CASSANDRA_HOST", "127.0.0.1"))
        if args.cell.startswith("cassandra")
        else Neo4jGraphCells(os.getenv("NEO4J_URI", "bolt://localhost:7687"), os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"), os.getenv("NEO4J_DATABASE", "neo4j"))
    )
    index = ScopedOnlineRetrievalIndex()
    print(json.dumps({"stage": "build_index", "cell": args.cell, "excluded_targets": len(targets)}), flush=True)
    for scope, scope_records in by_scope.items():
        base = [record for record in scope_records if record.memory_id not in targets]
        index.load_scope(
            scope,
            {record.memory_id: record.rawerk for record in base},
            {record.memory_id: memory_vectors[source_id(record.memory_id)] for record in base},
        )
    for target in targets:
        backend.delete(args.cell, by_id[target])

    target_cursor = 0
    read_cursor = 0
    output_rows: list[dict] = []
    scenario_rows: list[dict] = []

    try:
        for repetition in range(args.repetitions):
            for rate in rates:
                update_count = max(args.min_updates_per_scenario, int(round(rate * args.duration)))
                scenario_duration = update_count / rate
                scenario_updates = selected_updates[target_cursor:target_cursor + update_count]
                target_cursor += update_count
                operations = []
                for update in scenario_updates:
                    for _ in range(19):
                        operations.append({"kind": "read", "operation": reads[read_cursor % len(reads)]})
                        read_cursor += 1
                    operations.append({"kind": "update", "operation": update})

                pending = 0
                pending_updates = 0
                peak_pending = 0
                peak_pending_updates = 0
                completed = 0
                lock = threading.Lock()
                scenario_errors: list[str] = []

                def execute(item, enqueued_ns):
                    nonlocal pending, pending_updates, completed
                    started_ns = time.perf_counter_ns()
                    operation = item["operation"]
                    try:
                        scope = operation["scope_id"]
                        if item["kind"] == "read":
                            candidate_ids = backend.ids(args.cell, scope)
                            index.search(scope, operation["question"], qa_vectors[operation["source_qa_id"]], candidate_ids)
                            return None

                        record = by_id[operation["target_memory_id"]]
                        backend.commit_raw(args.cell, record)
                        committed_ns = time.perf_counter_ns()
                        backend.write_structured(args.cell, record)
                        projection = backend.fetch_graph(args.cell, scope, record.memory_id)
                        projection_ok = bool(projection) and digest(projection) == digest(record.graph_projection())
                        candidate_projection = backend.candidate_projection(args.cell, scope, record.memory_id)
                        structured_ok = projection_ok and candidate_projection == record.rawerk
                        relation_ok = all(
                            record.memory_id in backend.ids_by_relation(args.cell, scope, relation)
                            for relation in sorted({edge["relation"] for edge in record.edges})
                        )
                        structured_ns = time.perf_counter_ns()

                        sparse_applied = index.upsert_sparse(scope, record.memory_id, record.rawerk, 2)
                        dense_applied = index.upsert_dense(scope, record.memory_id, memory_vectors[source_id(record.memory_id)], 2)
                        sparse_visible, dense_visible = index.is_visible(scope, record.memory_id, 2)
                        index_ns = time.perf_counter_ns()

                        candidate_ids = backend.ids(args.cell, scope)
                        result, timings = index.search_with_timings(
                            scope, operation["question"], qa_vectors[operation["source_qa_id"]], candidate_ids
                        )
                        completed_ns = time.perf_counter_ns()
                        hit = record.memory_id in {memory_id for memory_id, _ in result.top10}
                        elapsed_ms = (completed_ns - enqueued_ns) / 1e6
                        row = {
                            "cell": args.cell,
                            "concurrency": args.concurrency,
                            "repetition": repetition,
                            "update_rate_s": rate,
                            "read_rate_s": rate * 19.0,
                            "op_id": operation["op_id"],
                            "qa_id": operation["qa_id"],
                            "scope_id": scope,
                            "target_memory_id": record.memory_id,
                            "queue_wait_ms": (started_ns - enqueued_ns) / 1e6,
                            "t_commit_ms": (committed_ns - enqueued_ns) / 1e6,
                            "t_structured_view_visible_ms": (structured_ns - enqueued_ns) / 1e6,
                            "t_index_visible_ms": (index_ns - enqueued_ns) / 1e6,
                            "pipeline_complete_ms": elapsed_ms,
                            "time_to_top10_ms": elapsed_ms if hit else "",
                            "final_hit_at_10": int(hit),
                            "structured_projection_ok": int(structured_ok),
                            "relation_candidate_ok": int(relation_ok),
                            "sparse_index_applied": int(sparse_applied),
                            "dense_index_applied": int(dense_applied),
                            "sparse_index_visible": int(sparse_visible),
                            "dense_index_visible": int(dense_visible),
                            "timeout_5000ms": int(elapsed_ms > 5000.0),
                            **timings,
                        }
                        for deadline in DEADLINES_MS:
                            row[f"fresh_hit_at_10_{deadline}ms"] = int(hit and elapsed_ms <= deadline)
                            row[f"pipeline_visible_{deadline}ms"] = int(elapsed_ms <= deadline)
                        return row
                    except Exception as exc:  # retain failures as measured outcomes
                        scenario_errors.append(f"{type(exc).__name__}: {exc}")
                        return {"error": f"{type(exc).__name__}: {exc}", "kind": item["kind"]}
                    finally:
                        with lock:
                            pending -= 1
                            if item["kind"] == "update":
                                pending_updates -= 1
                            completed += 1

                interval_s = 1.0 / (rate * 20.0)
                futures = []
                scenario_start = time.perf_counter()
                last_enqueue = scenario_start
                with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    for ordinal, item in enumerate(operations):
                        scheduled = scenario_start + ordinal * interval_s
                        delay = scheduled - time.perf_counter()
                        if delay > 0:
                            time.sleep(delay)
                        enqueued_ns = time.perf_counter_ns()
                        last_enqueue = time.perf_counter()
                        with lock:
                            pending += 1
                            if item["kind"] == "update":
                                pending_updates += 1
                            peak_pending = max(peak_pending, pending)
                            peak_pending_updates = max(peak_pending_updates, pending_updates)
                        futures.append(executor.submit(execute, item, enqueued_ns))
                    scenario_results = [future.result() for future in as_completed(futures)]
                scenario_end = time.perf_counter()

                update_rows = [row for row in scenario_results if row and "target_memory_id" in row]
                output_rows.extend(update_rows)
                all_errors = [row for row in scenario_results if row and "error" in row]
                completions = [float(row["pipeline_complete_ms"]) for row in update_rows]
                hits = [row for row in update_rows if int(row["final_hit_at_10"]) == 1]
                summary = {
                    "status": "PASS" if not all_errors and len(update_rows) == update_count else "FAIL",
                    "cell": args.cell,
                    "concurrency": args.concurrency,
                    "repetition": repetition,
                    "update_rate_s": rate,
                    "read_rate_s": rate * 19.0,
                    "total_arrival_rate_s": rate * 20.0,
                    "duration_s": scenario_duration,
                    "operations": len(operations),
                    "updates": update_count,
                    "completed_operations": completed,
                    "update_backlog_peak": peak_pending_updates,
                    "total_backlog_peak": peak_pending,
                    "backlog_drain_ms": max(0.0, (scenario_end - last_enqueue) * 1000.0),
                    "pipeline_complete_p50_ms": percentile(completions, 50),
                    "pipeline_complete_p95_ms": percentile(completions, 95),
                    "pipeline_complete_p99_ms": percentile(completions, 99),
                    "final_hit_at_10": len(hits) / max(len(update_rows), 1),
                    "fresh_hit_at_10_denominator": len(hits),
                    "timeout_rate_5000ms": sum(int(row["timeout_5000ms"]) for row in update_rows) / max(len(update_rows), 1),
                    "error_rate": len(all_errors) / max(len(operations), 1),
                    "missed_update_rate": sum(
                        not (int(row["structured_projection_ok"]) and int(row["relation_candidate_ok"]) and int(row["sparse_index_visible"]) and int(row["dense_index_visible"]))
                        for row in update_rows
                    ) / max(len(update_rows), 1),
                    "duplicate_update_rate": sum(
                        not (int(row["sparse_index_applied"]) and int(row["dense_index_applied"])) for row in update_rows
                    ) / max(len(update_rows), 1),
                }
                for deadline in DEADLINES_MS:
                    summary[f"fresh_hit_at_10_{deadline}ms"] = (
                        sum(int(row[f"fresh_hit_at_10_{deadline}ms"]) for row in hits) / len(hits)
                        if hits else ""
                    )
                    summary[f"pipeline_visible_{deadline}ms"] = sum(int(row[f"pipeline_visible_{deadline}ms"]) for row in update_rows) / max(len(update_rows), 1)
                scenario_rows.append(summary)
                print(json.dumps(summary), flush=True)

        OUT.mkdir(parents=True, exist_ok=True)
        stem = f"{args.cell}_c{args.concurrency}"
        with (OUT / f"{stem}_events.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader(); writer.writerows(output_rows)
        with (OUT / f"{stem}_scenarios.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scenario_rows[0]))
            writer.writeheader(); writer.writerows(scenario_rows)
        manifest = {
            "status": "PASS" if all(row["status"] == "PASS" for row in scenario_rows) else "FAIL",
            "protocol_id": "experiment11-online-freshness-v2",
            "cell": args.cell,
            "concurrency": args.concurrency,
            "rates": rates,
            "duration_s": args.duration,
            "min_updates_per_scenario": args.min_updates_per_scenario,
            "repetitions": args.repetitions,
            "read_update_ratio": "95:5",
            "deadlines_ms": list(DEADLINES_MS),
            "updates": len(output_rows),
            "scenarios": len(scenario_rows),
            "errors": sum(row["error_rate"] > 0 for row in scenario_rows),
        }
        (OUT / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if manifest["status"] != "PASS":
            raise SystemExit(2)
    finally:
        print(json.dumps({"stage": "restore_targets", "cell": args.cell, "targets": len(targets)}), flush=True)
        for target in targets:
            backend.insert(args.cell, by_id[target])
        backend.close()


if __name__ == "__main__":
    main()
