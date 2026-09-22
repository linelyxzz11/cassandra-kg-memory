"""Controlled recovery benchmark for the graph-aware 100K four-cell design.

The database remains online.  At the failure boundary the update/materializer
worker is forced to stop after committing one raw memory but before completing
its structured/index work.  Arrivals continue during a fixed outage.  A new
worker then replays the partial event and drains the queued logical updates.

This measures application-pipeline recovery; it is not a database-cluster
failover or multi-node availability experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cassmem.representation.graph_event import digest
from cassmem.backend.live_cells_graph import CassandraGraphCells, Neo4jGraphCells
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex
from cassmem.serving.environment import ROOT, env
from run_graph_100k_load_gate_v2 import expand
from run_graph_canonical_gate_v2 import build_graph_records


OUT = ROOT / "results" / "serving_100k" / "locomo_workload_graph_v2_100k" / "recovery_v2" / "runs"
CELLS = ("cassandra-base", "cassandra-materialized", "neo4j-native", "neo4j-materialized")
SLO_MS = 2000.0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_vectors():
    memory_ids = (ROOT / "data" / "locomo_memory_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    memory_array = np.load(ROOT / "data" / "locomo_memory_bge_large.npy", mmap_mode="r")
    qa_ids = (ROOT / "data" / "locomo_qa_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    qa_array = np.load(ROOT / "data" / "locomo_qa_bge_large.npy", mmap_mode="r")
    return (
        {memory_id: memory_array[index] for index, memory_id in enumerate(memory_ids)},
        {qa_id: qa_array[index] for index, qa_id in enumerate(qa_ids)},
    )


def source_id(replica_id: str) -> str:
    return replica_id.split("::", 1)[1]


def phase_for(enqueued_ns: int, stopped_ns: int, restarted_ns: int) -> str:
    if enqueued_ns < stopped_ns:
        return "pre_failure"
    if enqueued_ns < restarted_ns:
        return "outage"
    return "post_restart"


def first_slo_recovery_ms(rows: list[dict], restarted_ns: int, window: int = 5) -> float | None:
    post = sorted(
        (row for row in rows if int(row["completed_ns"]) >= restarted_ns),
        key=lambda row: int(row["completed_ns"]),
    )
    for start in range(0, max(0, len(post) - window + 1)):
        group = post[start:start + window]
        if all(float(row["pipeline_complete_ms"]) <= SLO_MS and row["status"] == "ok" for row in group):
            return (int(group[-1]["completed_ns"]) - restarted_ns) / 1e6
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=CELLS)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--update-rate", type=float, default=1.0)
    parser.add_argument("--pre-seconds", type=float, default=12.0)
    parser.add_argument("--outage-seconds", type=float, default=10.0)
    parser.add_argument("--post-seconds", type=float, default=20.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--probe-count", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    env()
    if args.concurrency < 2:
        raise ValueError("Recovery harness requires at least two workers")
    if args.smoke:
        args.pre_seconds = 1.5
        args.outage_seconds = 1.0
        args.post_seconds = 2.5
        args.repetitions = 1
        args.probe_count = 8

    records = expand(build_graph_records())
    by_id = {record.memory_id: record for record in records}
    by_scope: dict[str, list] = {}
    for record in records:
        by_scope.setdefault(record.scope_id, []).append(record)
    memory_vectors, qa_vectors = load_vectors()

    trace_dir = ROOT / "results" / "serving_100k" / "locomo_workload_100k" / "traces"
    all_ops = [operation for rep in range(3) for operation in load_jsonl(trace_dir / f"rep{rep}_measured.jsonl")]
    reads = [operation for operation in all_ops if operation["op_type"] == "read"]
    update_candidates = []
    seen_targets = set()
    for operation in all_ops:
        if operation["op_type"] == "update" and operation["target_memory_id"] not in seen_targets:
            update_candidates.append(operation)
            seen_targets.add(operation["target_memory_id"])

    total_duration = args.pre_seconds + args.outage_seconds + args.post_seconds
    updates_per_rep = max(3, int(round(args.update_rate * total_duration)))
    needed = updates_per_rep * args.repetitions
    if needed > len(update_candidates):
        raise RuntimeError(f"Need {needed} unique update targets but only {len(update_candidates)} are available")
    selected_updates = update_candidates[:needed]
    selected_targets = {operation["target_memory_id"] for operation in selected_updates}

    backend = (
        CassandraGraphCells(os.getenv("CASSANDRA_HOST", "127.0.0.1"))
        if args.cell.startswith("cassandra")
        else Neo4jGraphCells(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE", "neo4j"),
        )
    )
    index = ScopedOnlineRetrievalIndex()
    print(json.dumps({"stage": "prepare", "cell": args.cell, "excluded_targets": len(selected_targets)}), flush=True)
    for target in selected_targets:
        backend.delete(args.cell, by_id[target])
    for scope, scope_records in by_scope.items():
        base = [record for record in scope_records if record.memory_id not in selected_targets]
        index.load_scope(
            scope,
            {record.memory_id: record.rawerk for record in base},
            {record.memory_id: memory_vectors[source_id(record.memory_id)] for record in base},
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_update_rows: list[dict] = []
    summaries: list[dict] = []
    read_cursor = 0

    try:
        for repetition in range(args.repetitions):
            rep_updates = selected_updates[repetition * updates_per_rep:(repetition + 1) * updates_per_rep]
            update_cursor = 0
            update_queue: queue.Queue = queue.Queue()
            update_rows: list[dict] = []
            partial_attempts: list[dict] = []
            read_rows: list[dict] = []
            row_lock = threading.Lock()
            counter_lock = threading.Lock()
            crash_requested = threading.Event()
            worker_crashed = threading.Event()
            worker_restarted = threading.Event()
            crash_consumed = threading.Event()
            timing: dict[str, int] = {}
            counters = {"backlog_peak": 0, "read_errors": 0, "update_errors": 0}

            def execute_update(item: dict, generation: int) -> tuple[dict | None, bool]:
                operation = item["operation"]
                enqueued_ns = int(item["enqueued_ns"])
                started_ns = time.perf_counter_ns()
                record = by_id[operation["target_memory_id"]]
                scope = operation["scope_id"]
                backend.commit_raw(args.cell, record)
                committed_ns = time.perf_counter_ns()

                if generation == 0 and crash_requested.is_set() and not crash_consumed.is_set():
                    crash_consumed.set()
                    item["replay_count"] = int(item.get("replay_count", 0)) + 1
                    with row_lock:
                        partial_attempts.append({
                            "op_id": operation["op_id"],
                            "target_memory_id": record.memory_id,
                            "enqueued_ns": enqueued_ns,
                            "partial_commit_ns": committed_ns,
                        })
                    update_queue.put(item)
                    return None, True

                try:
                    backend.write_structured(args.cell, record)
                    projection = backend.fetch_graph(args.cell, scope, record.memory_id)
                    candidate_projection = backend.candidate_projection(args.cell, scope, record.memory_id)
                    relation_ok = all(
                        record.memory_id in backend.ids_by_relation(args.cell, scope, relation)
                        for relation in sorted({edge["relation"] for edge in record.edges})
                    )
                    structured_ns = time.perf_counter_ns()
                    sparse_applied = index.upsert_sparse(scope, record.memory_id, record.rawerk, 2)
                    dense_applied = index.upsert_dense(
                        scope, record.memory_id, memory_vectors[source_id(record.memory_id)], 2
                    )
                    sparse_visible, dense_visible = index.is_visible(scope, record.memory_id, 2)
                    index_ns = time.perf_counter_ns()
                    candidate_ids = backend.ids(args.cell, scope)
                    result, search_timings = index.search_with_timings(
                        scope, operation["question"], qa_vectors[operation["source_qa_id"]], candidate_ids
                    )
                    completed_ns = time.perf_counter_ns()
                    top10_ids = [memory_id for memory_id, _ in result.top10]
                    return ({
                        "status": "ok",
                        "cell": args.cell,
                        "repetition": repetition,
                        "op_id": operation["op_id"],
                        "qa_id": operation["qa_id"],
                        "scope_id": scope,
                        "target_memory_id": record.memory_id,
                        "enqueued_ns": enqueued_ns,
                        "started_ns": started_ns,
                        "completed_ns": completed_ns,
                        "replay_count": int(item.get("replay_count", 0)),
                        "queue_wait_ms": (started_ns - enqueued_ns) / 1e6,
                        "t_commit_ms": (committed_ns - enqueued_ns) / 1e6,
                        "t_structured_view_visible_ms": (structured_ns - enqueued_ns) / 1e6,
                        "t_index_visible_ms": (index_ns - enqueued_ns) / 1e6,
                        "pipeline_complete_ms": (completed_ns - enqueued_ns) / 1e6,
                        "time_to_top10_ms": (completed_ns - enqueued_ns) / 1e6 if record.memory_id in top10_ids else "",
                        "final_hit_at_10": int(record.memory_id in top10_ids),
                        "structured_projection_ok": int(bool(projection) and digest(projection) == digest(record.graph_projection())),
                        "candidate_projection_ok": int(candidate_projection == record.rawerk),
                        "relation_candidate_ok": int(relation_ok),
                        "sparse_index_applied": int(sparse_applied),
                        "dense_index_applied": int(dense_applied),
                        "sparse_index_visible": int(sparse_visible),
                        "dense_index_visible": int(dense_visible),
                        "top10_digest": sha256_text(json.dumps(top10_ids, separators=(",", ":"))),
                        **search_timings,
                    }, False)
                except Exception as exc:
                    completed_ns = time.perf_counter_ns()
                    return ({
                        "status": "error",
                        "cell": args.cell,
                        "repetition": repetition,
                        "op_id": operation["op_id"],
                        "qa_id": operation["qa_id"],
                        "scope_id": scope,
                        "target_memory_id": record.memory_id,
                        "enqueued_ns": enqueued_ns,
                        "started_ns": started_ns,
                        "completed_ns": completed_ns,
                        "replay_count": int(item.get("replay_count", 0)),
                        "queue_wait_ms": (started_ns - enqueued_ns) / 1e6,
                        "pipeline_complete_ms": (completed_ns - enqueued_ns) / 1e6,
                        "error": f"{type(exc).__name__}: {exc}",
                    }, False)

            def update_worker(generation: int) -> None:
                while True:
                    item = update_queue.get()
                    try:
                        if item is None:
                            return
                        row, injected_crash = execute_update(item, generation)
                        if row is not None:
                            with row_lock:
                                update_rows.append(row)
                                if row["status"] != "ok":
                                    counters["update_errors"] += 1
                        if injected_crash:
                            timing["worker_stopped_ns"] = time.perf_counter_ns()
                            worker_crashed.set()
                            return
                    finally:
                        update_queue.task_done()

            initial_worker = threading.Thread(target=update_worker, args=(0,), name=f"{args.cell}-update-0")
            initial_worker.start()

            run_started_ns = time.perf_counter_ns()

            def controller() -> None:
                target = run_started_ns / 1e9 + args.pre_seconds
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                timing["failure_requested_ns"] = time.perf_counter_ns()
                crash_requested.set()
                if not worker_crashed.wait(timeout=max(10.0, 3.0 / args.update_rate)):
                    timing["controller_error"] = "worker did not reach injected crash point"
                    return
                initial_worker.join(timeout=30.0)
                timing["worker_stopped_ns"] = timing.get("worker_stopped_ns", time.perf_counter_ns())
                time.sleep(args.outage_seconds)
                timing["worker_restarted_ns"] = time.perf_counter_ns()
                replacement = threading.Thread(target=update_worker, args=(1,), name=f"{args.cell}-update-1")
                timing["replacement_worker"] = replacement
                replacement.start()
                worker_restarted.set()

            controller_thread = threading.Thread(target=controller, name=f"{args.cell}-controller")
            controller_thread.start()

            total_rate = args.update_rate * 20.0
            total_operations = int(round(total_duration * total_rate))
            interval_s = 1.0 / total_rate
            read_futures = []

            def execute_read(operation: dict, enqueued_ns: int) -> dict:
                started_ns = time.perf_counter_ns()
                try:
                    candidate_ids = backend.ids(args.cell, operation["scope_id"])
                    index.search(
                        operation["scope_id"], operation["question"],
                        qa_vectors[operation["source_qa_id"]], candidate_ids,
                    )
                    completed_ns = time.perf_counter_ns()
                    return {"status": "ok", "enqueued_ns": enqueued_ns, "completed_ns": completed_ns,
                            "latency_ms": (completed_ns - enqueued_ns) / 1e6,
                            "queue_wait_ms": (started_ns - enqueued_ns) / 1e6}
                except Exception as exc:
                    completed_ns = time.perf_counter_ns()
                    return {"status": "error", "enqueued_ns": enqueued_ns, "completed_ns": completed_ns,
                            "latency_ms": (completed_ns - enqueued_ns) / 1e6,
                            "error": f"{type(exc).__name__}: {exc}"}

            with ThreadPoolExecutor(max_workers=args.concurrency - 1) as read_executor:
                for ordinal in range(total_operations):
                    scheduled = run_started_ns / 1e9 + ordinal * interval_s
                    delay = scheduled - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    enqueued_ns = time.perf_counter_ns()
                    if (ordinal + 1) % 20 == 0 and update_cursor < len(rep_updates):
                        update_queue.put({"operation": rep_updates[update_cursor], "enqueued_ns": enqueued_ns, "replay_count": 0})
                        update_cursor += 1
                        with counter_lock:
                            counters["backlog_peak"] = max(counters["backlog_peak"], update_queue.qsize())
                    else:
                        operation = reads[read_cursor % len(reads)]
                        read_cursor += 1
                        read_futures.append(read_executor.submit(execute_read, operation, enqueued_ns))
                read_rows.extend(future.result() for future in as_completed(read_futures))

            controller_thread.join(timeout=max(30.0, args.outage_seconds + 15.0))
            if timing.get("controller_error"):
                raise RuntimeError(str(timing["controller_error"]))
            if not worker_restarted.is_set():
                raise RuntimeError("replacement worker was not started")
            update_queue.join()
            replacement = timing["replacement_worker"]
            update_queue.put(None)
            update_queue.join()
            replacement.join(timeout=30.0)

            stopped_ns = int(timing["worker_stopped_ns"])
            restarted_ns = int(timing["worker_restarted_ns"])
            for row in update_rows:
                row["arrival_phase"] = phase_for(int(row["enqueued_ns"]), stopped_ns, restarted_ns)
                row["completed_after_restart"] = int(int(row["completed_ns"]) >= restarted_ns)

            recovery_cohort = [
                row for row in update_rows
                if int(row["enqueued_ns"]) <= restarted_ns or int(row.get("replay_count", 0)) > 0
            ]
            post_rows = [row for row in update_rows if int(row["completed_ns"]) >= restarted_ns]
            first_replay_ns = min((int(row["started_ns"]) for row in recovery_cohort if int(row["started_ns"]) >= restarted_ns), default=None)
            first_top10_ns = min((int(row["completed_ns"]) for row in post_rows if int(row.get("final_hit_at_10", 0)) == 1), default=None)
            drain_ns = max((int(row["completed_ns"]) for row in recovery_cohort), default=restarted_ns)
            ok_updates = [row for row in update_rows if row["status"] == "ok"]
            pre_rows = [row for row in ok_updates if int(row["completed_ns"]) < stopped_ns]
            recovered_slo_ms = first_slo_recovery_ms(
                ok_updates,
                restarted_ns,
                window=2 if args.smoke else 5,
            )
            missed = [row for row in ok_updates if not all(int(row.get(field, 0)) for field in (
                "structured_projection_ok", "candidate_projection_ok", "relation_candidate_ok",
                "sparse_index_visible", "dense_index_visible",
            ))]
            duplicate = [row for row in ok_updates if not all(int(row.get(field, 0)) for field in (
                "sparse_index_applied", "dense_index_applied",
            ))]
            outage_arrivals = sum(stopped_ns <= int(row["enqueued_ns"]) < restarted_ns for row in update_rows)
            timeout_rows = [row for row in ok_updates if float(row["pipeline_complete_ms"]) > 5000.0]
            status = "PASS" if (
                len(update_rows) == updates_per_rep
                and len(partial_attempts) == 1
                and not missed and not duplicate
                and counters["update_errors"] == 0
                and not any(row["status"] != "ok" for row in read_rows)
                and recovered_slo_ms is not None
            ) else "FAIL"
            summary = {
                "status": status,
                "protocol_id": "graph-aware-100k-controlled-recovery-v2",
                "failure_type": "update-materializer-worker-crash-after-raw-commit",
                "claim_boundary": "application-pipeline recovery; database remained online",
                "cell": args.cell,
                "concurrency": args.concurrency,
                "repetition": repetition,
                "update_rate_s": args.update_rate,
                "read_rate_s": args.update_rate * 19.0,
                "total_arrival_rate_s": total_rate,
                "pre_seconds": args.pre_seconds,
                "configured_outage_seconds": args.outage_seconds,
                "post_seconds": args.post_seconds,
                "updates": len(update_rows),
                "reads": len(read_rows),
                "partial_commit_replays": len(partial_attempts),
                "worker_downtime_ms": (restarted_ns - stopped_ns) / 1e6,
                "outage_update_arrivals": outage_arrivals,
                "update_backlog_peak": counters["backlog_peak"],
                "restart_to_first_replay_ms": None if first_replay_ns is None else (first_replay_ns - restarted_ns) / 1e6,
                "restart_to_first_top10_ms": None if first_top10_ns is None else (first_top10_ns - restarted_ns) / 1e6,
                "backlog_drain_ms": max(0.0, (drain_ns - restarted_ns) / 1e6),
                "time_to_slo_recovery_ms": recovered_slo_ms,
                "pre_failure_pipeline_p50_ms": percentile([float(row["pipeline_complete_ms"]) for row in pre_rows], 50),
                "pre_failure_pipeline_p95_ms": percentile([float(row["pipeline_complete_ms"]) for row in pre_rows], 95),
                "post_restart_pipeline_p50_ms": percentile([float(row["pipeline_complete_ms"]) for row in post_rows], 50),
                "post_restart_pipeline_p95_ms": percentile([float(row["pipeline_complete_ms"]) for row in post_rows], 95),
                "final_hit_at_10": sum(int(row["final_hit_at_10"]) for row in ok_updates) / max(len(ok_updates), 1),
                "timeout_rate_5000ms": len(timeout_rows) / max(len(ok_updates), 1),
                "read_error_rate": sum(row["status"] != "ok" for row in read_rows) / max(len(read_rows), 1),
                "update_error_rate": counters["update_errors"] / max(len(update_rows), 1),
                "missed_update_rate": len(missed) / max(len(ok_updates), 1),
                "duplicate_update_rate": len(duplicate) / max(len(ok_updates), 1),
                "projection_gates_passed": len(ok_updates) - len(missed),
            }
            suffix = "_smoke" if args.smoke else ""
            stem = f"{args.cell}_r{repetition}{suffix}"
            fields = sorted({key for row in update_rows for key in row})
            with (output_dir / f"{stem}_events.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(update_rows)
            (output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            (output_dir / f"{stem}_partial_attempts.json").write_text(json.dumps(partial_attempts, indent=2) + "\n", encoding="utf-8")
            all_update_rows.extend(update_rows)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            if status != "PASS":
                raise SystemExit(2)

        probes = []
        seen_probe_qids = set()
        for operation in reads:
            qa_id = operation["source_qa_id"]
            if qa_id in seen_probe_qids:
                continue
            seen_probe_qids.add(qa_id)
            candidate_ids = backend.ids(args.cell, operation["scope_id"])
            result = index.search(operation["scope_id"], operation["question"], qa_vectors[qa_id], candidate_ids)
            top10 = [memory_id for memory_id, _ in result.top10]
            probes.append({
                "cell": args.cell,
                "qa_id": qa_id,
                "scope_id": operation["scope_id"],
                "top10_ids": json.dumps(top10, separators=(",", ":")),
                "top10_digest": sha256_text(json.dumps(top10, separators=(",", ":"))),
            })
            if len(probes) >= args.probe_count:
                break
        suffix = "_smoke" if args.smoke else ""
        with (output_dir / f"{args.cell}{suffix}_post_recovery_probes.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(probes[0]))
            writer.writeheader()
            writer.writerows(probes)
        manifest = {
            "status": "PASS" if all(summary["status"] == "PASS" for summary in summaries) else "FAIL",
            "protocol_id": "graph-aware-100k-controlled-recovery-v2",
            "cell": args.cell,
            "repetitions": args.repetitions,
            "updates_per_repetition": updates_per_rep,
            "total_updates": len(all_update_rows),
            "probe_count": len(probes),
            "smoke": args.smoke,
        }
        (output_dir / f"{args.cell}{suffix}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
