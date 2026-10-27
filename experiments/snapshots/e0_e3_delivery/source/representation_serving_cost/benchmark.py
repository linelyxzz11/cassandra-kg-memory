"""Cassandra-backed three-view benchmark; preserves the common canonical contract.

Only repcost_* keyspaces on the isolated port 19042 are accepted. Existing
repository adapters/scorers are reused without source modifications.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import pickle
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from common import ARMS, HERE, ROOT, build_graph_records, index_space, lexical, np, rows, source_manifest, vectors, write_csv, write_json
from cassmem.backend.live_cells_graph import CassandraGraphCells
from cassmem.representation.graph_event import digest
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex
from run_graph_100k_load_gate_v2 import expand
import psutil

os.environ["CASSANDRA_PORT"] = "19042"


def backend(keyspace):
    if not keyspace.startswith("repcost_") or not keyspace.replace("_", "").isalnum():
        raise ValueError("Only isolated repcost_* keyspaces are allowed")
    b = CassandraGraphCells("127.0.0.1", keyspace=keyspace)
    from cassandra import ConsistencyLevel
    b.s.default_consistency_level = ConsistencyLevel.LOCAL_ONE
    b.s.default_timeout = 60
    for p in b.p.values():
        p.consistency_level = ConsistencyLevel.LOCAL_ONE
    return b


def data(n):
    original = build_graph_records()
    records = expand(original) if n == 100000 else original[:n]
    grouped = defaultdict(list)
    for r in records:
        grouped[r.scope_id].append(r)
    queries = defaultdict(list)
    for q in rows(ROOT / "data/frozen_retrieval/p3_queries.csv"):
        queries[q["conversation_id"]].append(q)
    assert all(s.split("::")[-1] in queries for s in grouped)
    return records, grouped, queries, vectors("memory"), vectors("qa")


def make_index(grouped, memory_vectors, arm):
    index = ScopedOnlineRetrievalIndex()
    for scope, recs in grouped.items():
        index.load_scope(scope, {r.memory_id: lexical(r, arm) for r in recs},
                         {r.memory_id: memory_vectors[r.memory_id.split("::")[-1]] for r in recs})
    return index


def load(args):
    records, grouped, _, _, _ = data(args.memories)
    b = backend(args.keyspace)
    try:
        for cell in b.names:
            count = sum(len(b.ids(cell, s)) for s in grouped)
            if count:
                raise RuntimeError(f"Refusing to reload nonempty {args.keyspace}/{cell}: {count}")
        stages = []
        for cell in b.names:
            began = time.perf_counter()
            with ThreadPoolExecutor(max_workers=16) as pool:
                for offset in range(0, len(records), 256):
                    list(pool.map(lambda r: b.insert(cell, r), records[offset:offset+256]))
                    if offset % 5120 == 0:
                        print(f"Load {cell}: {min(offset+256,len(records))}/{len(records)}", flush=True)
            assert sum(len(b.ids(cell, s)) for s in grouped) == len(records)
            for r in records[::max(1, len(records)//100)]:
                assert digest(b.fetch_graph(cell, r.scope_id, r.memory_id)) == digest(r.graph_projection())
            stages.append({"cell": cell, "memories": len(records), "seconds": time.perf_counter()-began})
        write_json(args.output / "load.json", {"status": "PASS", "keyspace": args.keyspace, "stages": stages})
    finally:
        b.close()


class Trial:
    def __init__(self, args):
        self.args = args
        self.records, self.grouped, self.queries, self.mv, self.qv = data(args.memories)
        self.scopes = sorted(self.grouped)
        self.b = backend(args.keyspace)
        self.index = make_index(self.grouped, self.mv, args.arm)
        self.touched = {}
        self.lock = threading.Lock()
        self.total_updates = self.pending = self.queued = 0
        self.peak_pending = self.peak_queued = 0
        self.events = []

    def operation(self, ordinal, phase, rng):
        scope = self.scopes[rng.randrange(len(self.scopes))]
        query = rng.choice(self.queries[scope.split("::")[-1]])
        source = rng.choice(self.grouped[scope])
        update = ordinal % 20 == 19
        target = dataclasses.replace(source, memory_id=f"repcost_{phase}_{ordinal:09d}::{source.memory_id}", version=2) if update else None
        return {"ordinal": ordinal, "phase": phase, "scope": scope, "query": query, "source": source,
                "target": target, "op_type": "update" if update else "read"}

    def register(self, op, arrival):
        op["arrival_ns"] = arrival
        op["submit_ns"] = time.perf_counter_ns()
        if op["target"]:
            with self.lock:
                self.pending += 1
                self.queued += 1
                self.total_updates += 1
                self.peak_pending = max(self.peak_pending, self.pending)
                self.peak_queued = max(self.peak_queued, self.queued)
        return op

    def execute(self, op):
        started = time.perf_counter_ns()
        row = {"ordinal": op["ordinal"], "phase": op["phase"], "op_type": op["op_type"],
               "scope": op["scope"], "query_id": op["query"]["query_id"], "arrival_ns": op["arrival_ns"],
               "submit_ns": op["submit_ns"], "worker_start_ns": started,
               "queue_wait_ms": (started-op["arrival_ns"])/1e6,
               "submission_lag_ms": (op["submit_ns"]-op["arrival_ns"])/1e6,
               "status": "error", "ready": False, "post_query_ok": False}
        target = op["target"]
        if target:
            with self.lock:
                self.queued -= 1
                self.touched[target.memory_id] = target
        try:
            if target:
                row["memory_id"] = target.memory_id
                row["raw_call_start_ns"] = time.perf_counter_ns()
                self.b.commit_raw(self.args.cell, target)
                row["raw_ack_ns"] = time.perf_counter_ns()
                self.b.write_structured(self.args.cell, target)
                row["structured_write_end_ns"] = time.perf_counter_ns()
                actual = self.b.fetch_graph(self.args.cell, target.scope_id, target.memory_id)
                assert actual is not None and digest(actual) == digest(target.graph_projection()), "canonical mismatch"
                row["projection_fetch_end_ns"] = time.perf_counter_ns()
                for rel in sorted({e["relation"] for e in target.edges}):
                    assert target.memory_id in self.b.ids_by_relation(self.args.cell, target.scope_id, rel), "relation mismatch"
                row["relation_checked_ns"] = time.perf_counter_ns()
                canonical_text = self.b.candidate_projection(self.args.cell, target.scope_id, target.memory_id)
                assert canonical_text == target.rawerk, "canonical text mismatch"
                row["projection_access_end_ns"] = time.perf_counter_ns()
                # Generate lexical input from the actual backend fields, not merely the input event.
                from cassmem.representation.graph_event import GraphRecord
                observed = GraphRecord(**{k: actual["memory"][k] for k in GraphRecord.__dataclass_fields__})
                document = lexical(observed, self.args.arm)
                vector = self.mv[op["source"].memory_id.split("::")[-1]]
                assert hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest() == observed.embedding_sha256
                assert self.index.upsert_sparse(target.scope_id, target.memory_id, document, target.version)
                row["sparse_updated_ns"] = time.perf_counter_ns()
                assert self.index.upsert_dense(target.scope_id, target.memory_id, vector, target.version)
                row["dense_updated_ns"] = time.perf_counter_ns()
                with self.index._scope_locks[target.scope_id]:
                    assert all(self.index.is_visible(target.scope_id, target.memory_id, target.version))
                    snapshot = self.index._sparse[target.scope_id]
                    assert snapshot.documents[snapshot.memory_ids.index(target.memory_id)] == document
                    dense = self.index._dense[target.scope_id]
                    expected = vector / max(float(np.linalg.norm(vector)), 1e-12)
                    assert np.allclose(dense.normalized_embeddings[dense.memory_ids.index(target.memory_id)], expected, rtol=1e-6, atol=1e-7)
                row["ready_checked_ns"] = time.perf_counter_ns()
                row["ready"] = True
                for field, begin, end in (
                    ("raw_write_call_ms", "raw_call_start_ns", "raw_ack_ns"),
                    ("arrival_to_raw_ack_ms", "arrival_ns", "raw_ack_ns"),
                    ("structured_write_ms", "raw_ack_ns", "structured_write_end_ns"),
                    ("projection_fetch_ms", "structured_write_end_ns", "projection_fetch_end_ns"),
                    ("relation_check_ms", "projection_fetch_end_ns", "relation_checked_ns"),
                    ("projection_access_ms", "relation_checked_ns", "projection_access_end_ns"),
                    ("sparse_update_ms", "projection_access_end_ns", "sparse_updated_ns"),
                    ("dense_update_ms", "sparse_updated_ns", "dense_updated_ns"),
                    ("ack_to_retrievable_ms", "raw_ack_ns", "ready_checked_ns"),
                    ("arrival_to_retrievable_ms", "arrival_ns", "ready_checked_ns"),
                    ("worker_to_retrievable_ms", "worker_start_ns", "ready_checked_ns"),
                ):
                    row[field] = (row[end]-row[begin])/1e6
                row["ready_within_500ms"] = row["arrival_to_retrievable_ms"] <= 500
            fetch_start = time.perf_counter_ns()
            candidates = self.b.ids(self.args.cell, op["scope"])
            row["candidate_access_ms"] = (time.perf_counter_ns()-fetch_start)/1e6
            result, timings = self.index.search_with_timings(op["scope"], op["query"]["question"],
                                                          self.qv[op["query"]["query_id"]], candidates)
            row.update(timings, candidate_count=len(candidates), status="ok", top10=";".join(x[0] for x in result.top10))
            if target:
                row["post_query_ok"] = True
                row["target_in_top10"] = target.memory_id in {mid for mid, _ in result.top10}
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            row["end_ns"] = time.perf_counter_ns()
            row["service_ms"] = (row["end_ns"]-started)/1e6
            if target:
                with self.lock:
                    self.pending -= 1
        return row

    def cleanup(self):
        # Only IDs inserted by this trial are deleted, never original rows.
        touched = list(self.touched.values())
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda r: self.b.delete(self.args.cell, r), touched))
        self.touched.clear()

    def phase(self, phase, duration, rate=0):
        rng = random.Random(20261012 + self.args.rep + (10000 if phase == "measured" else 0))
        output = []
        self.peak_pending = self.peak_queued = self.total_updates = 0
        start = time.perf_counter_ns()
        limit = start + int(duration * 1e9)
        ordinal = 0
        futures = set()
        deadline_exceeded = False
        with ThreadPoolExecutor(max_workers=self.args.concurrency) as pool:
            if rate:
                # Fixed offered updates and exactly 19 scheduled reads per update.
                count = int(round(duration * rate)) * 20
                for ordinal in range(count):
                    arrival = start + int(ordinal / (rate * 20) * 1e9)
                    remaining = (arrival-time.perf_counter_ns())/1e9
                    if remaining > 0:
                        time.sleep(remaining)
                    op = self.register(self.operation(ordinal, phase, rng), arrival)
                    futures.add(pool.submit(self.execute, op))
                    done = {f for f in futures if f.done()}
                    output.extend(f.result() for f in done)
                    futures -= done
                done, pending = wait(futures, timeout=self.args.drain_seconds)
                output.extend(f.result() for f in done)
                if pending:
                    deadline_exceeded = True
                    # Preserve all eventual event clocks and explicitly mark deadline failure.
                    output.extend(f.result() for f in pending)
            else:
                while time.perf_counter_ns() < limit or futures:
                    while len(futures) < self.args.concurrency and time.perf_counter_ns() < limit:
                        op = self.operation(ordinal, phase, rng)
                        ordinal += 1
                        futures.add(pool.submit(self.execute, self.register(op, time.perf_counter_ns())))
                    if futures:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        output.extend(f.result() for f in done)
        elapsed = (time.perf_counter_ns()-start)/1e9
        drain_deadline = limit + int(self.args.drain_seconds * 1e9)
        for row in output:
            row["ready_censored_at_deadline"] = bool(row["op_type"] == "update" and
                (not row["ready"] or row.get("ready_checked_ns", drain_deadline+1) > drain_deadline))
        return output, {"wall_seconds_including_drain": elapsed, "offered_window_seconds": duration,
                        "peak_pending_updates": self.peak_pending, "peak_queued_updates": self.peak_queued,
                        "drain_deadline_exceeded": deadline_exceeded}


def summarize(events, context):
    reads = [e for e in events if e["op_type"] == "read"]
    updates = [e for e in events if e["op_type"] == "update"]
    ready = [e for e in updates if e["ready"]]
    post = [e for e in updates if e["post_query_ok"]]
    result = {**context, "operations": len(events), "reads": len(reads), "updates": len(updates),
              "errors": sum(e["status"] != "ok" for e in events), "ready_updates": len(ready),
              "ready_censored_at_deadline": sum(e.get("ready_censored_at_deadline", False) for e in updates),
              "ready_within_500ms_count": sum(e.get("ready_within_500ms", False) for e in updates),
              "ready_within_500ms_fraction": sum(e.get("ready_within_500ms", False) for e in updates)/len(updates) if updates else None,
              "post_update_queries": len(post), "post_update_top10_hits": sum(e.get("target_in_top10", False) for e in post)}
    duration = context["wall_seconds_including_drain"]
    result.update(throughput_ops_s=sum(e["status"] == "ok" for e in events)/duration,
                  query_qps=sum(e["status"] == "ok" for e in reads)/duration,
                  update_qps=sum(e["status"] == "ok" for e in updates)/duration)
    for group_name, group in (("read", reads), ("update", updates)):
        fields = {key for e in group for key in e if key.endswith("_ms")}
        for field in sorted(fields):
            values = [e[field] for e in group if field in e]
            for quantile in (50, 95, 99):
                result[f"{group_name}_{field}_p{quantile}"] = float(np.percentile(values, quantile))
    result["status"] = "PASS" if not result["errors"] and len(ready) == len(updates) and not context["drain_deadline_exceeded"] else "FAIL"
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", {"status": "RUNNING", "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, "sources": source_manifest()})
    trial = Trial(args)
    stop = threading.Event()
    samples = []
    def sample_memory():
        proc = psutil.Process()
        while not stop.wait(1):
            samples.append({"time_ns": time.perf_counter_ns(), "rss_bytes": proc.memory_info().rss,
                            "system_available_bytes": psutil.virtual_memory().available})
    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    try:
        assert sum(len(trial.b.ids(args.cell, s)) for s in trial.grouped) == args.memories, "initial count mismatch"
        space = index_space(trial.index)
        space["sparse_snapshot_pickle5_bytes"] = len(pickle.dumps(trial.index._sparse, protocol=5))
        write_json(args.output / "initial_space.json", space)
        if args.warmup_seconds:
            warm, warm_context = trial.phase("warmup", args.warmup_seconds)
            write_csv(args.output / "warmup_events.csv", warm)
            assert all(e["status"] == "ok" for e in warm), "warm-up failed"
            trial.cleanup()
            del trial.index
            gc.collect()
            trial.index = make_index(trial.grouped, trial.mv, args.arm)
        events, context = trial.phase("measured", args.seconds, args.rate)
        write_csv(args.output / "events.csv", sorted(events, key=lambda e:e["ordinal"]))
        summary = summarize(events, context)
        summary.update(arm=args.arm, cell=args.cell, concurrency=args.concurrency, repetition=args.rep,
                       offered_updates_per_second=args.rate, initial_memories=args.memories,
                       final_memories=args.memories+len(trial.touched), lexical_cost_scope="marginal-under-common-canonical-contract")
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        if summary["status"] != "PASS":
            raise RuntimeError("Trial correctness/deadline gate failed; inspect saved events")
    finally:
        stop.set()
        monitor.join(timeout=2)
        write_csv(args.output / "client_memory.csv", samples)
        try:
            trial.cleanup()
        finally:
            trial.b.close()


def gate(args):
    """Frozen-layout parity and real insert/readiness checks for every arm."""
    args.output.mkdir(parents=True, exist_ok=False)
    records, grouped, queries, mv, qv = data(args.memories)
    b = backend(args.keyspace)
    report = []
    try:
        for arm in ARMS:
            index = make_index(grouped, mv, arm)
            for scope in sorted(grouped):
                for query in queries[scope.split("::")[-1]][:5]:
                    base = b.ids("cassandra-base", scope)
                    mat = b.ids("cassandra-materialized", scope)
                    assert base == mat == sorted(r.memory_id for r in grouped[scope])
                    a = index.search(scope, query["question"], qv[query["query_id"]], base)
                    z = index.search(scope, query["question"], qv[query["query_id"]], mat)
                    assert a.top10 == z.top10
                    report.append({"arm": arm, "scope": scope, "query": query["query_id"], "ordered_top10_equal": True})
            del index
        write_csv(args.output / "layout_parity.csv", report)
    finally:
        b.close()
    for arm in ARMS:
        for cell in ("cassandra-base", "cassandra-materialized"):
            child = argparse.Namespace(**vars(args))
            child.arm, child.cell = arm, cell
            child.concurrency, child.rep, child.warmup_seconds, child.seconds, child.rate, child.drain_seconds = 2, 0, 0, 2, 1, 60
            child.output = args.output / f"{arm}_{cell}"
            run(child)
    write_json(args.output / "gate.json", {"status": "PASS", "frozen_query_comparisons": len(report), "arms": ARMS, "layouts": 2, "updates_per_cell": 2})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("load", "gate", "run"))
    parser.add_argument("--keyspace", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memories", type=int, default=100000)
    parser.add_argument("--arm", choices=ARMS, default="RawERK")
    parser.add_argument("--cell", choices=("cassandra-base", "cassandra-materialized"), default="cassandra-materialized")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--warmup-seconds", type=float, default=60)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--rate", type=float, default=0)
    parser.add_argument("--drain-seconds", type=float, default=120)
    args = parser.parse_args()
    {"load": load, "gate": gate, "run": run}[args.action](args)


if __name__ == "__main__":
    main()
