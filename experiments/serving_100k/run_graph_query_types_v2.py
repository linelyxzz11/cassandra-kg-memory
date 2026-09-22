"""Run the four graph-aware query types at the fixed publication concurrency."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cassmem.backend.live_cells_graph import CassandraGraphCells, Neo4jGraphCells
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex
from cassmem.serving.environment import ROOT, env
from run_graph_100k_load_gate_v2 import expand
from run_graph_canonical_gate_v2 import build_graph_records

OUT = ROOT / "results" / "serving_100k" / "locomo_workload_graph_v2_100k" / "query_types_v2" / "runs"
SEED = 20260814


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=["cassandra-base", "cassandra-materialized", "neo4j-native", "neo4j-materialized"])
    parser.add_argument("--rep", required=True, type=int, choices=[0,1,2])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--measured", type=int, default=5000)
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args(); env(); rng = random.Random(SEED + args.rep)
    records = expand(build_graph_records()); by_scope = defaultdict(list); by_id = {}
    relation_expected = defaultdict(set)
    for record in records:
        by_scope[record.scope_id].append(record); by_id[record.memory_id] = record
        for edge in record.edges: relation_expected[(record.scope_id, edge["relation"])].add(record.memory_id)
    scopes = sorted(by_scope); relation_keys = sorted(relation_expected); record_ids = sorted(by_id)
    qa_rows = []
    with (ROOT / "data" / "locomo_qa_records.csv").open(encoding="utf-8-sig", newline="") as handle:
        qa_rows = list(csv.DictReader(handle))
    qa_by_scope = defaultdict(list)
    for replica in range(17):
        prefix = f"lr{replica:06d}::"
        for row in qa_rows: qa_by_scope[prefix + row["sample_id"]].append(row)
    memory_ids = (ROOT / "data" / "locomo_memory_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines(); memory_array = np.load(ROOT / "data" / "locomo_memory_bge_large.npy", mmap_mode="r"); memory_vector = {memory_id: memory_array[index] for index,memory_id in enumerate(memory_ids)}
    qa_ids = (ROOT / "data" / "locomo_qa_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines(); qa_array = np.load(ROOT / "data" / "locomo_qa_bge_large.npy", mmap_mode="r"); qa_vector = {qa_id: qa_array[index] for index,qa_id in enumerate(qa_ids)}
    backend = CassandraGraphCells(os.getenv("CASSANDRA_HOST","127.0.0.1")) if args.cell.startswith("cassandra") else Neo4jGraphCells(os.getenv("NEO4J_URI","bolt://localhost:7687"),os.getenv("NEO4J_USER","neo4j"),os.getenv("NEO4J_PASSWORD"),os.getenv("NEO4J_DATABASE","neo4j"))
    index = ScopedOnlineRetrievalIndex(); print(f"building_index cell={args.cell}", flush=True)
    for scope, scope_records in by_scope.items():
        index.load_scope(scope, {record.memory_id:record.rawerk for record in scope_records}, {record.memory_id:memory_vector[record.memory_id.split("::",1)[1]] for record in scope_records})

    total = args.warmup + args.measured
    operations = {
        "scope_fetch": [rng.choice(scopes) for _ in range(total)],
        "relation_filter": [rng.choice(relation_keys) for _ in range(total)],
        "candidate_projection": [rng.choice(record_ids) for _ in range(total)],
        "top10_retrieval": [],
    }
    qa_scopes = sorted(set(scopes) & set(qa_by_scope))
    for _ in range(total):
        scope = rng.choice(qa_scopes); row = rng.choice(qa_by_scope[scope]); operations["top10_retrieval"].append((scope,row))

    def execute(query_type, item):
        started = time.perf_counter_ns(); ok = True; result_count = 0
        if query_type == "scope_fetch":
            result = backend.ids(args.cell, item); result_count = len(result); ok = result == sorted(record.memory_id for record in by_scope[item])
        elif query_type == "relation_filter":
            scope, relation = item; result = backend.ids_by_relation(args.cell, scope, relation); result_count = len(result); ok = result == sorted(relation_expected[item])
        elif query_type == "candidate_projection":
            record = by_id[item]; result = backend.candidate_projection(args.cell, record.scope_id, record.memory_id); result_count = int(result is not None); ok = result == record.rawerk
        else:
            scope, row = item; candidates = backend.ids(args.cell, scope); result = index.search(scope,row["question"],qa_vector[row["qa_id"]],candidates); result_count = len(result.top10); ok = result_count == 10
        ended = time.perf_counter_ns(); return {"query_type":query_type,"latency_ms":(ended-started)/1e6,"status":"ok" if ok else "mismatch","result_count":result_count}

    output = Path(args.output); output.mkdir(parents=True,exist_ok=True); all_rows=[]; summaries=[]
    try:
        for query_type, items in operations.items():
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                list(executor.map(lambda item: execute(query_type,item), items[:args.warmup]))
            started=time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                rows=list(executor.map(lambda item: execute(query_type,item), items[args.warmup:]))
            seconds=time.perf_counter()-started; all_rows.extend(rows); lat=[row["latency_ms"] for row in rows]
            summaries.append({"status":"PASS" if all(row["status"]=="ok" for row in rows) else "FAIL","protocol_id":"graph-aware-query-types-v2","cell":args.cell,"concurrency":args.concurrency,"repetition":args.rep,"query_type":query_type,"operations":len(rows),"seconds":seconds,"qps":len(rows)/seconds,"p50_ms":percentile(lat,50),"p95_ms":percentile(lat,95),"p99_ms":percentile(lat,99),"error_rate":sum(row["status"]!="ok" for row in rows)/len(rows)})
            print(json.dumps(summaries[-1]),flush=True)
        suffix="" if args.measured==5000 and args.warmup==500 else f"_smoke{args.measured}"; stem=f"{args.cell}_c{args.concurrency}_r{args.rep}{suffix}"
        with (output/f"{stem}_events.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
        (output/f"{stem}_summary.json").write_text(json.dumps({"status":"PASS" if all(x["status"]=="PASS" for x in summaries) else "FAIL","results":summaries},indent=2)+"\n",encoding="utf-8")
        if any(x["status"]!="PASS" for x in summaries): raise SystemExit(2)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
