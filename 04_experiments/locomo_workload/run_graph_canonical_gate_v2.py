"""Load and verify the graph-aware v2 four-cell canonical gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from graph_event_v2 import GraphRecord, digest
from live_cells_graph_v2 import CassandraGraphCells, Neo4jGraphCells
from run_canonical_live_gate import ROOT, env

OUT = ROOT / "05_reports" / "locomo_workload_graph_v2_canonical_gate"


def csv_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_graph_records() -> list[GraphRecord]:
    memories = csv_rows(ROOT / "01_data" / "locomo_memory_records.csv")
    features = {row["memory_id"]: row for row in csv_rows(ROOT / "02_artifacts" / "p3_memory_features.csv")}
    ids = (ROOT / "01_data" / "locomo_memory_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    vectors = np.load(ROOT / "01_data" / "locomo_memory_bge_large.npy", mmap_mode="r")
    vector_hashes = {
        memory_id: hashlib.sha256(np.asarray(vectors[index], dtype=np.float32).tobytes()).hexdigest()
        for index, memory_id in enumerate(ids)
    }
    return [
        GraphRecord(
            row["sample_id"], row["memory_id"], 1, row["text"],
            features[row["memory_id"]]["entities"], features[row["memory_id"]]["relations"],
            features[row["memory_id"]]["keywords"], features[row["memory_id"]]["triples"],
            vector_hashes[row["memory_id"]],
        )
        for row in memories
    ]


def expected_relation_candidates(records):
    result = defaultdict(set)
    for record in records:
        for edge in record.edges:
            result[(record.scope_id, edge["relation"])].add(record.memory_id)
    return {key: sorted(value) for key, value in result.items()}


def distribution(values):
    data = np.asarray(values, dtype=float)
    return {
        "total": int(data.sum()), "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)), "p95": float(np.percentile(data, 95)),
        "max": int(data.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cassandra-workers", type=int, default=16)
    parser.add_argument("--neo4j-workers", type=int, default=8)
    args = parser.parse_args()
    env()
    records = build_graph_records()
    if args.limit:
        records = records[:args.limit]
    scopes = sorted({record.scope_id for record in records})
    expected_candidates = expected_relation_candidates(records)
    expected_digests = {record.memory_id: digest(record.graph_projection()) for record in records}
    cass = CassandraGraphCells(os.getenv("CASSANDRA_HOST", "127.0.0.1"))
    neo = Neo4jGraphCells(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"), os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD"), os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    started = time.time()
    stage_times = []
    try:
        if args.reset:
            before = time.time(); cass.reset(); neo.reset(); stage_times.append({"stage": "reset", "seconds": time.time() - before})
        for cell in cass.names:
            before = time.time()
            with ThreadPoolExecutor(max_workers=args.cassandra_workers) as executor:
                list(executor.map(lambda record: cass.insert(cell, record), records))
            stage_times.append({"stage": f"load:{cell}", "seconds": time.time() - before})
        for cell in neo.names:
            before = time.time(); neo.insert_many(cell, records, batch=100, workers=args.neo4j_workers)
            stage_times.append({"stage": f"load:{cell}", "seconds": time.time() - before})

        cells = cass.names + neo.names
        memory_counts = {}
        digest_diffs = []
        candidate_diffs = []
        for cell in cells:
            adapter = cass if cell.startswith("cassandra") else neo
            memory_counts[cell] = sum(len(adapter.ids(cell, scope)) for scope in scopes)
        for record in records:
            for cell in cells:
                adapter = cass if cell.startswith("cassandra") else neo
                observed_projection = adapter.fetch_graph(cell, record.scope_id, record.memory_id)
                observed = digest(observed_projection) if observed_projection else ""
                if observed != expected_digests[record.memory_id]:
                    digest_diffs.append({"memory_id": record.memory_id, "cell": cell, "expected": expected_digests[record.memory_id], "observed": observed})
        for (scope, relation), expected in sorted(expected_candidates.items()):
            for cell in cells:
                adapter = cass if cell.startswith("cassandra") else neo
                observed = adapter.ids_by_relation(cell, scope, relation)
                if observed != expected:
                    candidate_diffs.append({"scope_id": scope, "relation": relation, "cell": cell, "expected_ids": json.dumps(expected), "observed_ids": json.dumps(observed)})

        cass_counts = cass.table_counts()
        neo_counts = neo.graph_counts()
        actual_records = {
            "cassandra-base": sum(value for key, value in cass_counts.items() if key.startswith("g_base_")),
            "cassandra-materialized": sum(value for key, value in cass_counts.items() if key.startswith("g_mat_")),
            "neo4j-native": sum(neo_counts[key] for key in ("LWV2NativeMemory", "LWV2NativeFeature", "LWV2NativeEntity", "LWV2_HAS_FEATURE", "LWV2_NATIVE_MENTIONS", "LWV2_NATIVE_REL")),
            "neo4j-materialized": sum(neo_counts[key] for key in ("LWV2MatMemory", "LWV2MatEntity", "LWV2MatCandidate", "LWV2_MAT_MENTIONS", "LWV2_MAT_REL")),
        }
        attempted = {cell: distribution([record.expected_mutations(cell) for record in records]) for cell in cells}
        expected_count = len(records)
        full_run = expected_count == 5882
        status = "PASS" if all(value == expected_count for value in memory_counts.values()) and not digest_diffs and not candidate_diffs else "FAIL"
        payload = {
            "status": status, "protocol_id": "graph-aware-logical-event-v2",
            "full_canonical_run": full_run, "memories": expected_count,
            "scopes": len(scopes), "relations": len(expected_candidates),
            "edges": sum(len(record.edges) for record in records),
            "memory_counts": memory_counts,
            "graph_digest_comparisons": expected_count * 4,
            "graph_digest_mismatches": len(digest_diffs),
            "relation_candidate_comparisons": len(expected_candidates) * 4,
            "relation_candidate_mismatches": len(candidate_diffs),
            "attempted_logical_mutations": attempted,
            "actual_storage_records": actual_records,
            "cassandra_table_counts": cass_counts, "neo4j_graph_counts": neo_counts,
            "stage_times": stage_times, "elapsed_seconds": time.time() - started,
        }
        target = OUT if full_run else OUT / "smoke"
        target.mkdir(parents=True, exist_ok=True)
        (target / "graph_canonical_gate_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        for name, rows, fields in (
            ("graph_digest_diffs.csv", digest_diffs, ["memory_id", "cell", "expected", "observed"]),
            ("relation_candidate_diffs.csv", candidate_diffs, ["scope_id", "relation", "cell", "expected_ids", "observed_ids"]),
        ):
            with (target / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        print(json.dumps(payload, indent=2))
        if status != "PASS":
            raise SystemExit(2)
    finally:
        cass.close(); neo.close()


if __name__ == "__main__":
    main()
