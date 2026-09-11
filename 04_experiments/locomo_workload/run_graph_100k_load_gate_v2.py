"""Load the deterministic 100K graph-aware logical events into four v2 cells."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from graph_event_v2 import GraphRecord, digest
from live_cells_graph_v2 import CassandraGraphCells, Neo4jGraphCells
from run_canonical_live_gate import ROOT, env
from run_graph_canonical_gate_v2 import build_graph_records, distribution

OUT = ROOT / "05_reports" / "locomo_workload_graph_v2_100k"
TARGET = 100_000


def expand(source: list[GraphRecord]) -> list[GraphRecord]:
    result = []
    for ordinal in range(TARGET):
        replica, index = divmod(ordinal, len(source))
        record = source[index]
        prefix = f"lr{replica:06d}::"
        result.append(GraphRecord(
            prefix + record.scope_id, prefix + record.memory_id, record.version,
            record.raw_text, record.entities, record.relations, record.keywords,
            record.triples, record.embedding_sha256,
        ))
    return result


def expected_relation_candidates(records):
    result = defaultdict(set)
    for record in records:
        for edge in record.edges:
            result[(record.scope_id, edge["relation"])].add(record.memory_id)
    return {key: sorted(value) for key, value in result.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--cassandra-workers", type=int, default=32)
    parser.add_argument("--neo4j-workers", type=int, default=12)
    args = parser.parse_args(); env()
    records = expand(build_graph_records())
    scopes = sorted({record.scope_id for record in records})
    expected_relations = expected_relation_candidates(records)
    cass = CassandraGraphCells(os.getenv("CASSANDRA_HOST", "127.0.0.1"))
    neo = Neo4jGraphCells(os.getenv("NEO4J_URI", "bolt://localhost:7687"), os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"), os.getenv("NEO4J_DATABASE", "neo4j"))
    cells = cass.names + neo.names
    stages = []; started = time.time()
    try:
        if args.reset:
            before = time.time(); cass.reset(); neo.reset(); stages.append({"stage": "reset", "seconds": time.time() - before})
        for cell in cass.names:
            before = time.time()
            with ThreadPoolExecutor(max_workers=args.cassandra_workers) as executor:
                for offset, _ in enumerate(executor.map(lambda record: cass.insert(cell, record), records), start=1):
                    if offset % 10_000 == 0: print(f"{cell}={offset}/{TARGET}", flush=True)
            stages.append({"stage": f"load:{cell}", "seconds": time.time() - before})
        for cell in neo.names:
            before = time.time(); neo.insert_many(cell, records, batch=100, workers=args.neo4j_workers)
            stages.append({"stage": f"load:{cell}", "seconds": time.time() - before}); print(f"{cell}={TARGET}/{TARGET}", flush=True)

        memory_counts = {}; namespace_counts = {}
        for cell in cells:
            adapter = cass if cell.startswith("cassandra") else neo
            per_scope = [len(adapter.ids(cell, scope)) for scope in scopes]
            memory_counts[cell] = sum(per_scope); namespace_counts[cell] = sum(value > 0 for value in per_scope)

        sample = [records[(index * 100_003) % TARGET] for index in range(1000)]
        digest_diffs = []
        for record in sample:
            expected = digest(record.graph_projection())
            for cell in cells:
                adapter = cass if cell.startswith("cassandra") else neo
                projection = adapter.fetch_graph(cell, record.scope_id, record.memory_id)
                observed = digest(projection) if projection else ""
                if observed != expected:
                    digest_diffs.append({"memory_id": record.memory_id, "cell": cell, "expected": expected, "observed": observed})

        relation_keys = sorted(expected_relations)
        relation_sample = [relation_keys[(index * 1009) % len(relation_keys)] for index in range(min(1000, len(relation_keys)))]
        candidate_diffs = []
        for scope, relation in relation_sample:
            expected = expected_relations[(scope, relation)]
            for cell in cells:
                adapter = cass if cell.startswith("cassandra") else neo
                observed = adapter.ids_by_relation(cell, scope, relation)
                if observed != expected:
                    candidate_diffs.append({"scope_id": scope, "relation": relation, "cell": cell, "expected_ids": json.dumps(expected), "observed_ids": json.dumps(observed)})

        cass_counts = cass.table_counts(); neo_counts = neo.graph_counts()
        actual_records = {
            "cassandra-base": sum(value for key, value in cass_counts.items() if key.startswith("g_base_")),
            "cassandra-materialized": sum(value for key, value in cass_counts.items() if key.startswith("g_mat_")),
            "neo4j-native": sum(neo_counts[key] for key in ("LWV2NativeMemory", "LWV2NativeFeature", "LWV2NativeEntity", "LWV2_HAS_FEATURE", "LWV2_NATIVE_MENTIONS", "LWV2_NATIVE_REL")),
            "neo4j-materialized": sum(neo_counts[key] for key in ("LWV2MatMemory", "LWV2MatEntity", "LWV2MatCandidate", "LWV2_MAT_MENTIONS", "LWV2_MAT_REL")),
        }
        attempted = {cell: distribution([record.expected_mutations(cell) for record in records]) for cell in cells}
        status = "PASS" if (
            all(value == TARGET for value in memory_counts.values())
            and all(value == 171 for value in namespace_counts.values())
            and not digest_diffs and not candidate_diffs
        ) else "FAIL"
        summary = {
            "status": status, "protocol_id": "graph-aware-logical-event-v2-100k",
            "target_memories": TARGET, "expected_namespaces": 171,
            "edges": sum(len(record.edges) for record in records),
            "memory_counts": memory_counts, "namespace_counts": namespace_counts,
            "graph_digest_sample_memories": len(sample), "graph_digest_comparisons": len(sample) * 4,
            "graph_digest_mismatches": len(digest_diffs),
            "relation_candidate_sample_keys": len(relation_sample), "relation_candidate_comparisons": len(relation_sample) * 4,
            "relation_candidate_mismatches": len(candidate_diffs),
            "attempted_logical_mutations": attempted, "actual_storage_records": actual_records,
            "cassandra_table_counts": cass_counts, "neo4j_graph_counts": neo_counts,
            "stage_times": stages, "elapsed_seconds": time.time() - started,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "graph_100k_load_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for name, rows, fields in (
            ("graph_digest_diffs.csv", digest_diffs, ["memory_id", "cell", "expected", "observed"]),
            ("relation_candidate_diffs.csv", candidate_diffs, ["scope_id", "relation", "cell", "expected_ids", "observed_ids"]),
        ):
            with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        print(json.dumps(summary, indent=2))
        if status != "PASS": raise SystemExit(2)
    finally:
        cass.close(); neo.close()


if __name__ == "__main__":
    main()
