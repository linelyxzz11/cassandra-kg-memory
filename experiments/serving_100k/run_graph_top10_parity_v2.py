"""Verify full-scope and relation-candidate ZScore Top-10 parity on graph v2."""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

from cassmem.backend.live_cells_graph import CassandraGraphCells, Neo4jGraphCells
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex
from cassmem.serving.environment import ROOT, env
from run_graph_canonical_gate_v2 import build_graph_records, csv_rows

OUT = ROOT / "results" / "serving_100k" / "locomo_workload_graph_v2_canonical_gate"


def main():
    env()
    records = build_graph_records()
    by_id = {record.memory_id: record for record in records}
    by_scope = defaultdict(list)
    relations_by_scope = defaultdict(set)
    for record in records:
        by_scope[record.scope_id].append(record)
        for edge in record.edges:
            relations_by_scope[record.scope_id].add(edge["relation"])

    memory_ids = (ROOT / "data" / "locomo_memory_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    memory_vectors = np.load(ROOT / "data" / "locomo_memory_bge_large.npy", mmap_mode="r")
    memory_vector = {memory_id: memory_vectors[index] for index, memory_id in enumerate(memory_ids)}
    qa_ids = (ROOT / "data" / "locomo_qa_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    qa_vectors = np.load(ROOT / "data" / "locomo_qa_bge_large.npy", mmap_mode="r")
    qa_vector = {qa_id: qa_vectors[index] for index, qa_id in enumerate(qa_ids)}
    eligible = csv_rows(ROOT / "results" / "serving_100k" / "locomo_workload_profile" / "freshness_eligible_cat1_4.csv")

    indexes = {}
    expected_scope_ids = {}
    expected_relation_ids = defaultdict(lambda: defaultdict(set))
    for scope, scope_records in by_scope.items():
        index = ScopedOnlineRetrievalIndex()
        index.load_scope(
            scope,
            {record.memory_id: record.rawerk for record in scope_records},
            {record.memory_id: memory_vector[record.memory_id] for record in scope_records},
        )
        indexes[scope] = index
        expected_scope_ids[scope] = sorted(record.memory_id for record in scope_records)
        for record in scope_records:
            for edge in record.edges:
                expected_relation_ids[scope][edge["relation"]].add(record.memory_id)

    cass = CassandraGraphCells(os.getenv("CASSANDRA_HOST", "127.0.0.1"))
    neo = Neo4jGraphCells(os.getenv("NEO4J_URI", "bolt://localhost:7687"), os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"), os.getenv("NEO4J_DATABASE", "neo4j"))
    cells = cass.names + neo.names
    rows = []
    try:
        for offset, question in enumerate(eligible, start=1):
            gold = json.loads(question["gold_memory_ids"])[0]
            scope = by_id[gold].scope_id
            index = indexes[scope]
            expected_full = index.search(scope, question["question"], qa_vector[question["qa_id"]], expected_scope_ids[scope])
            expected_full_top10 = [memory_id for memory_id, _ in expected_full.top10]

            normalized_question = question["question"].lower()
            probe_relation = ""
            for relation in sorted(relations_by_scope[scope]):
                phrase = relation.replace("_", " ").lower()
                if phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized_question):
                    probe_relation = relation
                    break
            expected_relation_candidates = sorted(expected_relation_ids[scope].get(probe_relation, set())) if probe_relation else []
            expected_relation_top10 = []
            if expected_relation_candidates:
                expected_relation_top10 = [memory_id for memory_id, _ in index.search(scope, question["question"], qa_vector[question["qa_id"]], expected_relation_candidates).top10]

            for cell in cells:
                adapter = cass if cell.startswith("cassandra") else neo
                observed_full_candidates = adapter.ids(cell, scope)
                observed_full_top10 = [memory_id for memory_id, _ in index.search(scope, question["question"], qa_vector[question["qa_id"]], observed_full_candidates).top10]
                observed_relation_candidates = adapter.ids_by_relation(cell, scope, probe_relation) if probe_relation else []
                observed_relation_top10 = []
                if observed_relation_candidates:
                    observed_relation_top10 = [memory_id for memory_id, _ in index.search(scope, question["question"], qa_vector[question["qa_id"]], observed_relation_candidates).top10]
                rows.append({
                    "qa_id": question["qa_id"], "cell": cell, "scope_id": scope,
                    "full_candidate_parity": int(observed_full_candidates == expected_scope_ids[scope]),
                    "full_top10_parity": int(observed_full_top10 == expected_full_top10),
                    "probe_relation": probe_relation,
                    "relation_candidate_count": len(observed_relation_candidates),
                    "relation_candidate_parity": int(observed_relation_candidates == expected_relation_candidates) if probe_relation else "",
                    "relation_top10_parity": int(observed_relation_top10 == expected_relation_top10) if probe_relation else "",
                })
            if offset % 100 == 0:
                print(f"completed={offset}/{len(eligible)}", flush=True)
    finally:
        cass.close(); neo.close()

    relation_rows = [row for row in rows if row["probe_relation"]]
    status = "PASS" if (
        len(rows) == len(eligible) * 4
        and all(row["full_candidate_parity"] == 1 and row["full_top10_parity"] == 1 for row in rows)
        and all(row["relation_candidate_parity"] == 1 and row["relation_top10_parity"] == 1 for row in relation_rows)
    ) else "FAIL"
    summary = {
        "status": status, "questions": len(eligible), "comparisons": len(rows),
        "full_candidate_parity_pass": sum(row["full_candidate_parity"] for row in rows),
        "full_top10_parity_pass": sum(row["full_top10_parity"] for row in rows),
        "relation_probe_questions": len(relation_rows) // 4,
        "relation_candidate_parity_pass": sum(row["relation_candidate_parity"] for row in relation_rows),
        "relation_top10_parity_pass": sum(row["relation_top10_parity"] for row in relation_rows),
        "relation_probe_policy": "first lexicographically sorted scope relation occurring as a phrase in the question; parity-only, not a retrieval-effectiveness claim",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "graph_top10_parity_events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / "graph_top10_parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
