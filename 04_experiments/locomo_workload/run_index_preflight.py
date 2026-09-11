"""Diagnostic preflight for the shared scoped online index.

This is not a citation-ready backend result.  It isolates index build/update
and real ZScore Top-10 costs on canonical LoCoMo scopes so the live 2x2 runner
can be sized safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from online_retrieval import ScopedOnlineRetrievalIndex, render_rawerk


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "05_reports" / "locomo_workload_preflight"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summarize(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "mean_ms": statistics.mean(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    memories = read_csv(ROOT / "01_data" / "locomo_memory_records.csv")
    features = {row["memory_id"]: row for row in read_csv(ROOT / "02_artifacts" / "p3_memory_features.csv")}
    eligible = read_csv(ROOT / "05_reports" / "locomo_workload_profile" / "freshness_eligible_cat1_4.csv")
    memory_ids = (ROOT / "01_data" / "locomo_memory_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    qa_ids = (ROOT / "01_data" / "locomo_qa_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    memory_vectors = np.load(ROOT / "01_data" / "locomo_memory_bge_large.npy", mmap_mode="r")
    qa_vectors = np.load(ROOT / "01_data" / "locomo_qa_bge_large.npy", mmap_mode="r")
    memory_vector_by_id = {memory_id: memory_vectors[index] for index, memory_id in enumerate(memory_ids)}
    qa_vector_by_id = {qa_id: qa_vectors[index] for index, qa_id in enumerate(qa_ids)}

    memories_by_scope: dict[str, list[dict[str, str]]] = {}
    for row in memories:
        memories_by_scope.setdefault(row["sample_id"], []).append(row)

    # Round-robin categories/scopes rather than taking only the first file rows.
    eligible.sort(key=lambda row: (row["category"], row["sample_id"], row["qa_id"]))
    stride = max(1, len(eligible) // max(args.events, 1))
    selected = eligible[::stride][: args.events]
    results: list[dict[str, object]] = []

    for ordinal, question in enumerate(selected):
        gold_ids = json.loads(question["gold_memory_ids"])
        gold_id = gold_ids[0]
        scope = question["sample_id"]
        baseline_rows = [row for row in memories_by_scope[scope] if row["memory_id"] != gold_id]
        documents = {
            row["memory_id"]: render_rawerk(
                row["text"],
                features[row["memory_id"]].get("entities", ""),
                features[row["memory_id"]].get("relations", ""),
                features[row["memory_id"]].get("keywords", ""),
            )
            for row in baseline_rows
        }
        embeddings = {row["memory_id"]: memory_vector_by_id[row["memory_id"]] for row in baseline_rows}
        index = ScopedOnlineRetrievalIndex()

        start = time.perf_counter_ns()
        index.load_scope(scope, documents, embeddings)
        baseline_build_ms = (time.perf_counter_ns() - start) / 1e6

        gold_row = next(row for row in memories_by_scope[scope] if row["memory_id"] == gold_id)
        gold_feature = features[gold_id]
        gold_document = render_rawerk(
            gold_row["text"],
            gold_feature.get("entities", ""),
            gold_feature.get("relations", ""),
            gold_feature.get("keywords", ""),
        )
        start = time.perf_counter_ns()
        index.upsert_sparse(scope, gold_id, gold_document, 2)
        sparse_update_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        index.upsert_dense(scope, gold_id, memory_vector_by_id[gold_id], 2)
        dense_update_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        search = index.search(scope, question["question"], qa_vector_by_id[question["qa_id"]])
        fusion_search_ms = (time.perf_counter_ns() - start) / 1e6
        top_ids = [memory_id for memory_id, _ in search.top10]
        results.append({
            "ordinal": ordinal,
            "qa_id": question["qa_id"],
            "category": question["category"],
            "scope_id": scope,
            "scope_memory_count": len(baseline_rows) + 1,
            "gold_memory_id": gold_id,
            "baseline_build_ms": baseline_build_ms,
            "sparse_update_ms": sparse_update_ms,
            "dense_update_ms": dense_update_ms,
            "fusion_search_ms": fusion_search_ms,
            "index_plus_search_ms": sparse_update_ms + dense_update_ms + fusion_search_ms,
            "fresh_hit_at_10": int(gold_id in top_ids),
            "gold_rank": top_ids.index(gold_id) + 1 if gold_id in top_ids else "",
            "top10_ids": json.dumps(top_ids, ensure_ascii=False),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "index_preflight_events.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "status": "DIAGNOSTIC_NOT_CITATION_READY",
        "event_count": len(results),
        "fresh_hit_at_10": sum(int(row["fresh_hit_at_10"]) for row in results) / len(results),
        "baseline_build": summarize(results, "baseline_build_ms"),
        "sparse_update": summarize(results, "sparse_update_ms"),
        "dense_update": summarize(results, "dense_update_ms"),
        "fusion_search": summarize(results, "fusion_search_ms"),
        "index_plus_search": summarize(results, "index_plus_search_ms"),
        "limitations": [
            "No database/backend time is included.",
            "Events are isolated and sequential.",
            "Results size the formal runner; they are not paper evidence.",
        ],
    }
    (args.output / "index_preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
