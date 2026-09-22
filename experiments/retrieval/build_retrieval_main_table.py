#!/usr/bin/env python3
"""Build publication-facing retrieval metrics from frozen LoCoMo rankings.

The historical project files label first-relevant hit rates as R@K.  This
script reports those values as Hit@K and additionally computes true Recall@10
and binary-relevance nDCG@10 for queries with one or more annotated gold
memories.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "results" / "retrieval" / "retrieval_main_table"
QUERY_PATH = (
    ROOT
    / "data"
    / "retrieval_gold"
    / "locomo_cat1_4_gold_memory.csv"
)
MEMORY_PATH = ROOT / "data" / "locomo_memory_records.csv"

METHODS = [
    {
        "group": "Textual Retrieval Baselines",
        "method": "BM25",
        "display_method": "BM25",
        "scope": "sample-scoped",
        "comparable": True,
        "ranking": ROOT
        / "results"
        / "retrieval"
        / "official_eval"
        / "bm25_raw_ranking_canonical1540.csv",
    },
    {
        "group": "Textual Retrieval Baselines",
        "method": "Dense-bge",
        "display_method": "Dense-bge",
        "scope": "sample-scoped",
        "comparable": True,
        "ranking": ROOT
        / "results"
        / "retrieval"
        / "official_eval"
        / "dense_bge_ranking_canonical1540.csv",
    },
    {
        "group": "KG / Structured Retrieval Baselines",
        "method": "Dense+GlobalKG",
        "display_method": "Dense+GlobalKG",
        "scope": "sample-scoped",
        "comparable": True,
        "ranking": ROOT
        / "results"
        / "retrieval"
        / "dense_global_kg_rerun"
        / "dense_global_kg_top10.csv",
    },
    {
        "group": "KG / Structured Retrieval Baselines",
        "method": "RRF_compact",
        "display_method": "RRF_compact",
        "scope": "sample-scoped",
        "comparable": True,
        "ranking": ROOT
        / "results"
        / "retrieval"
        / "official_eval"
        / "rrf_compact_canonical1540"
        / "rrf_compact_top10.csv",
    },
    {
        "group": "Ablation",
        "method": "ZScore-Raw",
        "display_method": "ZScore-Raw",
        "scope": "sample-scoped",
        "comparable": True,
        "ranking": ROOT
        / "results"
        / "retrieval"
        / "official_eval"
        / "zscore_raw_ranking_canonical1540.csv",
    },
    {
        "group": "Ours",
        "method": "ZScore-RawERK",
        "display_method": "CassMem (Ours)",
        "scope": "sample-scoped",
        "comparable": True,
        "ranking": ROOT
        / "results"
        / "retrieval"
        / "official_eval"
        / "zscore_rawerk_ranking_canonical1540.csv",
    },
]

CATEGORY_LABELS = {
    "1": "Multi-Hop",
    "2": "Temporal",
    "3": "Open-Domain",
    "4": "Single-Hop",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gold_ids(value: str) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in value.split(";") if item.strip()))


def read_queries() -> tuple[dict[str, dict], list[dict]]:
    queries: dict[str, dict] = {}
    excluded: list[dict] = []
    with QUERY_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            gold_ids = parse_gold_ids(row.get("gold_memory_ids", ""))
            normalized = {
                "query_id": row["query_id"].strip(),
                "category": row["category"].strip(),
                "category_label": CATEGORY_LABELS[row["category"].strip()],
                "conversation_id": row["conversation_id"].strip(),
                "split": row["split"].strip(),
                "question": row["question"].strip(),
                "gold_ids": gold_ids,
            }
            queries[normalized["query_id"]] = normalized
            if not gold_ids:
                excluded.append(
                    {
                        "query_id": normalized["query_id"],
                        "category": normalized["category"],
                        "category_label": normalized["category_label"],
                        "split": normalized["split"],
                        "reason": "missing gold_memory_ids",
                    }
                )
    return queries, excluded


def read_top10(path: Path, ranking_format: str = "rank_rows") -> dict[str, list[str]]:
    if ranking_format == "top10_list":
        result = {}
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                result[row["qa_id"].strip()] = [
                    item.strip()
                    for item in row["dense_kg_top10_memory_ids"].split(";")
                    if item.strip()
                ][:10]
        return result

    ranked: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rank = int(float(row["rank"]))
            if rank <= 10:
                ranked[row["query_id"].strip()].append(
                    (rank, row["memory_id"].strip())
                )
    return {
        query_id: [
            memory_id
            for _, memory_id in sorted(items, key=lambda item: item[0])
        ]
        for query_id, items in ranked.items()
    }


def per_query_metrics(gold_ids: list[str], ranking: list[str]) -> dict[str, float]:
    gold = set(gold_ids)
    ranked = ranking[:10]
    relevant_ranks = [
        index for index, memory_id in enumerate(ranked, start=1) if memory_id in gold
    ]
    first_rank = min(relevant_ranks) if relevant_ranks else None

    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_count = min(len(gold), 10)
    idcg = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1)
    )

    return {
        "MRR@10": 0.0 if first_rank is None else 1.0 / first_rank,
        "Hit@1": float(first_rank is not None and first_rank <= 1),
        "Hit@5": float(first_rank is not None and first_rank <= 5),
        "Hit@10": float(first_rank is not None and first_rank <= 10),
        "Recall@10": len(set(ranked) & gold) / len(gold),
        "nDCG@10": 0.0 if idcg == 0 else dcg / idcg,
        "gold_count": float(len(gold)),
        "retrieved_gold_count": float(len(set(ranked) & gold)),
    }


def paired_bootstrap(
    ours: list[float],
    baseline: list[float],
    *,
    seed: int = 20260731,
    resamples: int = 10000,
) -> dict[str, float]:
    deltas = np.asarray(ours, dtype=float) - np.asarray(baseline, dtype=float)
    if not len(deltas):
        raise ValueError("paired bootstrap received no observations")
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    batch_size = 500
    for start in range(0, resamples, batch_size):
        batch = min(batch_size, resamples - start)
        indices = rng.integers(0, len(deltas), size=(batch, len(deltas)))
        samples.append(deltas[indices].mean(axis=1))
    boot = np.concatenate(samples)
    lower_tail = (int(np.count_nonzero(boot <= 0)) + 1) / (resamples + 1)
    upper_tail = (int(np.count_nonzero(boot >= 0)) + 1) / (resamples + 1)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return {
        "delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_p_two_sided": p_value,
        "resamples": resamples,
        "seed": seed,
    }


def mean(rows: list[dict], metric: str) -> float:
    return sum(float(row[metric]) for row in rows) / len(rows)


def summarize(rows: list[dict]) -> dict[str, float | int]:
    metrics = ["MRR@10", "Hit@1", "Hit@5", "Hit@10", "Recall@10", "nDCG@10"]
    return {"n": len(rows), **{metric: mean(rows, metric) for metric in metrics}}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_candidate_pool_audit(evaluable: dict[str, dict]) -> list[dict]:
    memories_by_conversation: dict[str, set[str]] = defaultdict(set)
    with MEMORY_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            memories_by_conversation[row["sample_id"].strip()].add(
                row["memory_id"].strip()
            )

    per_query = []
    for query in evaluable.values():
        candidates = memories_by_conversation[query["conversation_id"]]
        gold = set(query["gold_ids"])
        per_query.append(
            {
                "category_label": query["category_label"],
                "candidate_count": len(candidates),
                "oracle_candidate_recall": len(gold & candidates) / len(gold),
                "all_gold_in_candidate_pool": gold <= candidates,
            }
        )

    audit = []
    for label in ("Overall", "Single-Hop", "Multi-Hop", "Temporal", "Open-Domain"):
        rows = (
            per_query
            if label == "Overall"
            else [row for row in per_query if row["category_label"] == label]
        )
        audit.append(
            {
                "scope": label,
                "n": len(rows),
                "oracle_candidate_recall": sum(
                    row["oracle_candidate_recall"] for row in rows
                )
                / len(rows),
                "queries_with_all_gold": sum(
                    row["all_gold_in_candidate_pool"] for row in rows
                ),
                "avg_candidate_count": sum(row["candidate_count"] for row in rows)
                / len(rows),
                "min_candidate_count": min(row["candidate_count"] for row in rows),
                "max_candidate_count": max(row["candidate_count"] for row in rows),
                "interpretation": (
                    "Sanity check for the full conversation-scoped pool; not a "
                    "pruned KG-expansion candidate metric."
                ),
            }
        )
    return audit


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    queries, excluded = read_queries()
    evaluable = {
        query_id: row for query_id, row in queries.items() if row["gold_ids"]
    }
    candidate_audit = build_candidate_pool_audit(evaluable)

    per_query_rows: list[dict] = []
    overall_rows: list[dict] = []
    category_rows: list[dict] = []
    input_files = [{"path": str(QUERY_PATH), "sha256": sha256(QUERY_PATH)}]

    for method in METHODS:
        ranking_path = method["ranking"]
        rankings = read_top10(
            ranking_path,
            method.get("ranking_format", "rank_rows"),
        )
        input_files.append({"path": str(ranking_path), "sha256": sha256(ranking_path)})
        missing_rankings = sorted(set(evaluable) - set(rankings))
        if missing_rankings:
            raise RuntimeError(
                f"{method['method']} missing {len(missing_rankings)} evaluable rankings"
            )

        method_rows: list[dict] = []
        for query_id, query in evaluable.items():
            metrics = per_query_metrics(query["gold_ids"], rankings[query_id])
            row = {
                "group": method["group"],
                "method": method["method"],
                "display_method": method["display_method"],
                "scope": method["scope"],
                "comparable": method["comparable"],
                "query_id": query_id,
                "category": query["category"],
                "category_label": query["category_label"],
                "split": query["split"],
                **metrics,
            }
            method_rows.append(row)
            per_query_rows.append(row)

        for evaluation_split in ("test", "all"):
            split_rows = (
                method_rows
                if evaluation_split == "all"
                else [
                    row for row in method_rows if row["split"] == evaluation_split
                ]
            )
            overall_rows.append(
                {
                    "group": method["group"],
                    "method": method["method"],
                    "display_method": method["display_method"],
                    "scope": method["scope"],
                    "comparable": method["comparable"],
                    "evaluation_split": evaluation_split,
                    **summarize(split_rows),
                    "Candidate Recall": "",
                    "candidate_recall_status": (
                        "N/A: method ranks the full sample-scoped memory pool; no "
                        "distinct pruned candidate-generation stage"
                    ),
                }
            )

            for category in ("4", "1", "2", "3"):
                category_method_rows = [
                    row for row in split_rows if row["category"] == category
                ]
                category_rows.append(
                    {
                        "group": method["group"],
                        "method": method["method"],
                        "display_method": method["display_method"],
                        "scope": method["scope"],
                        "comparable": method["comparable"],
                        "evaluation_split": evaluation_split,
                        "category": category,
                        "category_label": CATEGORY_LABELS[category],
                        **summarize(category_method_rows),
                    }
                )

    # Reconcile the historical values after correcting the denominator and labels.
    expected = {
        "BM25": (0.3508, 0.2578, 0.4674, 0.5501),
        "Dense-bge": (0.4872, 0.3750, 0.6439, 0.7318),
        "ZScore-Raw": (0.5049, 0.3776, 0.6803, 0.7689),
        "RRF_compact": (0.5196, 0.3978, 0.6862, 0.7845),
        "ZScore-RawERK": (0.5455, 0.4238, 0.7109, 0.7962),
    }
    reconciliation = []
    overall_map = {
        row["method"]: row
        for row in overall_rows
        if row["evaluation_split"] == "all"
    }
    for method, anchors in expected.items():
        actual = overall_map[method]
        values = (
            actual["MRR@10"],
            actual["Hit@1"],
            actual["Hit@5"],
            actual["Hit@10"],
        )
        passed = all(round(a, 4) == e for a, e in zip(values, anchors))
        reconciliation.append(
            {
                "method": method,
                "expected_MRR": anchors[0],
                "actual_MRR@10": values[0],
                "expected_R@1_historical": anchors[1],
                "actual_Hit@1": values[1],
                "expected_R@5_historical": anchors[2],
                "actual_Hit@5": values[2],
                "expected_R@10_historical": anchors[3],
                "actual_Hit@10": values[3],
                "status": (
                    "UNCHANGED"
                    if passed
                    else "CHANGED_AFTER_CANONICAL_GOLD_V2"
                ),
            }
        )

    test_by_method = defaultdict(dict)
    for row in per_query_rows:
        if row["split"] == "test":
            test_by_method[row["method"]][row["query_id"]] = row
    significance_rows = []
    significance_metrics = (
        "MRR@10",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Recall@10",
        "nDCG@10",
    )
    for baseline_method in ("Dense+GlobalKG", "RRF_compact", "ZScore-Raw"):
        shared_ids = sorted(
            set(test_by_method["ZScore-RawERK"])
            & set(test_by_method[baseline_method])
        )
        for metric_index, metric in enumerate(significance_metrics):
            stats = paired_bootstrap(
                [
                    test_by_method["ZScore-RawERK"][query_id][metric]
                    for query_id in shared_ids
                ],
                [
                    test_by_method[baseline_method][query_id][metric]
                    for query_id in shared_ids
                ],
                seed=20260731 + metric_index,
            )
            significance_rows.append(
                {
                    "method": "ZScore-RawERK",
                    "baseline": baseline_method,
                    "split": "test",
                    "metric": metric,
                    "n": len(shared_ids),
                    **stats,
                }
            )

    # Holm correction controls family-wise error across all reported comparisons.
    order = sorted(
        range(len(significance_rows)),
        key=lambda index: significance_rows[index]["bootstrap_p_two_sided"],
    )
    running = 0.0
    total_tests = len(order)
    for rank, index in enumerate(order):
        raw_p = significance_rows[index]["bootstrap_p_two_sided"]
        adjusted = min(1.0, (total_tests - rank) * raw_p)
        running = max(running, adjusted)
        significance_rows[index]["holm_adjusted_p"] = running

    overall_fields = [
        "group",
        "method",
        "display_method",
        "scope",
        "comparable",
        "evaluation_split",
        "n",
        "MRR@10",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Recall@10",
        "nDCG@10",
        "Candidate Recall",
        "candidate_recall_status",
    ]
    category_fields = [
        "group",
        "method",
        "display_method",
        "scope",
        "comparable",
        "evaluation_split",
        "category",
        "category_label",
        "n",
        "MRR@10",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Recall@10",
        "nDCG@10",
    ]
    per_query_fields = [
        "group",
        "method",
        "display_method",
        "scope",
        "comparable",
        "query_id",
        "category",
        "category_label",
        "split",
        "MRR@10",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Recall@10",
        "nDCG@10",
        "gold_count",
        "retrieved_gold_count",
    ]

    write_csv(REPORT_DIR / "retrieval_main_overall.csv", overall_rows, overall_fields)
    write_csv(
        REPORT_DIR / "retrieval_main_by_category.csv",
        category_rows,
        category_fields,
    )
    write_csv(
        REPORT_DIR / "retrieval_main_per_query.csv",
        per_query_rows,
        per_query_fields,
    )
    write_csv(
        REPORT_DIR / "retrieval_exclusion_audit.csv",
        excluded,
        ["query_id", "category", "category_label", "split", "reason"],
    )
    write_csv(
        REPORT_DIR / "historical_metric_reconciliation.csv",
        reconciliation,
        list(reconciliation[0]),
    )
    write_csv(
        REPORT_DIR / "candidate_pool_oracle_audit.csv",
        candidate_audit,
        list(candidate_audit[0]),
    )
    write_csv(
        REPORT_DIR / "retrieval_main_significance.csv",
        significance_rows,
        [
            "method",
            "baseline",
            "split",
            "metric",
            "n",
            "delta",
            "ci95_low",
            "ci95_high",
            "bootstrap_p_two_sided",
            "holm_adjusted_p",
            "resamples",
            "seed",
        ],
    )
    (REPORT_DIR / "retrieval_main_data.json").write_text(
        json.dumps(
            {
                "overall": overall_rows,
                "category": category_rows,
                "exclusions": excluded,
                "reconciliation": reconciliation,
                "candidate_audit": candidate_audit,
                "significance": significance_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    gold_distribution = Counter(
        len(query["gold_ids"]) for query in evaluable.values()
    )
    manifest = {
        "title": "CassMem Retrieval Main Table",
        "protocol_version": "retrieval-main-v1",
        "query_universe": {
            "cat1_4_total": len(queries),
            "evaluable": len(evaluable),
            "excluded_missing_gold": len(excluded),
            "category_order": ["Single-Hop", "Multi-Hop", "Temporal", "Open-Domain"],
            "evaluable_by_category": dict(
                Counter(query["category_label"] for query in evaluable.values())
            ),
            "gold_memories_per_query": {
                str(key): value for key, value in sorted(gold_distribution.items())
            },
        },
        "metric_definitions": {
            "MRR@10": "Mean reciprocal rank of the first relevant memory within Top-10.",
            "Hit@K": "Fraction of queries with at least one relevant memory in Top-K.",
            "Recall@10": (
                "Macro-average of |Top10 intersect gold| / |gold|; this is distinct "
                "from the project's historical R@10 label."
            ),
            "nDCG@10": (
                "Binary-relevance normalized discounted cumulative gain at rank 10."
            ),
            "Candidate Recall": (
                "Not reported: frozen methods do not expose a distinct pruned "
                "candidate-generation artifact before ranking."
            ),
        },
        "comparability": {
            "primary_rows": "sample-scoped methods only",
            "primary_split": (
                "Held-out test split (n=1146 evidence-bearing queries). The all-query "
                "aggregate is diagnostic because method weights were selected on dev."
            ),
            "cat5": (
                "Excluded from retrieval effectiveness because adversarial Cat5 has "
                "no gold-memory retrieval target in this protocol."
            ),
            "reader_global_kg_warning": (
                "The existing GPT-4o reader Dense+GlobalKG row used the old ranking "
                "path and must be regenerated before it can represent this rerun."
            ),
        },
        "input_files": input_files,
        "outputs": [
            "retrieval_main_overall.csv",
            "retrieval_main_by_category.csv",
            "retrieval_main_per_query.csv",
            "retrieval_exclusion_audit.csv",
            "historical_metric_reconciliation.csv",
            "candidate_pool_oracle_audit.csv",
            "retrieval_main_significance.csv",
            "retrieval_main_data.json",
        ],
    }
    (REPORT_DIR / "retrieval_main_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest["query_universe"], ensure_ascii=False, indent=2))
    print(f"Wrote retrieval main-table artifacts to {REPORT_DIR}")


if __name__ == "__main__":
    main()
