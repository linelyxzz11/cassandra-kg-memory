"""Reaggregate the frozen P1-C rankings on the publication retrieval protocol.

The original P1-C report used the historical 1,150-query held-out scope.  The
publication retrieval table excludes four questions whose evidence cannot be
mapped to a canonical memory, leaving 1,146 evaluable test questions.  This
script does not rerun retrieval; it only scores the frozen rankings against
retrieval_gold_v2 with the same metric definitions as Table 1.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results" / "representation_ablation" / "p1_compact_component_ablation"
GOLD_PATH = ROOT / "data" / "retrieval_gold" / "locomo_cat1_4_gold_memory.csv"
RANKING_PATH = REPORT / "p1c_zscore_rankings_heldout1150.csv"
OVERALL_PATH = REPORT / "p1c_zscore_overall_heldout1146_aligned.csv"
CATEGORY_PATH = REPORT / "p1c_zscore_by_category_heldout1146_aligned.csv"
PER_QUERY_PATH = REPORT / "p1c_zscore_per_query_heldout1146_aligned.csv"
MANIFEST_PATH = REPORT / "p1c_heldout1146_alignment_manifest.json"

METRICS = ("MRR@10", "Hit@1", "Hit@5", "Hit@10", "Recall@10", "nDCG@10")
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


def score(gold_ids: list[str], ranking: list[str]) -> dict[str, float]:
    gold = set(gold_ids)
    ranked = ranking[:10]
    relevant_ranks = [
        rank for rank, memory_id in enumerate(ranked, start=1) if memory_id in gold
    ]
    first_rank = min(relevant_ranks) if relevant_ranks else None
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(gold), 10) + 1)
    )
    return {
        "MRR@10": 0.0 if first_rank is None else 1.0 / first_rank,
        "Hit@1": float(first_rank == 1),
        "Hit@5": float(first_rank is not None and first_rank <= 5),
        "Hit@10": float(first_rank is not None and first_rank <= 10),
        "Recall@10": len(set(ranked) & gold) / len(gold),
        "nDCG@10": 0.0 if not idcg else dcg / idcg,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        **{
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in METRICS
        },
    }


def main() -> None:
    gold: dict[str, dict] = {}
    with GOLD_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ids = [item for item in row["gold_memory_ids"].split(";") if item]
            if row["split"] == "test" and row["gold_status"] == "mapped" and ids:
                gold[row["query_id"]] = {
                    "category": row["category"],
                    "gold_ids": ids,
                }
    if len(gold) != 1146:
        raise RuntimeError(f"Expected 1,146 evaluable test queries, got {len(gold)}")

    rankings: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with RANKING_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            query_id = row["query_id"]
            rank = int(row["rank"])
            if query_id in gold and rank <= 10:
                rankings[row["variant"]][query_id].append((rank, row["memory_id"]))

    per_query: list[dict] = []
    for variant in sorted(rankings, key=lambda name: (len(name), name)):
        missing = set(gold) - set(rankings[variant])
        if missing:
            raise RuntimeError(f"{variant} is missing {len(missing)} evaluable queries")
        for query_id, gold_row in gold.items():
            ranking = [
                memory_id
                for _, memory_id in sorted(rankings[variant][query_id])
            ]
            per_query.append(
                {
                    "variant": variant,
                    "query_id": query_id,
                    "category": gold_row["category"],
                    "category_label": CATEGORY_LABELS[gold_row["category"]],
                    "gold_count": len(gold_row["gold_ids"]),
                    **score(gold_row["gold_ids"], ranking),
                }
            )

    variants = ["Raw", "RawE", "RawR", "RawK", "RawER", "RawEK", "RawRK", "RawERK", "RawERKT"]
    overall = []
    by_category = []
    for variant in variants:
        selected = [row for row in per_query if row["variant"] == variant]
        overall.append({"variant": variant, **summarize(selected)})
        for category in ("4", "1", "2", "3"):
            category_rows = [row for row in selected if row["category"] == category]
            by_category.append(
                {
                    "variant": variant,
                    "category": category,
                    "category_label": CATEGORY_LABELS[category],
                    **summarize(category_rows),
                }
            )

    write_csv(OVERALL_PATH, overall, ["variant", "n", *METRICS])
    write_csv(
        CATEGORY_PATH,
        by_category,
        ["variant", "category", "category_label", "n", *METRICS],
    )
    write_csv(
        PER_QUERY_PATH,
        per_query,
        ["variant", "query_id", "category", "category_label", "gold_count", *METRICS],
    )
    manifest = {
        "experiment": "P1-C representation ablation aligned to Retrieval Table 1",
        "operation": "metric-only reaggregation; frozen rankings were not rerun",
        "scope": "held-out test; Cat1-4; 1,146 mapped-evidence queries",
        "excluded": "four historical heldout1150 questions with unmapped gold memories",
        "metric_definitions": {
            "MRR@10": "reciprocal rank of first relevant memory within Top-10",
            "Hit@K": "at least one relevant memory in Top-K",
            "Recall@10": "fraction of all gold memories retrieved in Top-10",
            "nDCG@10": "binary-relevance nDCG at 10",
        },
        "inputs": {
            str(GOLD_PATH.relative_to(ROOT)): sha256(GOLD_PATH),
            str(RANKING_PATH.relative_to(ROOT)): sha256(RANKING_PATH),
        },
        "outputs": {
            str(OVERALL_PATH.relative_to(ROOT)): sha256(OVERALL_PATH),
            str(CATEGORY_PATH.relative_to(ROOT)): sha256(CATEGORY_PATH),
            str(PER_QUERY_PATH.relative_to(ROOT)): sha256(PER_QUERY_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(OVERALL_PATH)


if __name__ == "__main__":
    main()
