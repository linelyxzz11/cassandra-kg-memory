"""Build publication-aligned fusion ablation from frozen CassMem branches.

All methods consume the same frozen Dense branch and RawERK BM25 branch.  The
fusion parameters were selected on the historical development split and are
kept fixed here.  Evaluation uses retrieval_gold_v2 and the 1,146-query test
scope used by Retrieval Table 1.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
P1_SCRIPT = ROOT / "04_experiments" / "retrieval" / "p1_compact_component_ablation" / "run_p1c_ablation_v5.py"
P1_CONFIG = ROOT / "04_experiments" / "retrieval" / "p1_compact_component_ablation" / "p1c_config.json"
GOLD_PATH = ROOT / "02_artifacts" / "retrieval_gold_v2" / "locomo_cat1_4_gold_memory.csv"
TABLE1_PATH = ROOT / "05_reports" / "retrieval_main_table" / "retrieval_main_overall.csv"
OUTPUT = ROOT / "05_reports" / "fusion_ablation_aligned_v2"
OVERALL_PATH = OUTPUT / "fusion_ablation_overall_heldout1146.csv"
PER_QUERY_PATH = OUTPUT / "fusion_ablation_per_query_heldout1146.csv"
MANIFEST_PATH = OUTPUT / "manifest.json"
TABLE_PATH = OUTPUT / "TABLE3_FUSION_ABLATION_ALIGNED.md"

METRICS = ("MRR@10", "Hit@1", "Hit@5", "Hit@10", "Recall@10", "nDCG@10")


def load_p1_module():
    spec = importlib.util.spec_from_file_location("p1c_ablation_v5", P1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {P1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_order(dense, bm25):
    ordered = []
    seen = set()
    for memory_id, _ in [*dense, *bm25]:
        if memory_id not in seen:
            seen.add(memory_id)
            ordered.append(memory_id)
    return ordered


def rank_scores(scores: dict[str, float], order: list[str], top_n: int = 50):
    ordinal = {memory_id: index for index, memory_id in enumerate(order)}
    return sorted(scores.items(), key=lambda item: (-item[1], ordinal[item[0]]))[:top_n]


def raw_linear(dense, bm25, alpha: float = 0.9):
    order = candidate_order(dense, bm25)
    dense_map = dict(dense)
    bm25_map = dict(bm25)
    dense_missing = min(dense_map.values())
    bm25_missing = min(bm25_map.values())
    scores = {
        memory_id: alpha * dense_map.get(memory_id, dense_missing)
        + (1.0 - alpha) * bm25_map.get(memory_id, bm25_missing)
        for memory_id in order
    }
    return rank_scores(scores, order)


def rrf(dense, bm25, alpha: float, k: int = 10):
    order = candidate_order(dense, bm25)
    dense_rank = {memory_id: rank for rank, (memory_id, _) in enumerate(dense, 1)}
    bm25_rank = {memory_id: rank for rank, (memory_id, _) in enumerate(bm25, 1)}
    scores = {
        memory_id: alpha / (k + dense_rank.get(memory_id, 999))
        + (1.0 - alpha) / (k + bm25_rank.get(memory_id, 999))
        for memory_id in order
    }
    return rank_scores(scores, order)


def minmax(dense, bm25, alpha: float = 0.6):
    order = candidate_order(dense, bm25)
    dense_map = dict(dense)
    bm25_map = dict(bm25)
    dense_min, dense_max = min(dense_map.values()), max(dense_map.values())
    bm25_min, bm25_max = min(bm25_map.values()), max(bm25_map.values())
    dense_range = max(dense_max - dense_min, 1e-12)
    bm25_range = max(bm25_max - bm25_min, 1e-12)
    scores = {
        memory_id: alpha
        * (dense_map.get(memory_id, dense_min) - dense_min)
        / dense_range
        + (1.0 - alpha)
        * (bm25_map.get(memory_id, bm25_min) - bm25_min)
        / bm25_range
        for memory_id in order
    }
    return rank_scores(scores, order)


def score(gold_ids: list[str], ranking: list[str]):
    gold = set(gold_ids)
    ranked = ranking[:10]
    relevant_ranks = [rank for rank, memory_id in enumerate(ranked, 1) if memory_id in gold]
    first_rank = min(relevant_ranks) if relevant_ranks else None
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), 10) + 1))
    return {
        "MRR@10": 0.0 if first_rank is None else 1.0 / first_rank,
        "Hit@1": float(first_rank == 1),
        "Hit@5": float(first_rank is not None and first_rank <= 5),
        "Hit@10": float(first_rank is not None and first_rank <= 10),
        "Recall@10": len(set(ranked) & gold) / len(gold),
        "nDCG@10": 0.0 if not idcg else dcg / idcg,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p1 = load_p1_module()
    config = p1.load_json(P1_CONFIG)
    project_root = p1.resolve_path(config.get("project_root", "."), P1_CONFIG.parent)

    gold = {}
    with GOLD_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ids = [item for item in row["gold_memory_ids"].split(";") if item]
            if row["split"] == "test" and row["gold_status"] == "mapped" and ids:
                gold[row["query_id"]] = ids
    if len(gold) != 1146:
        raise RuntimeError(f"Expected 1,146 gold rows, got {len(gold)}")

    all_queries = p1.load_queries(p1.resolve_path(config["paths"]["queries"], project_root))
    queries = [query for query in all_queries if query.query_id in gold]
    if len(queries) != 1146:
        raise RuntimeError(f"Expected 1,146 query records, got {len(queries)}")
    inputs = p1.load_inputs(config, project_root, set(gold))
    documents = {
        memory_id: p1.build_document(memory_id, ("E", "R", "K"), inputs, config["renderer"])
        for memory_id in inputs.memory_records
    }
    bm25_by_query = p1.build_bm25_rankings_for_variant("RawERK", queries, inputs, config, documents)
    top_n = int(config["fusion"]["candidate_depth"])

    methods: list[tuple[str, str, Callable]] = [
        ("RawScore", "0.9*Dense + 0.1*BM25 (unnormalized)", raw_linear),
        ("Equal-RRF", "0.5 RRF(Dense) + 0.5 RRF(BM25), k=10", lambda d, b: rrf(d, b, 0.5)),
        ("Weighted-RRF", "0.6 RRF(Dense) + 0.4 RRF(BM25), k=10", lambda d, b: rrf(d, b, 0.6)),
        ("MinMax", "0.6 minmax(Dense) + 0.4 minmax(BM25)", minmax),
        (
            "ZScore",
            "0.6 z(Dense) + 0.4 z(BM25)",
            lambda d, b: p1.zscore_fuse(d, b, 0.6, "branch_min", top_n),
        ),
    ]

    per_query = []
    overall = []
    for method, formula, fuse in methods:
        method_rows = []
        for query in queries:
            dense = p1.top_dense_for_query(query, inputs, top_n)
            bm25 = bm25_by_query[query.query_id][:top_n]
            ranking = [memory_id for memory_id, _ in fuse(dense, bm25)[:10]]
            row = {
                "method": method,
                "query_id": query.query_id,
                "category": query.category,
                "top10_memory_ids": ";".join(ranking),
                **score(gold[query.query_id], ranking),
            }
            method_rows.append(row)
            per_query.append(row)
        overall.append(
            {
                "method": method,
                "formula": formula,
                "n": len(method_rows),
                **{
                    metric: sum(float(row[metric]) for row in method_rows) / len(method_rows)
                    for metric in METRICS
                },
            }
        )

    with TABLE1_PATH.open(encoding="utf-8-sig", newline="") as handle:
        table1 = next(
            row
            for row in csv.DictReader(handle)
            if row["method"] == "ZScore-RawERK" and row["evaluation_split"] == "test"
        )
    zscore = next(row for row in overall if row["method"] == "ZScore")
    for metric in METRICS:
        if abs(float(zscore[metric]) - float(table1[metric])) > 1e-12:
            raise RuntimeError(f"Table 1 parity failed for {metric}: {zscore[metric]} != {table1[metric]}")

    write_csv(OVERALL_PATH, overall, ["method", "formula", "n", *METRICS])
    write_csv(
        PER_QUERY_PATH,
        per_query,
        ["method", "query_id", "category", "top10_memory_ids", *METRICS],
    )

    table_lines = [
        "# Table 3: Fusion Ablation",
        "",
        "Held-out test, LoCoMo Cat1-4, n=1,146. Dense and RawERK BM25 branches are fixed.",
        "",
        "| Fusion | Formula | MRR@10 | Hit@10 | Recall@10 | nDCG@10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in overall:
        bold = row["method"] == "ZScore"
        values = [
            str(row["method"]),
            str(row["formula"]),
            f"{float(row['MRR@10']):.4f}",
            f"{float(row['Hit@10']):.4f}",
            f"{float(row['Recall@10']):.4f}",
            f"{float(row['nDCG@10']):.4f}",
        ]
        if bold:
            values = [f"**{value}**" for value in values]
        table_lines.append("| " + " | ".join(values) + " |")
    table_lines.extend(
        [
            "",
            "The historical `R@10` field was binary Hit@10. True multi-gold Recall@10 is reported separately.",
            "ZScore exactly matches the CassMem row in Retrieval Table 1 on all six metrics.",
        ]
    )
    TABLE_PATH.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    manifest = {
        "experiment": "Fusion ablation aligned to Retrieval Table 1",
        "scope": "held-out test; Cat1-4; n=1,146 mapped-evidence queries",
        "fixed_inputs": "same frozen Dense and RawERK BM25 top-50 branches for every method",
        "parameter_policy": "fusion parameters frozen from the development sweep; no test tuning",
        "table1_zscore_parity": "PASS on all six retrieval metrics",
        "inputs": {
            str(P1_CONFIG.relative_to(ROOT)): sha256(P1_CONFIG),
            str(GOLD_PATH.relative_to(ROOT)): sha256(GOLD_PATH),
            str(TABLE1_PATH.relative_to(ROOT)): sha256(TABLE1_PATH),
        },
        "outputs": {
            str(OVERALL_PATH.relative_to(ROOT)): sha256(OVERALL_PATH),
            str(PER_QUERY_PATH.relative_to(ROOT)): sha256(PER_QUERY_PATH),
            str(TABLE_PATH.relative_to(ROOT)): sha256(TABLE_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(OVERALL_PATH)


if __name__ == "__main__":
    main()
