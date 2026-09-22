#!/usr/bin/env python3
"""Re-run Dense+GlobalKG under a publication-safe LoCoMo protocol.

The legacy implementation added a fixed binary "has any KG edge" bonus with
weight 0.1.  This script keeps the retrieval universe sample-scoped, selects
the KG weight on the frozen dev split only, and reports the held-out test split
separately.  It evaluates both the legacy binary prior and a query-independent
degree-centrality prior so that any gain can be attributed and audited.
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
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "results" / "retrieval" / "dense_global_kg_rerun"

MEM_EMB_PATH = DATA_DIR / "locomo_memory_bge_large.npy"
QA_EMB_PATH = DATA_DIR / "locomo_qa_bge_large.npy"
MEM_ID_PATH = DATA_DIR / "locomo_memory_ids_bge.txt"
QA_ID_PATH = DATA_DIR / "locomo_qa_ids_bge.txt"
MEMORY_PATH = DATA_DIR / "locomo_memory_records.csv"
QUERY_PATH = (
    ROOT
    / "data"
    / "retrieval_gold"
    / "locomo_cat1_4_gold_memory.csv"
)
KG_EDGE_PATH = ROOT / "data" / "frozen_retrieval" / "locomo_kg_edges_spacy.csv"

LAMBDAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
PRIOR_TYPES = ("binary_coverage", "degree_centrality")
METRICS = ("MRR@10", "Hit@1", "Hit@5", "Hit@10", "Recall@10", "nDCG@10")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ids(value: str) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in value.split(";") if item.strip()))


def normalize_node(value: str) -> str:
    return " ".join(value.lower().split())


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(values.mean())) / std


def per_query_metrics(gold_ids: list[str], ranking: list[str]) -> dict[str, float]:
    gold = set(gold_ids)
    ranked = ranking[:10]
    relevant_ranks = [
        rank for rank, memory_id in enumerate(ranked, start=1) if memory_id in gold
    ]
    first = min(relevant_ranks) if relevant_ranks else None
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(gold), 10) + 1)
    )
    return {
        "MRR@10": 0.0 if first is None else 1.0 / first,
        "Hit@1": float(first is not None and first <= 1),
        "Hit@5": float(first is not None and first <= 5),
        "Hit@10": float(first is not None and first <= 10),
        "Recall@10": len(set(ranked) & gold) / len(gold),
        "nDCG@10": 0.0 if idcg == 0 else dcg / idcg,
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        **{
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in METRICS
        },
    }


def load_inputs():
    memory_rows = read_csv(MEMORY_PATH)
    query_rows = read_csv(QUERY_PATH)
    edge_rows = read_csv(KG_EDGE_PATH)

    mem_ids = [
        line.strip()
        for line in MEM_ID_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qa_ids = [
        line.strip()
        for line in QA_ID_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mem_embs = np.load(MEM_EMB_PATH).astype(np.float64)
    qa_embs = np.load(QA_EMB_PATH).astype(np.float64)
    mem_embs /= np.maximum(np.linalg.norm(mem_embs, axis=1, keepdims=True), 1e-12)
    qa_embs /= np.maximum(np.linalg.norm(qa_embs, axis=1, keepdims=True), 1e-12)

    if len(mem_ids) != len(mem_embs) or len(qa_ids) != len(qa_embs):
        raise RuntimeError("Embedding arrays and ID files have inconsistent lengths")

    return memory_rows, query_rows, edge_rows, mem_ids, qa_ids, mem_embs, qa_embs


def build_memory_priors(
    memory_rows: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict]:
    memory_sample = {}
    memory_by_turn: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in memory_rows:
        memory_id = row["memory_id"].strip()
        sample_id = row["sample_id"].strip()
        dia_id = row["dia_id"].strip()
        memory_sample[memory_id] = sample_id
        memory_by_turn[(sample_id, dia_id)].append(memory_id)

    mapped_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    sample_degree: dict[str, Counter[str]] = defaultdict(Counter)
    mapped_edge_count = 0
    for edge in edge_rows:
        sample_id = edge["graph_id"].strip()
        turn_id = edge["evidence"].strip()
        src = normalize_node(edge["src_id"])
        dst = normalize_node(edge["dst_id"])
        if not src or not dst:
            continue
        mapped_memories = memory_by_turn.get((sample_id, turn_id), [])
        if not mapped_memories:
            continue
        sample_degree[sample_id][src] += 1
        sample_degree[sample_id][dst] += 1
        for memory_id in mapped_memories:
            mapped_edges[memory_id].append((src, dst))
            mapped_edge_count += 1

    priors: dict[str, dict[str, float]] = {}
    for memory_id, sample_id in memory_sample.items():
        edges = mapped_edges.get(memory_id, [])
        binary = float(bool(edges))
        if not edges:
            centrality = 0.0
        else:
            degrees = sample_degree[sample_id]
            endpoint_salience = [
                math.log1p(degrees[src] + degrees[dst]) for src, dst in edges
            ]
            centrality = math.log1p(len(edges)) + sum(endpoint_salience) / len(
                endpoint_salience
            )
        priors[memory_id] = {
            "binary_coverage": binary,
            "degree_centrality": centrality,
            "edge_count": float(len(edges)),
        }

    audit = {
        "memory_count": len(memory_sample),
        "memory_with_edges": sum(bool(mapped_edges.get(mid)) for mid in memory_sample),
        "memory_edge_coverage": sum(
            bool(mapped_edges.get(mid)) for mid in memory_sample
        )
        / len(memory_sample),
        "mapped_edge_assignments": mapped_edge_count,
        "unmapped_edge_rows": sum(
            not memory_by_turn.get(
                (edge["graph_id"].strip(), edge["evidence"].strip())
            )
            for edge in edge_rows
        ),
    }
    return memory_sample, priors, audit


def build_query_cache(
    query_rows: list[dict[str, str]],
    mem_ids: list[str],
    qa_ids: list[str],
    mem_embs: np.ndarray,
    qa_embs: np.ndarray,
    memory_sample: dict[str, str],
    priors: dict[str, dict[str, float]],
) -> tuple[list[dict], list[dict]]:
    mem_index = {memory_id: index for index, memory_id in enumerate(mem_ids)}
    qa_index = {query_id: index for index, query_id in enumerate(qa_ids)}
    sample_candidates: dict[str, list[int]] = defaultdict(list)
    for memory_id, sample_id in memory_sample.items():
        if memory_id in mem_index:
            sample_candidates[sample_id].append(mem_index[memory_id])

    cache = []
    excluded = []
    for row in query_rows:
        query_id = row["query_id"].strip()
        sample_id = row["conversation_id"].strip()
        gold_ids = parse_ids(row.get("gold_memory_ids", ""))
        if not gold_ids:
            excluded.append(
                {
                    "query_id": query_id,
                    "category": row["category"].strip(),
                    "split": row["split"].strip(),
                    "reason": "official evidence list is empty",
                }
            )
            continue
        if query_id not in qa_index:
            raise RuntimeError(f"Missing query embedding: {query_id}")
        candidate_indices = sample_candidates.get(sample_id, [])
        if not candidate_indices:
            raise RuntimeError(f"No sample-scoped candidates for {query_id}")

        candidate_ids = [mem_ids[index] for index in candidate_indices]
        missing_gold = sorted(set(gold_ids) - set(candidate_ids))
        if missing_gold:
            raise RuntimeError(
                f"{query_id} has gold outside its sample-scoped candidate pool: "
                f"{missing_gold}"
            )

        dense = mem_embs[candidate_indices] @ qa_embs[qa_index[query_id]]
        cache.append(
            {
                "query_id": query_id,
                "category": row["category"].strip(),
                "split": row["split"].strip(),
                "sample_id": sample_id,
                "gold_ids": gold_ids,
                "candidate_ids": candidate_ids,
                "dense_z": zscore(dense),
                "binary_coverage_z": zscore(
                    np.array(
                        [priors[mid]["binary_coverage"] for mid in candidate_ids],
                        dtype=np.float64,
                    )
                ),
                "degree_centrality_z": zscore(
                    np.array(
                        [priors[mid]["degree_centrality"] for mid in candidate_ids],
                        dtype=np.float64,
                    )
                ),
            }
        )
    return cache, excluded


def rank_query(item: dict, prior_type: str, kg_lambda: float) -> tuple[list[str], list]:
    final = item["dense_z"] + kg_lambda * item[f"{prior_type}_z"]
    candidate_ids = item["candidate_ids"]
    top_indices = sorted(
        range(len(candidate_ids)),
        key=lambda index: (-float(final[index]), candidate_ids[index]),
    )[:10]
    ranking = [candidate_ids[index] for index in top_indices]
    score_rows = [
        (
            candidate_ids[index],
            float(item["dense_z"][index]),
            float(item[f"{prior_type}_z"][index]),
            float(final[index]),
        )
        for index in top_indices
    ]
    return ranking, score_rows


def evaluate(cache: list[dict], prior_type: str, kg_lambda: float, split: str) -> dict:
    selected = cache if split == "all" else [row for row in cache if row["split"] == split]
    metrics = []
    for item in selected:
        ranking, _ = rank_query(item, prior_type, kg_lambda)
        metrics.append(per_query_metrics(item["gold_ids"], ranking))
    return summarize(metrics)


def bootstrap_delta(
    paired_rows: list[dict[str, float]], metric: str, iterations: int = 10000
) -> dict[str, float]:
    deltas = np.array([row[f"kg_{metric}"] - row[f"dense_{metric}"] for row in paired_rows])
    rng = np.random.default_rng(20260731)
    sample_indices = rng.integers(
        0, len(deltas), size=(iterations, len(deltas)), endpoint=False
    )
    boot = deltas[sample_indices].mean(axis=1)
    return {
        "delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_p_two_sided": float(
            min(1.0, 2.0 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
        ),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (
        memory_rows,
        query_rows,
        edge_rows,
        mem_ids,
        qa_ids,
        mem_embs,
        qa_embs,
    ) = load_inputs()
    memory_sample, priors, kg_audit = build_memory_priors(memory_rows, edge_rows)
    cache, excluded = build_query_cache(
        query_rows,
        mem_ids,
        qa_ids,
        mem_embs,
        qa_embs,
        memory_sample,
        priors,
    )

    grid_rows = []
    best_configs = {}
    for prior_type in PRIOR_TYPES:
        for kg_lambda in LAMBDAS:
            dev = evaluate(cache, prior_type, kg_lambda, "dev")
            grid_rows.append(
                {
                    "prior_type": prior_type,
                    "kg_lambda": kg_lambda,
                    "split": "dev",
                    **dev,
                }
            )
        dev_rows = [
            row
            for row in grid_rows
            if row["prior_type"] == prior_type and row["split"] == "dev"
        ]
        best = max(
            dev_rows,
            key=lambda row: (
                row["MRR@10"],
                row["nDCG@10"],
                row["Recall@10"],
                -row["kg_lambda"],
            ),
        )
        best_configs[prior_type] = float(best["kg_lambda"])

    summary_rows = []
    ranking_rows = []
    significance_rows = []
    for prior_type in PRIOR_TYPES:
        kg_lambda = best_configs[prior_type]
        for split in ("dev", "test", "all"):
            summary_rows.append(
                {
                    "prior_type": prior_type,
                    "selected_on": "dev MRR@10",
                    "kg_lambda": kg_lambda,
                    "split": split,
                    **evaluate(cache, prior_type, kg_lambda, split),
                }
            )

        paired_test = []
        for item in [row for row in cache if row["split"] == "test"]:
            dense_ranking, _ = rank_query(item, prior_type, 0.0)
            kg_ranking, score_rows = rank_query(item, prior_type, kg_lambda)
            dense_metrics = per_query_metrics(item["gold_ids"], dense_ranking)
            kg_metrics = per_query_metrics(item["gold_ids"], kg_ranking)
            paired_test.append(
                {
                    **{f"dense_{metric}": dense_metrics[metric] for metric in METRICS},
                    **{f"kg_{metric}": kg_metrics[metric] for metric in METRICS},
                }
            )
            if prior_type == "degree_centrality":
                ranking_rows.extend(
                    {
                        "query_id": item["query_id"],
                        "category": item["category"],
                        "split": item["split"],
                        "prior_type": prior_type,
                        "kg_lambda": kg_lambda,
                        "rank": rank,
                        "memory_id": memory_id,
                        "dense_z": dense_score,
                        "kg_prior_z": kg_score,
                        "final_score": final_score,
                        "is_gold": int(memory_id in set(item["gold_ids"])),
                    }
                    for rank, (memory_id, dense_score, kg_score, final_score) in enumerate(
                        score_rows, start=1
                    )
                )

        if prior_type == "degree_centrality":
            for item in [row for row in cache if row["split"] != "test"]:
                _, score_rows = rank_query(item, prior_type, kg_lambda)
                ranking_rows.extend(
                    {
                        "query_id": item["query_id"],
                        "category": item["category"],
                        "split": item["split"],
                        "prior_type": prior_type,
                        "kg_lambda": kg_lambda,
                        "rank": rank,
                        "memory_id": memory_id,
                        "dense_z": dense_score,
                        "kg_prior_z": kg_score,
                        "final_score": final_score,
                        "is_gold": int(memory_id in set(item["gold_ids"])),
                    }
                    for rank, (memory_id, dense_score, kg_score, final_score) in enumerate(
                        score_rows, start=1
                    )
                )

        for metric in ("MRR@10", "Recall@10", "nDCG@10"):
            significance_rows.append(
                {
                    "prior_type": prior_type,
                    "kg_lambda": kg_lambda,
                    "split": "test",
                    "metric": metric,
                    **bootstrap_delta(paired_test, metric),
                }
            )

    metric_fields = ["n", *METRICS]
    write_csv(
        REPORT_DIR / "dense_global_kg_dev_grid.csv",
        grid_rows,
        ["prior_type", "kg_lambda", "split", *metric_fields],
    )
    write_csv(
        REPORT_DIR / "dense_global_kg_summary.csv",
        summary_rows,
        [
            "prior_type",
            "selected_on",
            "kg_lambda",
            "split",
            *metric_fields,
        ],
    )
    write_csv(
        REPORT_DIR / "dense_global_kg_top10.csv",
        ranking_rows,
        [
            "query_id",
            "category",
            "split",
            "prior_type",
            "kg_lambda",
            "rank",
            "memory_id",
            "dense_z",
            "kg_prior_z",
            "final_score",
            "is_gold",
        ],
    )
    write_csv(
        REPORT_DIR / "dense_global_kg_significance.csv",
        significance_rows,
        [
            "prior_type",
            "kg_lambda",
            "split",
            "metric",
            "delta",
            "ci95_low",
            "ci95_high",
            "bootstrap_p_two_sided",
        ],
    )
    write_csv(
        REPORT_DIR / "evidence_empty_queries.csv",
        excluded,
        ["query_id", "category", "split", "reason"],
    )

    manifest = {
        "experiment": "Dense+GlobalKG publication-safe rerun",
        "method": {
            "candidate_scope": "all memories from the query's LoCoMo conversation",
            "dense_score": "query-wise z-score of cosine similarity",
            "legacy_prior": "z-score of binary KG-edge coverage",
            "corrected_prior": (
                "z-score of query-independent memory degree centrality: "
                "log1p(edge count) plus mean log1p(endpoint degrees)"
            ),
            "fusion": "dense_z + kg_lambda * kg_prior_z",
            "lambda_grid": list(LAMBDAS),
            "selection": (
                "maximize dev MRR@10; tie-break by dev nDCG@10, dev Recall@10, "
                "then smaller lambda"
            ),
            "final_reporting": "held-out test; all-query aggregate is diagnostic",
        },
        "query_counts": {
            "cat1_4_total": len(query_rows),
            "evaluable": len(cache),
            "evidence_empty": len(excluded),
            "split": dict(Counter(row["split"] for row in cache)),
        },
        "kg_mapping_audit": kg_audit,
        "selected_lambdas": best_configs,
        "inputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (
                MEM_EMB_PATH,
                QA_EMB_PATH,
                MEM_ID_PATH,
                QA_ID_PATH,
                MEMORY_PATH,
                QUERY_PATH,
                KG_EDGE_PATH,
            )
        ],
        "outputs": [
            "dense_global_kg_dev_grid.csv",
            "dense_global_kg_summary.csv",
            "dense_global_kg_top10.csv",
            "dense_global_kg_significance.csv",
            "evidence_empty_queries.csv",
        ],
    }
    (REPORT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
