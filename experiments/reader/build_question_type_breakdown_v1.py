#!/usr/bin/env python3
"""Build retrieval + Reader question-type mechanism analysis from frozen data."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/retrieval/question_type_breakdown_v1"
RETRIEVAL = ROOT / "results/retrieval/retrieval_main_table/retrieval_main_per_query.csv"
READER = ROOT / "results/reader/reader_offline_metrics_v4/reader_metrics_per_query.jsonl"
METHODS = ("BM25", "Dense-bge", "RRF_compact", "ZScore-Raw", "ZScore-RawERK")
CATEGORY_NAMES = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop", 5: "Adversarial"}
ORDER = (4, 1, 2, 3, 5)
COMPARISONS = (
    ("Dense_to_CassMem", "Dense-bge", "ZScore-RawERK", "total_pipeline_gain"),
    ("Raw_to_RawERK", "ZScore-Raw", "ZScore-RawERK", "incremental_ERK_gain"),
)
SEED = 20260812
BOOTSTRAPS = 10000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_bootstrap(deltas: list[float], seed_offset: int) -> tuple[float, float, float]:
    if not deltas:
        return 0.0, 0.0, 1.0
    rng = np.random.default_rng(SEED + seed_offset)
    values = np.asarray(deltas, dtype=np.float64)
    n = len(values)
    chunks: list[np.ndarray] = []
    for start in range(0, BOOTSTRAPS, 500):
        size = min(500, BOOTSTRAPS - start)
        indices = rng.integers(0, n, size=(size, n))
        chunks.append(values[indices].mean(axis=1))
    estimates = np.sort(np.concatenate(chunks))
    low = float(estimates[int(0.025 * BOOTSTRAPS)])
    high = float(estimates[int(0.975 * BOOTSTRAPS)])
    nonpositive = float(np.mean(estimates <= 0))
    nonnegative = float(np.mean(estimates >= 0))
    return low, high, min(1.0, 2 * min(nonpositive, nonnegative))


def main() -> None:
    retrieval_rows = [
        row for row in read_csv(RETRIEVAL)
        if row["method"] in METHODS and row["comparable"] == "True"
    ]
    retrieval_values = {
        (row["method"], row["query_id"]): (int(row["category"]), float(row["MRR@10"]))
        for row in retrieval_rows
    }

    reader_rows = []
    with READER.open(encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            if row["method"] in METHODS:
                reader_rows.append(row)
    reader_values = {
        (row["setting"], row["method"], row["qa_id"]): (int(row["category"]), float(row["official_f1"]))
        for row in reader_rows
    }

    aggregate_rows: list[dict] = []
    for method in METHODS:
        selected = [(cat, value) for (m, _), (cat, value) in retrieval_values.items() if m == method]
        for category in (4, 1, 2, 3):
            values = [value for cat, value in selected if cat == category]
            aggregate_rows.append({
                "layer": "retrieval", "setting": "Cat1-4", "metric": "MRR@10",
                "method": method, "category": category, "category_label": CATEGORY_NAMES[category],
                "n": len(values), "score": mean(values),
            })
        aggregate_rows.append({
            "layer": "retrieval", "setting": "Cat1-4", "metric": "MRR@10",
            "method": method, "category": "Overall", "category_label": "Overall",
            "n": len(selected), "score": mean(value for _, value in selected),
        })

    for setting, setting_label in (("a_unified", "Cat.x"), ("b_category", "Cat.v")):
        for method in METHODS:
            selected = [
                (cat, value) for (s, m, _), (cat, value) in reader_values.items()
                if s == setting and m == method
            ]
            for category in ORDER:
                values = [value for cat, value in selected if cat == category]
                aggregate_rows.append({
                    "layer": "reader", "setting": setting_label, "metric": "F1",
                    "method": method, "category": category, "category_label": CATEGORY_NAMES[category],
                    "n": len(values), "score": mean(values),
                })
            aggregate_rows.append({
                "layer": "reader", "setting": setting_label, "metric": "F1",
                "method": method, "category": "Overall", "category_label": "Overall",
                "n": len(selected), "score": mean(value for _, value in selected),
            })

    delta_rows: list[dict] = []
    seed_offset = 0
    for comparison, baseline, target, interpretation in COMPARISONS:
        for layer, setting, categories in (
            ("retrieval", "Cat1-4", (4, 1, 2, 3)),
            ("reader", "Cat.x", ORDER),
            ("reader", "Cat.v", ORDER),
        ):
            if layer == "retrieval":
                shared = sorted(
                    qid for method, qid in retrieval_values
                    if method == baseline and (target, qid) in retrieval_values
                )
                tuples = [
                    (qid, retrieval_values[(baseline, qid)][0], retrieval_values[(target, qid)][1] - retrieval_values[(baseline, qid)][1])
                    for qid in shared
                ]
                metric = "MRR@10"
            else:
                internal_setting = "a_unified" if setting == "Cat.x" else "b_category"
                shared = sorted(
                    qid for s, method, qid in reader_values
                    if s == internal_setting and method == baseline and (internal_setting, target, qid) in reader_values
                )
                tuples = [
                    (qid, reader_values[(internal_setting, baseline, qid)][0], reader_values[(internal_setting, target, qid)][1] - reader_values[(internal_setting, baseline, qid)][1])
                    for qid in shared
                ]
                metric = "F1"
            total_n = len(tuples)
            for category in categories:
                deltas = [delta for _, cat, delta in tuples if cat == category]
                seed_offset += 1
                low, high, p = paired_bootstrap(deltas, seed_offset)
                delta_rows.append({
                    "comparison": comparison, "interpretation": interpretation,
                    "layer": layer, "setting": setting, "metric": metric,
                    "category": category, "category_label": CATEGORY_NAMES[category],
                    "n": len(deltas), "baseline": baseline, "target": target,
                    "delta": mean(deltas), "ci95_low": low, "ci95_high": high,
                    "bootstrap_p_two_sided": p,
                    "weighted_contribution_to_overall_delta": len(deltas) / total_n * mean(deltas),
                })
            all_deltas = [delta for _, _, delta in tuples]
            seed_offset += 1
            low, high, p = paired_bootstrap(all_deltas, seed_offset)
            delta_rows.append({
                "comparison": comparison, "interpretation": interpretation,
                "layer": layer, "setting": setting, "metric": metric,
                "category": "Overall", "category_label": "Overall", "n": len(all_deltas),
                "baseline": baseline, "target": target, "delta": mean(all_deltas),
                "ci95_low": low, "ci95_high": high, "bootstrap_p_two_sided": p,
                "weighted_contribution_to_overall_delta": mean(all_deltas),
            })

    OUT.mkdir(parents=True, exist_ok=True)
    aggregate_path = OUT / "question_type_scores.csv"
    delta_path = OUT / "question_type_deltas.csv"
    write_csv(aggregate_path, aggregate_rows, list(aggregate_rows[0]))
    write_csv(delta_path, delta_rows, list(delta_rows[0]))

    def delta(comparison: str, layer: str, setting: str, category: str) -> dict:
        return next(
            row for row in delta_rows
            if row["comparison"] == comparison and row["layer"] == layer
            and row["setting"] == setting and str(row["category_label"]) == category
        )

    retrieval_overall = delta("Dense_to_CassMem", "retrieval", "Cat1-4", "Overall")
    reader_a_overall = delta("Dense_to_CassMem", "reader", "Cat.x", "Overall")
    reader_b_overall = delta("Dense_to_CassMem", "reader", "Cat.v", "Overall")
    raw_erk_b = [
        row for row in delta_rows
        if row["comparison"] == "Raw_to_RawERK" and row["layer"] == "reader"
        and row["setting"] == "Cat.v" and row["category"] != "Overall"
    ]
    strongest = max(raw_erk_b, key=lambda row: row["delta"])
    lines = [
        "# Question Type Breakdown V1", "",
        "## Why this analysis is needed", "",
        "The main tables establish effectiveness; this analysis tests where the gain comes from.",
        "Dense-to-CassMem is the total pipeline gain. ZScore-Raw-to-RawERK is the cleaner incremental ERK comparison.", "",
        "## Correct category counts", "",
        "| Column | LoCoMo category | Reader n | Retrieval-evaluable n |", "|---|---:|---:|---:|",
        "| Single-Hop | Cat4 | 841 | 841 |", "| Multi-Hop | Cat1 | 282 | 282 |",
        "| Temporal | Cat2 | 321 | 321 |", "| Open-Domain | Cat3 | 96 | 92 |",
        "| Adversarial | Cat5 | 446 | n/a |", "",
        "Four Cat3 questions have empty gold evidence and are excluded only from retrieval metrics.", "",
        "## Main results", "",
        f"- Retrieval Dense-to-CassMem overall MRR@10: {100*retrieval_overall['delta']:+.2f} points.",
        f"- Reader Dense-to-CassMem Overall F1: Cat.x {100*reader_a_overall['delta']:+.2f} points; Cat.v {100*reader_b_overall['delta']:+.2f} points.",
        f"- Under Cat.v, the largest incremental Raw-to-RawERK Reader gain is {strongest['category_label']} ({100*strongest['delta']:+.2f} F1 points).", "",
        "## Mechanism conclusion", "",
        "The evidence does not support a claim that ERK helps mainly Multi-Hop. Retrieval gains are largest on Single-Hop and Temporal; Reader gains depend on prompt protocol and are strongest on Open-Domain under Cat.v. Therefore the defensible claim is that compact ERK signals improve several question types, not specifically multi-hop reasoning.", "",
        "Do not infer mechanism from Dense-to-CassMem alone because that contrast changes both fusion and representation.",
    ]
    report_path = OUT / "QUESTION_TYPE_BREAKDOWN_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "version": "question-type-breakdown-v1", "api_calls": 0,
        "bootstrap_samples": BOOTSTRAPS, "seed": SEED,
        "category_mapping": CATEGORY_NAMES,
        "inputs": [
            {"path": str(RETRIEVAL.relative_to(ROOT)), "sha256": sha256(RETRIEVAL)},
            {"path": str(READER.relative_to(ROOT)), "sha256": sha256(READER)},
        ],
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (aggregate_path, delta_path, report_path)
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "scores": len(aggregate_rows), "deltas": len(delta_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
