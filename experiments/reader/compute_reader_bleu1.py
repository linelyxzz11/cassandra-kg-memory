#!/usr/bin/env python3
"""Recompute reader BLEU-1 from frozen dual-setting predictions.

Protocol: lowercase, NLTK ``word_tokenize``, sentence-level BLEU with
weights=(1, 0, 0, 0), and smoothing method1. Cat1-4 excludes adversarial
category 5. A Full5 lexical BLEU value is retained for audit only. The
publication-facing five-category aggregate is explicitly named a hybrid
BLEU-1/Cat5-accuracy metric because LoCoMo's official Cat5 score is binary
abstention correctness, not answer-overlap BLEU.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "results"
    / "retrieval"
    / "official_eval"
    / "gpt4o_dual_setting_locomo_corrected_v3"
)
INPUTS = {
    "Cat.✗": REPORT_DIR / "settingA_scored_predictions.jsonl",
    "Cat.✓": REPORT_DIR / "settingB_scored_predictions.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokens(text: str) -> list[str]:
    return word_tokenize(text.lower())


def paired_bootstrap(
    ours: list[float],
    baseline: list[float],
    *,
    seed: int,
    resamples: int = 10000,
) -> dict[str, float]:
    deltas = np.asarray(ours, dtype=float) - np.asarray(baseline, dtype=float)
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for start in range(0, resamples, 500):
        batch = min(500, resamples - start)
        indices = rng.integers(0, len(deltas), size=(batch, len(deltas)))
        samples.append(deltas[indices].mean(axis=1))
    boot = np.concatenate(samples)
    lower_tail = (int(np.count_nonzero(boot <= 0)) + 1) / (resamples + 1)
    upper_tail = (int(np.count_nonzero(boot >= 0)) + 1) / (resamples + 1)
    return {
        "delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_p_two_sided": min(1.0, 2.0 * min(lower_tail, upper_tail)),
        "resamples": resamples,
        "seed": seed,
    }


def main() -> None:
    smoothing = SmoothingFunction().method1
    output_rows: list[dict] = []
    per_query: list[dict] = []
    row_counts: dict[str, int] = {}

    for setting, path in INPUTS.items():
        by_method: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                row = json.loads(line)
                reference = tokens(row["gold_answer"])
                hypothesis = tokens(row["prediction_for_eval"])
                score = sentence_bleu(
                    [reference],
                    hypothesis,
                    weights=(1, 0, 0, 0),
                    smoothing_function=smoothing,
                )
                by_method[row["method"]].append(
                    (int(row["category"]), score, float(row["official_f1"]))
                )
                per_query.append(
                    {
                        "qa_id": row["qa_id"],
                        "method": row["method"],
                        "setting": setting,
                        "category": int(row["category"]),
                        "bleu1": score,
                        "official_f1": float(row["official_f1"]),
                        "hybrid": (
                            float(row["official_f1"])
                            if int(row["category"]) == 5
                            else score
                        ),
                    }
                )

        row_counts[setting] = sum(len(rows) for rows in by_method.values())
        for method, scored in by_method.items():
            cat1_4 = [score for category, score, _ in scored if category != 5]
            full5_lexical = [score for _, score, _ in scored]
            full5_hybrid = [
                official_score if category == 5 else score
                for category, score, official_score in scored
            ]
            output_rows.append(
                {
                    "method": method,
                    "setting": setting,
                    "cat1_4_bleu1": sum(cat1_4) / len(cat1_4),
                    "full5_lexical_bleu1_audit": (
                        sum(full5_lexical) / len(full5_lexical)
                    ),
                    "full5_hybrid_bleu1_cat5_accuracy": (
                        sum(full5_hybrid) / len(full5_hybrid)
                    ),
                    "cat1_4_n": len(cat1_4),
                    "full5_n": len(full5_lexical),
                }
            )

    method_order = {
        method: index
        for index, method in enumerate(
            (
                "BM25",
                "Dense-bge",
                "Dense+GlobalKG",
                "RRF_compact",
                "ZScore-Raw",
                "ZScore-RawERK",
            )
        )
    }
    setting_order = {"Cat.✗": 0, "Cat.✓": 1}
    output_rows.sort(
        key=lambda row: (setting_order[row["setting"]], method_order[row["method"]])
    )

    output_path = REPORT_DIR / "bleu1_offline_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    setting_b = [row for row in per_query if row["setting"] == "Cat.✓"]
    by_method_query = defaultdict(dict)
    for row in setting_b:
        by_method_query[row["method"]][row["qa_id"]] = row
    significance_rows = []
    metric_specs = (
        ("cat1_4_f1", "official_f1", lambda row: row["category"] != 5),
        ("full5_f1", "official_f1", lambda row: True),
        ("cat1_4_bleu1", "bleu1", lambda row: row["category"] != 5),
        ("full5_hybrid", "hybrid", lambda row: True),
    )
    for baseline in ("RRF_compact", "ZScore-Raw"):
        shared = sorted(
            set(by_method_query["ZScore-RawERK"]) & set(by_method_query[baseline])
        )
        for metric_index, (metric, field, include) in enumerate(metric_specs):
            selected = [
                query_id
                for query_id in shared
                if include(by_method_query["ZScore-RawERK"][query_id])
            ]
            stats = paired_bootstrap(
                [
                    by_method_query["ZScore-RawERK"][query_id][field]
                    for query_id in selected
                ],
                [
                    by_method_query[baseline][query_id][field]
                    for query_id in selected
                ],
                seed=20260741 + metric_index,
            )
            significance_rows.append(
                {
                    "method": "ZScore-RawERK",
                    "baseline": baseline,
                    "setting": "Cat.✓",
                    "metric": metric,
                    "n": len(selected),
                    **stats,
                }
            )
    order = sorted(
        range(len(significance_rows)),
        key=lambda index: significance_rows[index]["bootstrap_p_two_sided"],
    )
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(
            1.0,
            (len(order) - rank)
            * significance_rows[index]["bootstrap_p_two_sided"],
        )
        running = max(running, adjusted)
        significance_rows[index]["holm_adjusted_p"] = running
    significance_path = REPORT_DIR / "reader_main_significance.csv"
    with significance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "baseline",
                "setting",
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
        writer.writeheader()
        writer.writerows(significance_rows)

    manifest = {
        "metric": "BLEU-1",
        "protocol": {
            "case": "lowercase",
            "tokenizer": "nltk.word_tokenize",
            "aggregation": "mean sentence_bleu across questions",
            "weights": [1, 0, 0, 0],
            "smoothing": "nltk SmoothingFunction.method1",
            "cat1_4": "categories 1-4 only",
            "full5_lexical_audit": (
                "lexical BLEU-1 on categories 1-5; not aligned with official Cat5"
            ),
            "full5_hybrid": (
                "Cat1-4 sentence BLEU-1 plus official binary Cat5 accuracy, "
                "micro-averaged over all 1986 questions; do not label this BLEU-1"
            ),
        },
        "inputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in INPUTS.values()
        ],
        "input_row_counts": row_counts,
        "output": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "rows": len(output_rows),
        },
        "significance_output": {
            "path": str(significance_path),
            "sha256": sha256(significance_path),
            "rows": len(significance_rows),
            "comparisons": "CassMem vs RRF_compact and ZScore-Raw, Setting B",
            "multiple_testing": "Holm correction across 8 comparisons",
        },
    }
    (REPORT_DIR / "bleu1_protocol.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["output"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
