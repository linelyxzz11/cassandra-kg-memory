#!/usr/bin/env python3
"""Rebuild six-method Reader metrics from frozen predictions without API calls.

This normalizes Cat5 option labels to their deterministic option text before
LoCoMo F1, BLEU-1, and downstream judge-cache alignment.  It never changes the
raw prediction files.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "03_src/evaluation/run_locomo_prompt_protocol_gpt4o.py"
EVALUATOR_PATH = ROOT / "03_src/evaluation/locomo_official_eval_v1.py"
LEGACY_ROOT = ROOT / "05_reports/locomo_gpt4o_prompt_protocol"
CORRECTED_DENSEKG_ROOT = ROOT / "05_reports/locomo_gpt4o_prompt_protocol_corrected_densekg_v2"
OUTPUT = ROOT / "05_reports/reader_offline_metrics_v4"
METHODS = ("BM25", "Dense-bge", "Dense+GlobalKG", "RRF_compact", "ZScore-Raw", "ZScore-RawERK")
SETTINGS = (
    ("setting_a_unified", "a_unified", "Cat.x"),
    ("setting_b_category", "b_category", "Cat.v"),
)
CAT5_REFERENCE = "Not mentioned in the conversation"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_choice(value: str) -> str:
    text = str(value or "").strip().lower()
    patterns = (
        r"^\s*\(?\s*([ab])\s*\)?\s*[\.:,;!]?\s*$",
        r"^\s*(?:answer|option|choice)\s*(?:is|:)?\s*\(?\s*([ab])\s*\)?",
        r"^\s*\(?\s*([ab])\s*\)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def normalize_prediction(row: dict[str, Any], setting: str, runner) -> dict[str, str]:
    raw = str(row.get("prediction", "") or "").strip()
    category = str(row["category"])
    if category != "5" or setting != "b_category":
        return {
            "prediction_for_eval": raw,
            "option_a_text": "",
            "option_b_text": "",
            "parsed_choice": "",
            "mapped_option_text": "",
            "parse_status": "n/a",
        }

    adversarial = str(row.get("gold_answer", "") or "").strip()
    if runner.deterministic_swap(str(row["qa_id"])):
        option_a, option_b = adversarial, CAT5_REFERENCE
    else:
        option_a, option_b = CAT5_REFERENCE, adversarial
    choice = parse_choice(raw)
    mapped = option_a if choice == "a" else option_b if choice == "b" else ""
    return {
        "prediction_for_eval": mapped or raw,
        "option_a_text": option_a,
        "option_b_text": option_b,
        "parsed_choice": choice,
        "mapped_option_text": mapped,
        "parse_status": "choice_parsed" if mapped else "unparsed_passthrough",
    }


def bleu1(reference: str, prediction: str, smoothing) -> float:
    ref = word_tokenize(str(reference).lower())
    hyp = word_tokenize(str(prediction).lower())
    if not hyp:
        return 0.0
    return float(sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=smoothing))


def main() -> None:
    runner = load_module("reader_runner_v4", RUNNER_PATH)
    evaluator = load_module("locomo_eval_v4", EVALUATOR_PATH)
    smoothing = SmoothingFunction().method1
    all_rows: list[dict[str, Any]] = []
    input_manifest: list[dict[str, Any]] = []

    for setting_dir, setting, setting_label in SETTINGS:
        for method in METHODS:
            base = CORRECTED_DENSEKG_ROOT if method == "Dense+GlobalKG" else LEGACY_ROOT
            path = base / setting_dir / method / "reader_predictions.jsonl"
            rows = read_jsonl(path)
            if len(rows) != 1986 or len({str(row["qa_id"]) for row in rows}) != 1986:
                raise RuntimeError(f"Incomplete Reader cache: {path}")
            input_manifest.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "rows": len(rows)})
            counts = Counter(str(row["category"]) for row in rows)
            if counts != Counter({"1": 282, "2": 321, "3": 96, "4": 841, "5": 446}):
                raise RuntimeError(f"Category mismatch in {path}: {dict(counts)}")

            for row in rows:
                category = int(row["category"])
                normalized = normalize_prediction(row, setting, runner)
                scored = evaluator.score_one(category, normalized["prediction_for_eval"], row["gold_answer"])
                lexical_reference = CAT5_REFERENCE if category == 5 else str(row["gold_answer"])
                b1 = bleu1(lexical_reference, normalized["prediction_for_eval"], smoothing)
                all_rows.append({
                    "method": method,
                    "setting": setting,
                    "setting_label": setting_label,
                    "qa_id": row["qa_id"],
                    "category": category,
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "cat5_lexical_reference": CAT5_REFERENCE if category == 5 else "",
                    "raw_prediction": row["prediction"],
                    **normalized,
                    "official_f1": float(scored["official_score"]),
                    "bleu1": b1,
                    "top10_memory_ids": row.get("top10_memory_ids", ""),
                    "model": row.get("model", ""),
                    "temperature": row.get("temperature", ""),
                    "prediction_cache_source": str(path.relative_to(ROOT)).replace("\\", "/"),
                })

    if len(all_rows) != 2 * 6 * 1986:
        raise RuntimeError(f"Unexpected normalized row count: {len(all_rows)}")

    summary_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for setting_dir, setting, setting_label in SETTINGS:
        del setting_dir
        for method in METHODS:
            selected = [row for row in all_rows if row["setting"] == setting and row["method"] == method]
            summary_rows.append({
                "method": method,
                "setting": setting,
                "setting_label": setting_label,
                "n": len(selected),
                "overall_f1": sum(row["official_f1"] for row in selected) / len(selected),
                "overall_b1": sum(row["bleu1"] for row in selected) / len(selected),
                "cat1_4_f1": sum(row["official_f1"] for row in selected if row["category"] != 5) / 1540,
                "cat1_4_b1": sum(row["bleu1"] for row in selected if row["category"] != 5) / 1540,
                "cat5_accuracy": sum(row["official_f1"] for row in selected if row["category"] == 5) / 446,
                "cat5_bleu1": sum(row["bleu1"] for row in selected if row["category"] == 5) / 446,
                "cat5_unparsed": sum(row["parse_status"] == "unparsed_passthrough" for row in selected if row["category"] == 5),
            })
            for category in range(1, 6):
                subset = [row for row in selected if row["category"] == category]
                category_rows.append({
                    "method": method,
                    "setting": setting,
                    "setting_label": setting_label,
                    "category": category,
                    "n": len(subset),
                    "f1": sum(row["official_f1"] for row in subset) / len(subset),
                    "bleu1": sum(row["bleu1"] for row in subset) / len(subset),
                })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    normalized_path = OUTPUT / "reader_metrics_per_query.jsonl"
    normalized_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    write_csv(OUTPUT / "reader_metrics_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(OUTPUT / "reader_metrics_by_category.csv", category_rows, list(category_rows[0]))
    manifest = {
        "protocol_version": "reader-offline-v4",
        "api_calls": 0,
        "prediction_rows": len(all_rows),
        "methods": list(METHODS),
        "settings": [item[1] for item in SETTINGS],
        "f1": "official LoCoMo category-aware per-query score, micro mean over 1,986",
        "bleu1": {
            "tokenizer": "NLTK word_tokenize, lowercase",
            "weights": [1, 0, 0, 0],
            "smoothing": "method1",
            "cat5_reference": CAT5_REFERENCE,
            "cat5_prediction": "deterministic option label restored to option text before scoring",
        },
        "inputs": input_manifest,
        "outputs": [
            {"path": str(normalized_path.relative_to(ROOT)), "sha256": sha256(normalized_path)},
            {"path": "05_reports/reader_offline_metrics_v4/reader_metrics_summary.csv", "sha256": sha256(OUTPUT / "reader_metrics_summary.csv")},
            {"path": "05_reports/reader_offline_metrics_v4/reader_metrics_by_category.csv", "sha256": sha256(OUTPUT / "reader_metrics_by_category.csv")},
        ],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"api_calls": 0, "rows": len(all_rows), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
