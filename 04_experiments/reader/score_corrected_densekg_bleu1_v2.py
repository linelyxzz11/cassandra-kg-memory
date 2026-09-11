#!/usr/bin/env python3
"""Compute corrected Dense+GlobalKG B1 with the frozen Mem0/A-MEM protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


ROOT = Path(__file__).resolve().parents[2]
PRED_ROOT = ROOT / "05_reports/locomo_gpt4o_prompt_protocol_corrected_densekg_v2"
SCORE_ROOT = ROOT / "05_reports/official_eval/gpt4o_corrected_densekg_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    smoothing = SmoothingFunction().method1
    summaries = []
    for setting_dir, setting in [
        ("setting_a_unified", "Cat.x"),
        ("setting_b_category", "Cat.v"),
    ]:
        pred_path = PRED_ROOT / setting_dir / "Dense+GlobalKG/reader_predictions.jsonl"
        score_path = SCORE_ROOT / setting_dir / "per_query_scores.csv"
        predictions = {row["query_id"]: row for row in read_jsonl(pred_path)}
        with score_path.open(encoding="utf-8-sig", newline="") as handle:
            official = {row["qa_id"]: row for row in csv.DictReader(handle)}
        cat14_bleu = []
        full5_lexical = []
        full5_hybrid = []
        for qid, row in predictions.items():
            reference = word_tokenize(str(row["gold_answer"]).lower())
            hypothesis = word_tokenize(str(row["prediction"]).lower())
            bleu = sentence_bleu(
                [reference], hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing
            )
            category = str(row["category"])
            full5_lexical.append(bleu)
            if category == "5":
                full5_hybrid.append(float(official[qid]["official_score"]))
            else:
                cat14_bleu.append(bleu)
                full5_hybrid.append(bleu)
        summaries.append({
            "method": "Dense+GlobalKG",
            "setting": setting,
            "cat1_4_bleu1": sum(cat14_bleu) / len(cat14_bleu),
            "full5_lexical_bleu1_audit": sum(full5_lexical) / len(full5_lexical),
            "overall_b1_hybrid_cat5_accuracy": sum(full5_hybrid) / len(full5_hybrid),
            "cat1_4_n": len(cat14_bleu),
            "full5_n": len(full5_hybrid),
            "predictions_sha256": sha256(pred_path),
            "official_scores_sha256": sha256(score_path),
        })
    output = SCORE_ROOT / "bleu1_corrected_densekg_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    protocol = {
        "case": "lowercase",
        "tokenizer": "nltk.word_tokenize",
        "sentence_metric": "sentence_bleu weights=(1,0,0,0), method1 smoothing",
        "overall_b1": "Cat1-4 BLEU-1 plus official binary Cat5 accuracy, micro-average n=1986",
        "rows": summaries,
    }
    (SCORE_ROOT / "bleu1_corrected_densekg_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
