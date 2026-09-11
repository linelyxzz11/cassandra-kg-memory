#!/usr/bin/env python3
"""Verify completeness and provenance of corrected Dense+GlobalKG reader outputs."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "03_src/evaluation/run_locomo_prompt_protocol_gpt4o.py"
OUTPUT_ROOT = ROOT / "05_reports/locomo_gpt4o_prompt_protocol_corrected_densekg_v2"


def load_runner():
    spec = importlib.util.spec_from_file_location("locomo_reader_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = load_runner()
    questions = runner.load_questions(ROOT / "01_data/locomo_qa_records.csv")
    memories = runner.load_memory_text(ROOT / "01_data/locomo_memory_records.csv")
    ranking = runner.build_full5_rankings(questions, memories, "Dense+GlobalKG")
    summaries = []
    for setting_dir, setting, use_category_fmt in [
        ("setting_a_unified", "a_unified", False),
        ("setting_b_category", "b_category", True),
    ]:
        path = OUTPUT_ROOT / setting_dir / "Dense+GlobalKG/reader_predictions.jsonl"
        rows = runner.load_records(path)
        qids = [row.get("query_id") for row in rows]
        problems = []
        returned_models = Counter()
        for row in rows:
            qid = row.get("query_id")
            if qid not in questions:
                problems.append({"query_id": qid, "problem": "unknown_qid"})
                continue
            prompt = runner.build_prompt(
                questions[qid], runner.render_context(ranking[qid], memories), use_category_fmt
            )
            if row.get("prompt_sha256") != runner.sha256_text(prompt):
                problems.append({"query_id": qid, "problem": "prompt_hash_mismatch"})
            if row.get("ranking_sha256") != runner.ranking_sha256(ranking[qid]):
                problems.append({"query_id": qid, "problem": "ranking_hash_mismatch"})
            if not str(row.get("prediction", "")).strip():
                problems.append({"query_id": qid, "problem": "empty_prediction"})
            if row.get("prediction_source") == "api_generated":
                returned_models[str(row.get("model_returned") or "missing")] += 1
        duplicate_count = len(qids) - len(set(qids))
        missing = sorted(set(questions) - set(qids))
        summary = {
            "setting": setting,
            "rows": len(rows),
            "unique_qids": len(set(qids)),
            "duplicate_qids": duplicate_count,
            "missing_qids": len(missing),
            "category_counts": dict(sorted(Counter(str(r.get("category")) for r in rows).items())),
            "prediction_sources": dict(sorted(Counter(str(r.get("prediction_source")) for r in rows).items())),
            "api_returned_models": dict(sorted(returned_models.items())),
            "problems": len(problems),
            "problem_examples": problems[:10],
        }
        summary["pass"] = (
            len(rows) == 1986
            and len(set(qids)) == 1986
            and duplicate_count == 0
            and not missing
            and not problems
            and summary["category_counts"] == {"1": 282, "2": 321, "3": 96, "4": 841, "5": 446}
        )
        summaries.append(summary)
    report = {"all_pass": all(item["pass"] for item in summaries), "settings": summaries}
    (OUTPUT_ROOT / "integrity_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
