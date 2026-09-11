#!/usr/bin/env python3
"""Seed corrected Dense+GlobalKG reader runs only from byte-identical prompts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "03_src" / "evaluation" / "run_locomo_prompt_protocol_gpt4o.py"
OLD_ROOT = ROOT / "05_reports" / "locomo_gpt4o_prompt_protocol"
DEFAULT_NEW_ROOT = ROOT / "05_reports" / "locomo_gpt4o_prompt_protocol_corrected_densekg_v2"


def load_runner():
    spec = importlib.util.spec_from_file_location("locomo_reader_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "01_data/locomo_qa_records.csv")
    parser.add_argument("--memories", type=Path, default=ROOT / "01_data/locomo_memory_records.csv")
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runner = load_runner()
    questions = runner.load_questions(args.questions)
    memories = runner.load_memory_text(args.memories)
    corrected = runner.build_full5_rankings(questions, memories, "Dense+GlobalKG")

    summary = []
    for setting_dir, setting, use_category_fmt in [
        ("setting_a_unified", "a_unified", False),
        ("setting_b_category", "b_category", True),
    ]:
        old_path = OLD_ROOT / setting_dir / "Dense+GlobalKG" / "reader_predictions.jsonl"
        output_dir = args.new_root / setting_dir / "Dense+GlobalKG"
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = output_dir / "reader_predictions.jsonl"
        if pred_path.exists() and not args.force:
            raise RuntimeError(f"Refusing to overwrite existing file: {pred_path}")

        old_by_qid = {row["query_id"]: row for row in read_jsonl(old_path)}
        audit_rows = []
        reused_rows = []
        for qid in sorted(questions):
            q = questions[qid]
            old = old_by_qid[qid]
            old_ids = [value for value in str(old.get("top10_memory_ids", "")).split(";") if value]
            new_ids = corrected[qid]
            old_prompt = runner.build_prompt(q, runner.render_context(old_ids, memories), use_category_fmt)
            new_prompt = runner.build_prompt(q, runner.render_context(new_ids, memories), use_category_fmt)
            old_hash = runner.sha256_text(old_prompt)
            new_hash = runner.sha256_text(new_prompt)
            reusable = old_hash == new_hash and bool(str(old.get("prediction", "")).strip())
            audit_rows.append({
                "query_id": qid,
                "category": q["category"],
                "setting": setting,
                "old_prompt_sha256": old_hash,
                "new_prompt_sha256": new_hash,
                "prompt_identical": int(old_hash == new_hash),
                "action": "reuse" if reusable else "regenerate",
                "old_ranking_sha256": runner.ranking_sha256(old_ids),
                "new_ranking_sha256": runner.ranking_sha256(new_ids),
            })
            if reusable:
                copied = dict(old)
                copied.update({
                    "top10_memory_ids": ";".join(new_ids),
                    "prompt_sha256": new_hash,
                    "ranking_sha256": runner.ranking_sha256(new_ids),
                    "prediction_source": "prompt_hash_reuse",
                    "source_prediction_file": str(old_path.relative_to(ROOT)).replace("\\", "/"),
                })
                reused_rows.append(copied)

        pred_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reused_rows),
            encoding="utf-8",
        )
        audit_path = output_dir / "prompt_reuse_audit.csv"
        with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
            writer.writeheader()
            writer.writerows(audit_rows)
        action_counts = Counter(row["action"] for row in audit_rows)
        cat_reuse = Counter(row["category"] for row in audit_rows if row["action"] == "reuse")
        item = {
            "setting": setting,
            "total": len(audit_rows),
            "reused": action_counts["reuse"],
            "api_regeneration_required": action_counts["regenerate"],
            "reused_by_category": dict(sorted(cat_reuse.items())),
            "reuse_rule": "old_prompt_sha256 == new_prompt_sha256 and nonempty prediction",
        }
        (output_dir / "reuse_manifest.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary.append(item)

    args.new_root.mkdir(parents=True, exist_ok=True)
    (args.new_root / "reuse_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
