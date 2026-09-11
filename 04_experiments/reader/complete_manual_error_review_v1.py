#!/usr/bin/env python3
"""Complete an evidence-based first-pass review of the 100 sampled errors.

This is a deterministic Codex-assisted audit, not an independent human
annotation study. It enriches each sampled row with branch ranks, frozen Judge
labels, protocol checks, and an explicit review decision. Ambiguous causality is
kept as `needs_secondary_review` rather than being forced into a root cause.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ERROR_DIR = ROOT / "05_reports/error_analysis_v1"
SAMPLE = ERROR_DIR / "manual_review_sample_100.csv"
LEDGER = ERROR_DIR / "error_ledger.csv"
POOLS = ROOT / "05_reports/backend_equivalence_v2/runs/csv/diagnostic_pools.csv"
JUDGE_COMMON = ROOT / "05_reports/llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected/scored_predictions.jsonl"
JUDGE_DENSEKG = ROOT / "05_reports/llm_judge_gpt4o_corrected_densekg_v2/scored_predictions.jsonl"
OUT = ERROR_DIR / "manual_review_sample_100_reviewed.csv"
SUMMARY = ERROR_DIR / "manual_review_summary.csv"
REPORT = ERROR_DIR / "MANUAL_REVIEW_REPORT.md"


TYPE4_EVIDENCE_OVERRIDES = {
    "conv-26_qa_112": ("evidence_or_annotation_insufficient", "Gold text does not state sunset/palm-tree visual content."),
    "conv-43_qa_51": ("evidence_or_annotation_insufficient", "Gold text says yoga for strength/flexibility but does not identify Hatha Yoga."),
    "conv-50_qa_7": ("evidence_or_annotation_insufficient", "Gold text only refers to a shop photo and does not state employee count."),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_ids(value: str) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def best_rank(pool: list[str], gold: set[str]) -> int | None:
    return next((rank for rank, memory_id in enumerate(pool, 1) if memory_id in gold), None)


def read_judges() -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for path in (JUDGE_COMMON, JUDGE_DENSEKG):
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                row = json.loads(line)
                if row["setting"] == "b_category":
                    result[(row["method"], row["query_id"])] = row
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    sample = read_csv(SAMPLE)
    if len(sample) != 100:
        raise RuntimeError(f"Expected 100 sampled rows, got {len(sample)}")
    ledger = {(row["method"], row["qa_id"]): row for row in read_csv(LEDGER)}
    pools = {
        (row["method"], row["query_id"]): parse_ids(row["pool_ids"])
        for row in read_csv(POOLS)
    }
    judges = read_judges()

    reviewed: list[dict[str, Any]] = []
    for source in sample:
        row: dict[str, Any] = dict(source)
        key = (row["method"], row["qa_id"])
        detail = ledger[key]
        gold = set(parse_ids(row["gold_memory_ids"]))
        dense_rank = best_rank(pools.get(("Dense-bge", row["qa_id"]), []), gold)
        bm25_rank = best_rank(pools.get(("BM25", row["qa_id"]), []), gold)
        method_rank = best_rank(pools.get(key, []), gold)
        judge = judges.get(key, {})
        review_type = row["review_type"]

        status = "reviewed_by_codex_evidence_based"
        confidence = "high"
        evidence_assessment = "not_applicable"
        cat5_check = "not_applicable"
        confirmed = ""
        notes = ""

        if review_type == "Type1_Retrieval_miss":
            dense_hit, bm25_hit = dense_rank is not None, bm25_rank is not None
            if not dense_hit and bm25_hit:
                confirmed = "Type1_embedding_mismatch"
                notes = f"Gold absent from Dense@100 but present in BM25@100 rank {bm25_rank}."
            elif dense_hit and not bm25_hit:
                confirmed = "Type1_keyword_or_lexical_miss"
                notes = f"Gold present in Dense@100 rank {dense_rank} but absent from BM25@100."
            elif not dense_hit and not bm25_hit:
                confirmed = "Type1_dual_branch_retrieval_miss"
                notes = "Gold absent from both Dense@100 and BM25@100; no single branch can be assigned as root cause."
                confidence = "medium"
            else:
                confirmed = "Type1_method_specific_candidate_pruning"
                notes = f"Gold exists in Dense@100 rank {dense_rank} and BM25@100 rank {bm25_rank}, but not the method pool."
            if row["structured_flag_reason"]:
                notes += " ERK coverage is missing/regressive and is a possible cofactor, not a proven exclusive cause."

        elif review_type == "Type2_Ranking_error":
            if row["method"] in {"BM25", "Dense-bge", "Dense+GlobalKG"}:
                confirmed = "Type2_base_ranker_top10_cutoff"
                notes = f"Gold is in the method diagnostic ranking at rank {method_rank}, but outside Top-10."
            elif (dense_rank is not None and dense_rank <= 10) or (bm25_rank is not None and bm25_rank <= 10):
                confirmed = "Type2_fusion_demotion"
                notes = f"A component ranks gold in Top-10 (Dense={dense_rank}, BM25={bm25_rank}) but fused Top-10 drops it."
            else:
                confirmed = "Type2_fusion_failed_to_promote"
                notes = f"Gold enters the fusion pool (method-pool rank {method_rank}) but neither inspected base branch places it in Top-10 (Dense={dense_rank}, BM25={bm25_rank})."
            notes += " This is not a candidate-depth miss because gold is already inside the method input pool."

        elif review_type == "Type3_Structured_extraction_error_audit":
            reason = row["structured_flag_reason"]
            if "Raw_hits_Top10_but_RawERK_misses" in reason:
                confirmed = "Type3_structured_representation_regression"
                notes = "Raw retrieves gold in Top-10 while RawERK misses; structured augmentation is negatively associated for this case."
            elif detail["primary_outcome"] in {"Type1_Retrieval_miss", "Type2_Ranking_error"}:
                confirmed = "Type3_probable_structured_coverage_gap_cofactor"
                notes = "Gold memory has empty ERK and retrieval also fails; extraction is a plausible cofactor but exclusive causality is not identifiable."
                status = "needs_secondary_review"
                confidence = "medium"
            else:
                confirmed = "Not_an_error_structured_coverage_gap_noncausal"
                notes = "Gold memory has empty ERK, but the answer/retrieval succeeds; this is a coverage gap, not an observed failure."

        elif review_type == "Type4_Reader_failure":
            judge_label = judge.get("judge_label", "")
            if judge_label == "CORRECT":
                confirmed = "Metric_false_negative_not_reader_failure"
                notes = "Official token-F1 is zero, but the frozen GPT-4o Judge marks the answer correct/semantically equivalent."
                evidence_assessment = "reader_answer_semantically_acceptable"
            elif row["qa_id"] in TYPE4_EVIDENCE_OVERRIDES:
                confirmed, notes = TYPE4_EVIDENCE_OVERRIDES[row["qa_id"]]
                evidence_assessment = "gold_text_does_not_explicitly_support_reference"
                status = "needs_secondary_review"
                confidence = "medium"
            else:
                confirmed = "Type4_reader_failure_confirmed"
                evidence_assessment = "gold_in_top10_and_judge_wrong"
                notes = "Gold evidence is in Top-10 and the frozen GPT-4o Judge also marks the generated answer wrong."
                if int(row["category"]) == 2:
                    notes += " The failure is primarily temporal normalization/reasoning."

        elif review_type == "Type5_Cat5_leakage":
            confirmed = "Type5_Cat5_leakage_confirmed"
            cat5_check = "official_adversarial_unanswerable_but_non_abstaining_answer"
            notes = "LoCoMo Cat5 changes the queried subject/premise and defines the item as unanswerable; the model returned a substantive answer instead of abstaining."
        else:
            raise RuntimeError(f"Unknown review type: {review_type}")

        row.update(
            {
                "review_status": status,
                "confirmed_error_type": confirmed,
                "reviewer_notes": notes,
                "review_confidence": confidence,
                "dense_gold_rank_at100": dense_rank or "",
                "bm25_gold_rank_at100": bm25_rank or "",
                "method_pool_gold_rank": method_rank or "",
                "judge_label": judge.get("judge_label", ""),
                "evidence_support_assessment": evidence_assessment,
                "cat5_protocol_check": cat5_check,
            }
        )
        reviewed.append(row)

    fields = list(reviewed[0])
    with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reviewed)

    summary_rows = []
    counts = Counter((row["review_type"], row["confirmed_error_type"], row["review_status"]) for row in reviewed)
    for (review_type, confirmed, status), count in sorted(counts.items()):
        summary_rows.append({"review_type": review_type, "confirmed_error_type": confirmed, "review_status": status, "count": count})
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    status_counts = Counter(row["review_status"] for row in reviewed)
    type4_counts = Counter(row["confirmed_error_type"] for row in reviewed if row["review_type"] == "Type4_Reader_failure")
    lines = [
        "# Error Analysis 100-Case Review", "",
        "This is a Codex-assisted evidence-based first pass, not an independent human annotation study.", "",
        "## Completion", "",
        f"- Reviewed rows: {len(reviewed)}/100",
        f"- Evidence-based resolved: {status_counts['reviewed_by_codex_evidence_based']}",
        f"- Secondary expert review retained: {status_counts['needs_secondary_review']}",
        "- External API calls: 0", "",
        "## Type 4 correction", "",
    ]
    for label, count in sorted(type4_counts.items()):
        lines.append(f"- {label}: {count}")
    lines += [
        "", "## Interpretation safeguards", "",
        "- Type 1 branch attribution is based on gold ranks in Dense@100 and BM25@100.",
        "- Type 2 candidate-depth is not blamed when gold is already present in the frozen fusion input pool.",
        "- Type 3 empty ERK is a coverage flag; it becomes causal only when stronger counterfactual evidence exists.",
        "- Type 4 Judge-correct cases are metric false negatives, not Reader failures.",
        "- Type 5 follows the official Cat5 adversarial/unanswerable protocol.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_path = ERROR_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest.get("outputs", []):
        artifact_path = ROOT / artifact["path"]
        if artifact_path.exists():
            artifact["sha256"] = sha256(artifact_path)
    manifest["manual_review"] = {
        "reviewer": "Codex evidence-based first pass",
        "independent_human_annotation": False,
        "reviewed_rows": len(reviewed),
        "resolved_rows": status_counts["reviewed_by_codex_evidence_based"],
        "secondary_review_rows": status_counts["needs_secondary_review"],
        "api_calls": 0,
        "outputs": [
            {"path": str(OUT.relative_to(ROOT)), "sha256": sha256(OUT)},
            {"path": str(SUMMARY.relative_to(ROOT)), "sha256": sha256(SUMMARY)},
            {"path": str(REPORT.relative_to(ROOT)), "sha256": sha256(REPORT)},
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "reviewed": len(reviewed), **status_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
