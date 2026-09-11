#!/usr/bin/env python3
"""Build a reproducible retrieval-reader error funnel and 100-case review set."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "05_reports/error_analysis_v1"
METRICS = ROOT / "05_reports/reader_offline_metrics_v4/reader_metrics_per_query.jsonl"
POOLS = ROOT / "05_reports/backend_equivalence_v2/runs/csv/diagnostic_pools.csv"
TOP10 = ROOT / "05_reports/backend_equivalence_v2/runs/csv/top10.csv"
GOLD = ROOT / "02_artifacts/retrieval_gold_v2/locomo_cat1_4_gold_memory.csv"
MEMORIES = ROOT / "01_data/locomo_memory_records.csv"
FEATURES = ROOT / "02_artifacts/p3_memory_features.csv"
METHODS = ("BM25", "Dense-bge", "Dense+GlobalKG", "RRF_compact", "ZScore-Raw", "ZScore-RawERK")
SETTING = "b_category"
SEED = 20260812


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_ids(value: str) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in str(value or "").split(";") if item.strip()))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def hit(ids: list[str], gold: set[str]) -> bool:
    return bool(set(ids) & gold)


def main() -> None:
    metric_rows = [row for row in read_jsonl(METRICS) if row["setting"] == SETTING]
    if len(metric_rows) != 6 * 1986:
        raise RuntimeError(f"Expected 11,916 Setting-B metric rows, got {len(metric_rows)}")
    gold_rows = {row["query_id"]: row for row in read_csv(GOLD)}
    memory_rows = {row["memory_id"]: row for row in read_csv(MEMORIES)}
    feature_rows = {row["memory_id"]: row for row in read_csv(FEATURES)}

    pools: dict[tuple[str, str], list[str]] = {}
    for row in read_csv(POOLS):
        pools[(row["method"], row["query_id"])] = parse_ids(row["pool_ids"])
    rankings: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for row in read_csv(TOP10):
        rankings[(row["method"], row["query_id"])].append((int(row["rank"]), row["memory_id"]))
    top10 = {key: [memory_id for _, memory_id in sorted(values)] for key, values in rankings.items()}

    ledger: list[dict[str, Any]] = []
    for row in metric_rows:
        method, query_id, category = row["method"], row["qa_id"], int(row["category"])
        official_f1 = float(row["official_f1"])
        if category == 5:
            primary = "Correct_Cat5_abstention" if official_f1 > 0 else "Type5_Cat5_leakage"
            ledger.append({
                "method": method, "setting": SETTING, "qa_id": query_id, "category": category,
                "primary_outcome": primary, "is_error": int(official_f1 == 0),
                "official_f1": official_f1, "gold_count": 0, "top10_gold_hit": "", "pool_gold_hit": "",
                "structured_extraction_flag": 0, "structured_flag_reason": "",
                "cause_candidate": "should_abstain_but_answered" if official_f1 == 0 else "",
                "question": row["question"], "gold_answer": "Not mentioned in the conversation",
                "prediction": row["prediction_for_eval"], "raw_prediction": row["raw_prediction"],
                "gold_memory_ids": "", "top10_memory_ids": row["top10_memory_ids"], "diagnostic_pool_ids": "",
            })
            continue

        gold_info = gold_rows.get(query_id)
        gold_ids = set(parse_ids(gold_info.get("gold_memory_ids", "") if gold_info else ""))
        if not gold_ids:
            primary = "Excluded_empty_gold"
            is_error = 0
        else:
            method_top10 = top10.get((method, query_id), parse_ids(row["top10_memory_ids"]))
            method_pool = pools.get((method, query_id), method_top10)
            top_hit, pool_hit = hit(method_top10, gold_ids), hit(method_pool, gold_ids)
            if official_f1 > 0:
                primary, is_error = "Correct_or_partial_answer", 0
            elif not pool_hit:
                primary, is_error = "Type1_Retrieval_miss", 1
            elif not top_hit:
                primary, is_error = "Type2_Ranking_error", 1
            else:
                primary, is_error = "Type4_Reader_failure", 1

        method_top10 = top10.get((method, query_id), parse_ids(row["top10_memory_ids"]))
        method_pool = pools.get((method, query_id), method_top10)
        dense_pool_hit = hit(pools.get(("Dense-bge", query_id), []), gold_ids) if gold_ids else False
        bm25_pool_hit = hit(pools.get(("BM25", query_id), []), gold_ids) if gold_ids else False
        raw_top_hit = hit(top10.get(("ZScore-Raw", query_id), []), gold_ids) if gold_ids else False
        cass_top_hit = hit(top10.get(("ZScore-RawERK", query_id), []), gold_ids) if gold_ids else False
        gold_features = [feature_rows.get(memory_id, {}) for memory_id in gold_ids]
        structured_covered = sum(bool((f.get("entities", "") + f.get("relations", "") + f.get("keywords", "")).strip()) for f in gold_features)
        structured_reasons = []
        if gold_ids and structured_covered == 0:
            structured_reasons.append("all_gold_memories_have_empty_ERK")
        if method == "ZScore-RawERK" and raw_top_hit and not cass_top_hit:
            structured_reasons.append("Raw_hits_Top10_but_RawERK_misses")
        cause = ""
        if primary == "Type1_Retrieval_miss":
            if not dense_pool_hit and bm25_pool_hit:
                cause = "embedding_mismatch_candidate"
            elif dense_pool_hit and not bm25_pool_hit:
                cause = "keyword_or_lexical_miss_candidate"
            elif not dense_pool_hit and not bm25_pool_hit:
                cause = "dual_branch_retrieval_miss"
            else:
                cause = "method_specific_candidate_pruning"
        elif primary == "Type2_Ranking_error":
            cause = "fusion_or_ranking_error"
        elif primary == "Type4_Reader_failure":
            cause = "gold_in_top10_but_answer_score_zero"

        ledger.append({
            "method": method, "setting": SETTING, "qa_id": query_id, "category": category,
            "primary_outcome": primary, "is_error": is_error, "official_f1": official_f1,
            "gold_count": len(gold_ids), "top10_gold_hit": int(hit(method_top10, gold_ids)) if gold_ids else "",
            "pool_gold_hit": int(hit(method_pool, gold_ids)) if gold_ids else "",
            "dense_pool_gold_hit": int(dense_pool_hit) if gold_ids else "",
            "bm25_pool_gold_hit": int(bm25_pool_hit) if gold_ids else "",
            "raw_top10_gold_hit": int(raw_top_hit) if gold_ids else "",
            "cassmem_top10_gold_hit": int(cass_top_hit) if gold_ids else "",
            "gold_structured_covered_count": structured_covered,
            "structured_extraction_flag": int(bool(structured_reasons)),
            "structured_flag_reason": ";".join(structured_reasons), "cause_candidate": cause,
            "question": row["question"], "gold_answer": row["gold_answer"],
            "prediction": row["prediction_for_eval"], "raw_prediction": row["raw_prediction"],
            "gold_memory_ids": ";".join(sorted(gold_ids)), "top10_memory_ids": ";".join(method_top10),
            "diagnostic_pool_ids": ";".join(method_pool),
        })

    if len(ledger) != 6 * 1986:
        raise RuntimeError("Incomplete error ledger")

    distribution: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in ledger if row["method"] == method]
        eligible = [row for row in selected if row["primary_outcome"] != "Excluded_empty_gold"]
        counts = Counter(row["primary_outcome"] for row in eligible)
        for outcome, count in sorted(counts.items()):
            distribution.append({
                "method": method, "setting": SETTING, "outcome": outcome,
                "count": count, "denominator": len(eligible), "rate": count / len(eligible),
            })
        flagged = sum(int(row["structured_extraction_flag"]) for row in eligible)
        distribution.append({
            "method": method, "setting": SETTING, "outcome": "Type3_structured_extraction_audit_flag",
            "count": flagged, "denominator": len(eligible), "rate": flagged / len(eligible),
        })

    # Deterministic 100-case stratified review set: 20 candidates for each requested type.
    rng = random.Random(SEED)
    sample: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    specs = (
        ("Type1_Retrieval_miss", lambda r: r["primary_outcome"] == "Type1_Retrieval_miss"),
        ("Type2_Ranking_error", lambda r: r["primary_outcome"] == "Type2_Ranking_error"),
        ("Type3_Structured_extraction_error_audit", lambda r: int(r["structured_extraction_flag"]) == 1),
        ("Type4_Reader_failure", lambda r: r["primary_outcome"] == "Type4_Reader_failure"),
        ("Type5_Cat5_leakage", lambda r: r["primary_outcome"] == "Type5_Cat5_leakage"),
    )
    for review_type, predicate in specs:
        candidates = [row for row in ledger if predicate(row) and (row["method"], row["qa_id"]) not in used]
        candidates.sort(key=lambda r: (0 if r["method"] == "ZScore-RawERK" else 1, r["method"], r["qa_id"]))
        cassmem = [row for row in candidates if row["method"] == "ZScore-RawERK"]
        others = [row for row in candidates if row["method"] != "ZScore-RawERK"]
        rng.shuffle(cassmem); rng.shuffle(others)
        chosen = (cassmem[:10] + others[:10])[:20]
        if len(chosen) < 20:
            remainder = [row for row in cassmem[10:] + others[10:] if row not in chosen]
            chosen.extend(remainder[:20 - len(chosen)])
        for row in chosen:
            used.add((row["method"], row["qa_id"]))
            gold_text = " || ".join(memory_rows.get(mid, {}).get("text", "") for mid in parse_ids(row["gold_memory_ids"]))
            top_text = " || ".join(memory_rows.get(mid, {}).get("text", "") for mid in parse_ids(row["top10_memory_ids"])[:10])
            sample.append({
                "review_type": review_type, "method": row["method"], "qa_id": row["qa_id"],
                "category": row["category"], "question": row["question"], "gold_answer": row["gold_answer"],
                "prediction": row["prediction"], "primary_outcome": row["primary_outcome"],
                "cause_candidate": row["cause_candidate"], "structured_flag_reason": row["structured_flag_reason"],
                "gold_memory_ids": row["gold_memory_ids"], "gold_memory_text": gold_text,
                "top10_memory_ids": row["top10_memory_ids"], "top10_memory_text": top_text,
                "review_status": "pending", "confirmed_error_type": "", "reviewer_notes": "",
            })
    if len(sample) != 100 or Counter(row["review_type"] for row in sample) != Counter({name: 20 for name, _ in specs}):
        raise RuntimeError(f"Could not form balanced 100-case sample: {Counter(row['review_type'] for row in sample)}")

    OUT.mkdir(parents=True, exist_ok=True)
    ledger_fields = [
        "method", "setting", "qa_id", "category", "primary_outcome", "is_error", "official_f1",
        "gold_count", "top10_gold_hit", "pool_gold_hit", "dense_pool_gold_hit", "bm25_pool_gold_hit",
        "raw_top10_gold_hit", "cassmem_top10_gold_hit", "gold_structured_covered_count",
        "structured_extraction_flag", "structured_flag_reason", "cause_candidate", "question", "gold_answer",
        "prediction", "raw_prediction", "gold_memory_ids", "top10_memory_ids", "diagnostic_pool_ids",
    ]
    write_csv(OUT / "error_ledger.csv", ledger, ledger_fields)
    write_csv(OUT / "error_distribution.csv", distribution, ["method", "setting", "outcome", "count", "denominator", "rate"])
    write_csv(OUT / "manual_review_sample_100.csv", sample, list(sample[0]))

    cass_counts = Counter(row["primary_outcome"] for row in ledger if row["method"] == "ZScore-RawERK" and row["primary_outcome"] != "Excluded_empty_gold")
    lines = [
        "# CassMem Error Analysis V1", "", "## Definition", "",
        "The primary funnel is mutually exclusive. Type 3 is an audit flag rather than an automatically asserted cause.", "",
        "| CassMem outcome | Count |", "|---|---:|",
    ]
    for key, value in sorted(cass_counts.items()):
        lines.append(f"| {key} | {value} |")
    lines += [
        "", "## Interpretation", "",
        "- Type 1: no gold memory enters the method's Top-100 or fusion input pool.",
        "- Type 2: a gold memory is in the diagnostic pool but not Top-10.",
        "- Type 3: missing ERK coverage or Raw-hit/RawERK-miss; requires manual confirmation.",
        "- Type 4: gold memory is in Top-10 but LoCoMo answer score is zero.",
        "- Type 5: Cat5 should abstain but the normalized prediction does not.",
        "", "The balanced `manual_review_sample_100.csv` contains 20 candidates per requested type.",
    ]
    report_path = OUT / "ERROR_ANALYSIS_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "version": "error-analysis-v1", "setting": SETTING, "api_calls": 0,
        "ledger_rows": len(ledger), "manual_sample_rows": len(sample), "seed": SEED,
        "classification": "mutually exclusive primary funnel plus non-causal structured extraction audit flag",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in (METRICS, POOLS, TOP10, GOLD, MEMORIES, FEATURES)],
        "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in (OUT / "error_ledger.csv", OUT / "error_distribution.csv", OUT / "manual_review_sample_100.csv", report_path)],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "ledger": len(ledger), "sample": len(sample), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
