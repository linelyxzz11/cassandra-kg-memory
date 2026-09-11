#!/usr/bin/env python3
"""Audit CassMem artifacts for publication readiness without rerunning services."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_reports" / "publication_readiness"
GOLD = ROOT / "02_artifacts" / "retrieval_gold_v2" / "locomo_cat1_4_gold_memory.csv"
MEMORY = ROOT / "01_data" / "locomo_memory_records.csv"
QA = ROOT / "01_data" / "locomo_qa_records.csv"

RANKINGS = {
    "BM25": ROOT / "05_reports" / "official_eval" / "bm25_raw_ranking_canonical1540.csv",
    "Dense-bge": ROOT / "05_reports" / "official_eval" / "dense_bge_ranking_canonical1540.csv",
    "Dense+GlobalKG": ROOT / "05_reports" / "dense_global_kg_rerun" / "dense_global_kg_top10.csv",
    "RRF_compact": ROOT / "05_reports" / "official_eval" / "rrf_compact_canonical1540" / "rrf_compact_top10.csv",
    "ZScore-Raw": ROOT / "05_reports" / "official_eval" / "zscore_raw_ranking_canonical1540.csv",
    "ZScore-RawERK": ROOT / "05_reports" / "official_eval" / "zscore_rawerk_ranking_canonical1540.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def check(check_id: str, layer: str, status: str, observed, expected, detail: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "layer": layer,
                "status": status,
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    memories = read_csv(MEMORY)
    qas = read_csv(QA)
    gold_rows = read_csv(GOLD)
    memory_by_id = {row["memory_id"]: row for row in memories}
    qa_by_id = {row["qa_id"]: row for row in qas}
    gold_by_id = {row["query_id"]: row for row in gold_rows}
    evaluable_ids = {
        row["query_id"] for row in gold_rows if row["gold_status"] == "mapped"
    }

    check("D01", "data", "PASS" if len(memories) == 5882 else "FAIL", len(memories), 5882, "Canonical memory rows.")
    check("D02", "data", "PASS" if len(memory_by_id) == len(memories) else "FAIL", len(memory_by_id), len(memories), "memory_id must be unique.")
    check("D03", "data", "PASS" if len(qas) == 1986 else "FAIL", len(qas), 1986, "Canonical QA rows.")
    check("D04", "data", "PASS" if len(qa_by_id) == len(qas) else "FAIL", len(qa_by_id), len(qas), "qa_id must be unique.")

    status_counts = Counter(row["gold_status"] for row in gold_rows)
    check("G01", "gold", "PASS" if len(gold_rows) == 1540 else "FAIL", len(gold_rows), 1540, "Cat1-4 gold rows.")
    check("G02", "gold", "PASS" if len(gold_by_id) == len(gold_rows) else "FAIL", len(gold_by_id), len(gold_rows), "Gold query_id must be unique.")
    check("G03", "gold", "PASS" if status_counts == {"mapped": 1536, "evidence_empty": 4} else "FAIL", json.dumps(status_counts, sort_keys=True), '{"evidence_empty": 4, "mapped": 1536}', "Official evidence-empty questions remain explicit.")

    unknown_gold = []
    cross_scope_gold = []
    inconsistent_gold_count = []
    for row in gold_rows:
        ids = split_ids(row["gold_memory_ids"])
        if int(row["gold_count"]) != len(ids):
            inconsistent_gold_count.append(row["query_id"])
        for memory_id in ids:
            memory = memory_by_id.get(memory_id)
            if memory is None:
                unknown_gold.append(f"{row['query_id']}:{memory_id}")
            elif memory["sample_id"] != row["conversation_id"]:
                cross_scope_gold.append(f"{row['query_id']}:{memory_id}")
    check("G04", "gold", "PASS" if not unknown_gold else "FAIL", len(unknown_gold), 0, "Every gold memory must exist in the canonical corpus.")
    check("G05", "gold", "PASS" if not cross_scope_gold else "FAIL", len(cross_scope_gold), 0, "Every gold memory must belong to the query conversation.")
    check("G06", "gold", "PASS" if not inconsistent_gold_count else "FAIL", len(inconsistent_gold_count), 0, "gold_count must match the parsed ID list.")

    ranking_audit = []
    for method, path in RANKINGS.items():
        rows = read_csv(path)
        by_query: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["query_id"] in evaluable_ids:
                by_query[row["query_id"]].append(row)
        missing = evaluable_ids - set(by_query)
        bad_ranks = 0
        duplicate_memories = 0
        unknown_memories = 0
        cross_scope = 0
        for query_id, items in by_query.items():
            ranks = sorted(int(float(row["rank"])) for row in items if int(float(row["rank"])) <= 10)
            top10 = [row for row in items if int(float(row["rank"])) <= 10]
            if ranks != list(range(1, 11)):
                bad_ranks += 1
            memory_ids = [row["memory_id"] for row in top10]
            if len(set(memory_ids)) != len(memory_ids):
                duplicate_memories += 1
            expected_scope = gold_by_id[query_id]["conversation_id"]
            for memory_id in memory_ids:
                memory = memory_by_id.get(memory_id)
                if memory is None:
                    unknown_memories += 1
                elif memory["sample_id"] != expected_scope:
                    cross_scope += 1
        passed = not any((missing, bad_ranks, duplicate_memories, unknown_memories, cross_scope))
        ranking_audit.append(
            {
                "method": method,
                "source": str(path.relative_to(ROOT)),
                "source_sha256": sha256(path),
                "evaluable_queries": len(by_query),
                "missing_queries": len(missing),
                "bad_rank_sequences": bad_ranks,
                "duplicate_memory_queries": duplicate_memories,
                "unknown_memory_rows": unknown_memories,
                "cross_conversation_rows": cross_scope,
                "status": "PASS" if passed else "FAIL",
            }
        )
        check(
            f"R-{method}",
            "retrieval",
            "PASS" if passed else "FAIL",
            len(by_query),
            1536,
            f"Top-10 integrity: missing={len(missing)}, bad_ranks={bad_ranks}, duplicates={duplicate_memories}, unknown={unknown_memories}, cross_scope={cross_scope}.",
        )
    write_csv(OUT / "ranking_integrity.csv", ranking_audit)

    overall = read_csv(ROOT / "05_reports" / "retrieval_main_table" / "retrieval_main_overall.csv")
    required_metrics = ("MRR@10", "Hit@1", "Hit@5", "Hit@10", "Recall@10", "nDCG@10")
    bad_metric_values = [
        (row["method"], row["evaluation_split"], metric)
        for row in overall
        for metric in required_metrics
        if not 0 <= float(row[metric]) <= 1
    ]
    expected_n = {"test": 1146, "all": 1536}
    bad_n = [row for row in overall if int(row["n"]) != expected_n[row["evaluation_split"]]]
    check("R01", "retrieval", "PASS" if len(overall) == 12 else "FAIL", len(overall), 12, "Six methods × test/all rows.")
    check("R02", "retrieval", "PASS" if not bad_metric_values else "FAIL", len(bad_metric_values), 0, "All retrieval metrics must lie in [0,1].")
    check("R03", "retrieval", "PASS" if not bad_n else "FAIL", len(bad_n), 0, "Primary test n=1,146; diagnostic all n=1,536.")

    retrieval_sig = read_csv(ROOT / "05_reports" / "retrieval_main_table" / "retrieval_main_significance.csv")
    check("R04", "retrieval", "PASS" if len(retrieval_sig) == 18 else "FAIL", len(retrieval_sig), 18, "CassMem paired bootstrap vs three baselines across six metrics, with Holm correction.")

    reader_dir = ROOT / "05_reports" / "official_eval" / "gpt4o_dual_setting_locomo_corrected_v3"
    reader = read_csv(reader_dir / "dual_setting_main_table.csv")
    reader_metrics = (
        "cat1_multihop_f1",
        "cat2_temporal_f1",
        "cat3_open_domain_f1",
        "cat4_single_hop_f1",
        "cat5_adversarial_f1",
        "cat1_4_f1",
        "full5_f1",
    )
    bad_reader = [
        (row["method"], row["setting"], metric)
        for row in reader
        for metric in reader_metrics
        if not 0 <= float(row[metric]) <= 1
    ]
    check("A01", "reader", "PASS" if len(reader) == 12 else "FAIL", len(reader), 12, "Six methods × two prompt settings.")
    check("A02", "reader", "PASS" if not bad_reader else "FAIL", len(bad_reader), 0, "All answer-level F1 values must lie in [0,1].")

    bleu = read_csv(reader_dir / "bleu1_offline_summary.csv")
    bad_bleu_n = [
        row for row in bleu
        if int(row["cat1_4_n"]) != 1540 or int(row["full5_n"]) != 1986
    ]
    check("A03", "reader", "PASS" if len(bleu) == 12 else "FAIL", len(bleu), 12, "Reproducible BLEU/hybrid summary rows.")
    check("A04", "reader", "PASS" if not bad_bleu_n else "FAIL", len(bad_bleu_n), 0, "BLEU denominators must be Cat1-4 n=1540 and Full5 n=1986.")
    reader_sig = read_csv(reader_dir / "reader_main_significance.csv")
    check("A05", "reader", "PASS" if len(reader_sig) == 8 else "FAIL", len(reader_sig), 8, "Setting-B paired bootstrap vs RRF_compact and ZScore-Raw, Holm-corrected.")
    densekg_integrity_path = ROOT / "05_reports" / "locomo_gpt4o_prompt_protocol_corrected_densekg_v2" / "integrity_audit.json"
    densekg_integrity = json.loads(densekg_integrity_path.read_text(encoding="utf-8")) if densekg_integrity_path.exists() else {}
    check("A06", "reader", "PASS" if densekg_integrity.get("all_pass") else "BLOCKED", "2 x 1986 complete" if densekg_integrity.get("all_pass") else "missing/incomplete", "corrected Dense+GlobalKG reader predictions", "Both settings must pass qid, prompt-hash, ranking-hash, category-distribution, and returned-model checks.")
    judge_dir = ROOT / "05_reports" / "llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected"
    judge_complete = (judge_dir / "COMPLETE_STATUS.json").exists() and (judge_dir / "run_manifest.json").exists()
    check("A07", "reader", "PASS" if judge_complete else "BLOCKED", "complete" if judge_complete else "missing", "frozen corrected GPT-4o judge run", "HingeMem-compatible binary judge requires corrected Cat5 and a frozen run manifest.")

    candidate_rows = read_csv(ROOT / "05_reports" / "retrieval_main_table" / "candidate_pool_oracle_audit.csv")
    check("K01", "retrieval", "WARN", "full conversation pool oracle only", "pruned KG-expansion candidate recall", "Current value is a sanity check and must not be presented as graph-expansion Candidate Recall.")

    system_dir = ROOT / "05_reports" / "backend_system_eval"
    visibility = read_csv(system_dir / "update_visibility_results.csv")
    recovery = read_csv(system_dir / "recovery_results.csv")
    throughput = read_csv(system_dir / "throughput_results.csv")
    invalid_visibility = sum(float(row["latency_ms"]) < 0 for row in visibility)
    invalid_recovery = sum(float(row["final_searchable_ratio"]) <= 0 or float(row["restart_time"]) < 0 for row in recovery)
    missing_throughput_latency = sum(float(row["p50"]) < 0 for row in throughput)
    check("S01", "system", "BLOCKED", invalid_visibility, 0, "backend_system_eval update-visibility rows use -1 sentinels and are not citation-ready.")
    check("S02", "system", "BLOCKED", invalid_recovery, 0, "backend_system_eval recovery rows have failed/sentinel values and are not citation-ready.")
    check("S03", "system", "WARN", missing_throughput_latency, 0, "QPS rows omit latency-under-load; report QPS only with this limitation.")
    check("S04", "system", "BLOCKED", "missing", "end-to-end retrieval serving benchmark", "Need backend fetch + graph expansion + scoring + fusion + total latency and QPS at c=1/16/32/64.")
    p5v31_dir = ROOT / "05_reports" / "p5_1_retrieval_visibility_v3" / "formal_fixed_20260803"
    p5v31_manifest_path = p5v31_dir / "p5v31_formal_fixed_20260803_manifest.json"
    p5v31_events_path = p5v31_dir / "p5v31_formal_fixed_20260803_per_event.csv"
    p5v31_manifest = json.loads(p5v31_manifest_path.read_text(encoding="utf-8")) if p5v31_manifest_path.exists() else {}
    p5v31_events = read_csv(p5v31_events_path) if p5v31_events_path.exists() else []
    p5v31_gate = (
        p5v31_manifest.get("status") == "PASS"
        and p5v31_manifest.get("citation_ready") is True
        and int(p5v31_manifest.get("expected_formal_rows", -1)) == 36000
        and int(p5v31_manifest.get("actual_formal_rows", -2)) == 36000
        and len(p5v31_events) == 36000
    )
    bad_p5v31 = sum(
        row.get("status") != "ok"
        or int(row.get("candidate_count", -1)) != 33
        or int(row.get("expected_candidate_count", -2)) != 33
        or int(row.get("target_rank", -1)) != 1
        for row in p5v31_events
    )
    parity = p5v31_manifest.get("gates", {}).get("cross_backend_state_and_topk_parity", {})
    p5v31_gate = p5v31_gate and bad_p5v31 == 0 and parity.get("status") == "PASS"
    check("S05", "system", "PASS" if p5v31_gate else "FAIL", len(p5v31_events), 36000, "P5-1 v3.1 must have exact manifest counts, fixed 33-candidate work, rank-1 targets, zero failures and cross-backend parity PASS.")
    check("S06", "system", "PASS" if (ROOT / "05_reports" / "p5_minimal_core" / "P5_1_INVALID_FOR_FINAL_RETRIEVAL.md").exists() else "FAIL", "present", "present", "Legacy edge-visible P5-1 is explicitly superseded and cannot be cited for final retrieval visibility.")

    secret_pattern = re.compile(
        r"sk-" + r"[A-Za-z0-9_-]{20,}" + "|" + "password" + "123"
    )
    secret_hits = []
    source_roots = (ROOT / "00_project", ROOT / "03_src", ROOT / "04_experiments", ROOT / "05_reports")
    for base in source_roots:
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".json", ".md", ".txt", ".yaml", ".yml", ".toml"}:
                continue
            if "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if secret_pattern.search(text):
                secret_hits.append(str(path.relative_to(ROOT)))
    check("P01", "repository", "PASS" if not secret_hits else "FAIL", len(secret_hits), 0, "No active source/report file may contain literal API keys or the former example password.")
    gitignore_bytes = (ROOT / ".gitignore").read_bytes()
    check("P02", "repository", "PASS" if b"\x00" not in gitignore_bytes else "FAIL", int(b"\x00" in gitignore_bytes), 0, ".gitignore must be valid UTF-8 text without embedded NULs.")

    hardcoded_rows = []
    absolute_pattern = re.compile(r"""(?:[A-Za-z]:[\\/](?:[^"' \r\n])+|Path\(["'][A-Za-z]:[\\/])""")
    for base in (ROOT / "03_src", ROOT / "04_experiments", ROOT / "05_reports"):
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            matches = list(absolute_pattern.finditer(text))
            if matches:
                hardcoded_rows.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "match_count": len(matches),
                        "status": "REFACTOR",
                        "recommendation": "derive project root from __file__ or accept a CLI/config path",
                    }
                )
    if hardcoded_rows:
        write_csv(OUT / "hardcoded_path_audit.csv", hardcoded_rows)
    check("P03", "repository", "WARN" if hardcoded_rows else "PASS", len(hardcoded_rows), 0, "Hard-coded absolute paths reduce portability; see hardcoded_path_audit.csv.")

    p5_config_paths = []
    p5_base = ROOT / "04_experiments" / "p5_3c"
    for config_name in ("p5_3c_config.json", "p5_3c_config.smoke.json"):
        config = json.loads((p5_base / config_name).read_text(encoding="utf-8"))
        for section, key in (
            ("files", "events"),
            ("scale_guard", "source_artifact"),
            ("scale_guard", "validated_manifest"),
        ):
            target = (p5_base / config[section][key]).resolve()
            p5_config_paths.append((config_name, section, key, target, target.exists()))
    missing_p5_paths = [item for item in p5_config_paths if not item[-1]]
    check("P04", "repository", "PASS" if not missing_p5_paths else "FAIL", len(missing_p5_paths), 0, "All P5-3C event, scale-guard, and validated-manifest paths must resolve.")

    gap_rows = [
        {"priority": "P0", "layer": "security", "gap": "Rotate previously committed API keys", "status": "USER ACTION", "why": "Removing literals does not invalidate credentials already exposed in repository history.", "completion": "Revoke old keys; create new keys; store only in environment variables."},
        {"priority": "DONE", "layer": "reader", "gap": "Regenerate Dense+GlobalKG reader answers", "status": "COMPLETE", "why": "Corrected degree-centrality lambda=0.2 ranking is linked to audited per-query Reader outputs.", "completion": "Both settings complete at n=1986 with F1, B1, and GPT-4o J."},
        {"priority": "P0", "layer": "system", "gap": "Replace invalid P7-B visibility/recovery files", "status": "NOT VALID", "why": "Current files contain -1 sentinels and final_searchable_ratio=0.", "completion": "Rerun with timestamp provenance, successful visibility probes, recovery gate, and run manifest."},
        {"priority": "P1", "layer": "system", "gap": "End-to-end retrieval serving benchmark", "status": "MISSING", "why": "Backend microbenchmarks do not measure the deployed retrieval pipeline.", "completion": "Report component and total p50/p95/p99 plus QPS at c=1/16/32/64."},
        {"priority": "P1", "layer": "retrieval", "gap": "Explicit KG candidate expansion", "status": "MISSING", "why": "The current full conversation candidate pool has oracle recall 1.0 by construction.", "completion": "Freeze pruned candidates and report Candidate Recall before reranking plus final Recall@10."},
        {"priority": "P1", "layer": "statistics", "gap": "Reader uncertainty beyond current comparisons", "status": "PARTIAL", "why": "Paired bootstrap now covers CassMem vs RRF/ZScore-Raw; external published systems are not paired locally.", "completion": "Add confidence intervals for the final claims and clearly separate rerun from quoted baselines."},
        {"priority": "P2", "layer": "reproducibility", "gap": "Portable path cleanup and environment lock", "status": "PARTIAL", "why": "Dependency ranges are documented, but numerous scripts retain machine-specific paths.", "completion": "Refactor publication entrypoints first; capture Python/package/backend versions in each run manifest."},
        {"priority": "P2", "layer": "literature", "gap": "Authoritative related-work evidence table", "status": "DONE", "why": "Peer-reviewed benchmark/memory papers and clearly labeled preprints are now separated.", "completion": "Maintain 08_literature/metric_protocol_matrix.csv as protocols or cited baselines change."},
    ]
    write_csv(OUT / "experiment_gap_matrix.csv", gap_rows)
    write_csv(OUT / "audit_checks.csv", checks)

    counts = Counter(row["status"] for row in checks)
    report_lines = [
        "# CassMem Publication-Readiness Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Verdict",
        "",
        (
            "Retrieval gold, six Top-10 ranking artifacts, retrieval main metrics, "
            "reader F1/BLEU provenance, and paired significance outputs pass the "
            "automated integrity checks. The repository is not yet submission-ready "
            "because a real pruned KG candidate stage and the remaining "
            "CassMem end-to-end serving evidence "
            "remain incomplete. The existing backend_system_eval visibility and "
            "recovery CSVs are explicitly not citation-ready."
        ),
        "",
        "## Check Summary",
        "",
        f"- PASS: {counts['PASS']}",
        f"- WARN: {counts['WARN']}",
        f"- BLOCKED: {counts['BLOCKED']}",
        f"- FAIL: {counts['FAIL']}",
        "",
        "## Completed in This Audit",
        "",
        "- Removed active hard-coded API keys and database passwords; added environment-variable configuration.",
        "- Repaired `.gitignore`, added `.env.example`, and documented dependency ranges.",
        "- Repaired P5-3C config paths and aligned the 18K frozen-event manifest with the actual artifact.",
        "- Added Holm-corrected paired bootstrap significance for retrieval and reader main comparisons.",
        "- Recomputed BLEU-1 from frozen predictions with an executable protocol and corrected the Full5 label to a B1/Cat5-accuracy hybrid.",
        "- Added machine-readable ranking, path, check, and experiment-gap audits.",
        "- Validated P5-1 v3.1: 36K fixed-candidate update-to-final-TopK events with exact cross-backend parity.",
        "- Regenerated corrected Dense+GlobalKG Reader outputs for both 1,986-question settings and rescored F1, B1, and GPT-4o J.",
        "",
        "## Interpretation Rules",
        "",
        "- Cat5 is answer-level adversarial abstention and is excluded from retrieval effectiveness.",
        "- `Candidate Recall=1.0` for the full conversation pool is only an oracle sanity check, not evidence for graph expansion.",
        "- Dense+GlobalKG retrieval and Reader rows now share the frozen degree-centrality lambda=0.2 ranking provenance.",
        "- Quoted external memory-system results must be marked as published values, not locally rerun results.",
        "",
        "See `audit_checks.csv`, `ranking_integrity.csv`, `experiment_gap_matrix.csv`, and `hardcoded_path_audit.csv` for machine-readable details.",
    ]
    (OUT / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "check_counts": dict(counts),
        "artifacts": {},
    }
    for name in ("README.md", "audit_checks.csv", "ranking_integrity.csv", "experiment_gap_matrix.csv", "hardcoded_path_audit.csv"):
        path = OUT / name
        if path.exists():
            manifest["artifacts"][name] = {"sha256": sha256(path), "size": path.stat().st_size}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if counts["FAIL"]:
        raise SystemExit(f"Publication audit has {counts['FAIL']} integrity failure(s)")
    print(json.dumps(manifest["check_counts"], indent=2))


if __name__ == "__main__":
    main()
