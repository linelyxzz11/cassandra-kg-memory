"""Generate a current claim-to-evidence gap matrix for the paper handoff."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_reports" / "publication_readiness_v2"


ROWS = [
    ("R1", "retrieval", "Canonical gold mapping", "DONE", "core", "02_artifacts/retrieval_gold_v2", "1540 total; 1536 evidence-bearing; four documented empty-gold queries"),
    ("R2", "retrieval", "Six-method main effectiveness", "DONE", "core", "05_reports/retrieval_main_table", "Held-out main table, all-query diagnostic, category metrics, paired Holm-corrected bootstrap"),
    ("R3", "retrieval", "ERK component ablation", "DONE", "core", "05_reports/p1_compact_component_ablation", "E/R/K combinations, leave-one-out, time sensitivity, parity and bootstrap"),
    ("R4", "retrieval", "Fusion strategy and alpha selection", "DONE", "core", "05_reports/fusion_audit", "Raw/RRF/MinMax/ZScore audit and dev-only alpha=0.6 selection"),
    ("R5", "retrieval", "Corrected Dense+GlobalKG", "DONE", "core", "05_reports/dense_global_kg_rerun", "Corrected degree prior, dev lambda selection, held-out result and significance"),
    ("R6", "retrieval", "Actual candidate-stage recall", "DONE_NEW", "core", "05_reports/candidate_stage_analysis_v1", "Actual frozen pool recall and candidate-to-Top10 ranking-loss decomposition"),
    ("R7", "retrieval", "Question-type breakdown", "DONE", "analysis", "05_reports/question_type_breakdown_v1", "Correct LoCoMo mapping; mechanism claim does not overstate Multi-Hop"),
    ("A1", "reader", "Six-method dual-setting F1/B1", "DONE", "core", "05_reports/reader_offline_metrics_v4", "23832 per-query metric rows; corrected Cat5 option restoration"),
    ("A2", "reader", "Corrected Dense+GlobalKG Reader", "DONE", "core", "05_reports/locomo_gpt4o_prompt_protocol_corrected_densekg_v2", "Both settings 1986/1986 with integrity gate"),
    ("A3", "reader", "HingeMem-style F1/J/B1 table", "DONE", "core", "05_reports/reader_main_hingemem_style", "Corrected local rows plus clearly quoted external rows"),
    ("A4", "reader", "GPT-4o Judge", "DONE", "core", "05_reports/llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected", "Six methods, two settings, five categories and micro Overall"),
    ("A5", "reader", "Reader paired uncertainty", "DONE", "statistics", "05_reports/official_eval/gpt4o_dual_setting_locomo_corrected_v3/reader_main_significance.csv", "Paired bootstrap; several Reader gains are not significant and must be stated honestly"),
    ("A6", "analysis", "100-case error analysis", "PARTIAL_HUMAN", "analysis", "05_reports/error_analysis_v1", "100 reviewed; 91 evidence-resolved; nine require independent adjudication"),
    ("P1", "protocol", "Unified evaluation contract", "DONE", "core", "05_reports/unified_evaluation_protocol_v1", "Frozen scopes, models, prompts, metrics, embeddings, Cat5 and cache rules"),
    ("S1", "system", "Six-method backend bridge", "DONE", "core", "05_reports/backend_equivalence_v2", "CSV/Cassandra/Neo4j exact Top10 and metric parity; corrected-edge digest parity"),
    ("S2", "system", "Graph-aware canonical parity", "DONE", "core", "05_reports/locomo_workload_graph_v2_canonical_gate", "5882-memory four-cell semantic and Top10 gate"),
    ("S3", "system", "100K load and equal-work gate", "DONE", "core", "05_reports/locomo_workload_graph_v2_100k/graph_100k_load_gate_summary.json", "100K, 171 namespaces, zero graph/relation parity mismatches"),
    ("S4", "system", "100K storage/write amplification", "DONE_NEW", "core", "05_reports/locomo_workload_graph_v2_100k/storage_work_v2", "Four-cell physical records, writes/event and load times under semantic parity"),
    ("S5", "system", "100K 95:5 concurrency matrix", "DONE", "core", "05_reports/locomo_workload_graph_v2_100k/main_95_5_v2", "c=1/8/16/32/64, three reps, 300K measured operations"),
    ("S6", "system", "Four query types", "DONE", "core", "05_reports/locomo_workload_graph_v2_100k/query_types_v2", "Scope, relation, candidate projection and end-to-end Top10 at c=32"),
    ("S7", "system", "Online Time-to-Top10", "DONE", "core", "05_reports/experiment11_online_freshness_v2", "Commit through real ZScore-RawERK Top10, rates 1/2/5/10, FreshHit deadlines and SLO"),
    ("S8", "system", "Controlled four-cell worker recovery", "DONE", "appendix", "05_reports/locomo_workload_graph_v2_100k/recovery_v2", "Same 100K graph-aware contract; 12/12 gates pass; zero missed/duplicate updates and post-recovery Top10 mismatches; databases remain online"),
    ("G1", "generalization", "Graph-aware 90:10 and 99:1 mix sensitivity", "OPTIONAL_NOT_RUN", "appendix", "", "95:5 is complete; extra mixes improve robustness but are not needed for the frozen main claim"),
    ("G2", "generalization", "Database restart or multi-node failover", "EXTERNAL_BLOCKER", "future", "", "Controlled application-worker recovery is complete but does not test database failover or distributed fault tolerance"),
    ("G3", "generalization", "Independent cross-judge", "OPTIONAL_PAID", "robustness", "05_reports/reader_judge_protocol_v3", "Not needed for HingeMem-compatible main J; useful only as judge-sensitivity appendix"),
    ("G4", "generalization", "Second memory benchmark", "EXTERNAL_BLOCKER", "future", "", "No second compatible frozen corpus/prediction set is present locally"),
    ("G5", "systems", "Multi-node scale-out", "EXTERNAL_BLOCKER", "future", "", "Current formal evidence is single-machine/single-instance; do not claim distributed scale-out"),
    ("G6", "baselines", "Local rerun of external memory systems", "EXTERNAL_BLOCKER", "future", "", "External HingeMem-table values are quoted, not paired local reruns"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment_id, layer, experiment, status, role, evidence, note in ROWS:
        evidence_path = ROOT / evidence if evidence else None
        exists = bool(evidence_path and evidence_path.exists())
        if status.startswith("DONE") and not exists:
            raise FileNotFoundError(f"missing evidence for {experiment_id}: {evidence}")
        rows.append({
            "experiment_id": experiment_id,
            "layer": layer,
            "experiment": experiment,
            "status": status,
            "paper_role": role,
            "evidence": evidence,
            "evidence_exists": int(exists),
            "note": note,
        })
    csv_path = OUT / "experiment_gap_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    done = sum(row["status"].startswith("DONE") for row in rows)
    report = [
        "# CassMem deep experiment audit v2",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Verdict",
        "",
        "The three core axes are now internally complete: retrieval effectiveness, Reader answer quality, and graph-aware backend serving/freshness. "
        "The audit found two locally fillable presentation/evidence gaps and completed both: actual candidate-stage recall/ranking-loss decomposition and the 100K four-cell storage/write-amplification table.",
        "",
        f"- Completed evidence items: {done}/{len(rows)}",
        "- Core blockers remaining in the current local scope: 0",
        "- Human adjudication remaining: nine ambiguous error cases",
        "- External/generalization limits: second dataset, multi-node scale-out, and identical local reruns of quoted memory systems",
        "",
        "## What is ready for the paper",
        "",
        "1. Retrieval main table plus held-out significance, ERK ablation, fusion ablation, question-type analysis, and candidate-stage decomposition.",
        "2. HingeMem-style Reader F1/J/B1 table with corrected Dense+GlobalKG and explicit quoted-vs-local provenance.",
        "3. Six-method CSV/Cassandra/Neo4j backend bridge with exact ranking and input-digest parity.",
        "4. Graph-aware 100K four-cell equal-work/storage table, 95:5 concurrency table, query-type table, actual Time-to-Top10 freshness table, and controlled update/materializer-worker recovery.",
        "",
        "## What must not be claimed",
        "",
        "- Do not call the current candidate stage arbitrary graph expansion; it is conversation-scoped frozen branch pruning.",
        "- Do not claim Cassandra is universally faster: Neo4j is competitive at the light-load point in Experiment 11.",
        "- Do not claim distributed scale-out or multi-node fault tolerance from single-instance Docker experiments.",
        "- Do not claim significant Reader superiority over RRF_compact; the paired intervals do not support it.",
        "- Do not describe the nine ambiguous error cases as independently human-resolved.",
        "",
        "## Remaining optional experiments",
        "",
        "Graph-aware 90:10/99:1 sensitivity remains an appendix-strengthening experiment. Controlled four-cell application-worker recovery is complete. Database restart, multi-node failover, and distributed fault tolerance remain outside the current evidence boundary.",
        "",
        "## Machine-readable matrix",
        "",
        "See `experiment_gap_matrix.csv`.",
        "",
    ]
    report_path = OUT / "DEEP_EXPERIMENT_AUDIT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "status": "PASS_WITH_EXTERNAL_LIMITS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_rows": len(rows),
        "completed_items": done,
        "core_local_blockers": 0,
        "human_adjudication_rows": 9,
        "outputs": [
            {"path": str(csv_path.relative_to(ROOT)), "sha256": sha256(csv_path)},
            {"path": str(report_path.relative_to(ROOT)), "sha256": sha256(report_path)},
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
