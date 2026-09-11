#!/usr/bin/env python3
"""Regenerate the canonical artifact manifest from publication-facing roots."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "00_project" / "ARTIFACT_MANIFEST.csv"

SPECS = (
    ("data", "canonical_input", "01_data/locomo_memory_records.csv"),
    ("data", "canonical_input", "01_data/locomo_qa_records.csv"),
    ("data", "canonical_input", "01_data/c3_source_scale_1M.csv"),
    ("data", "frozen_embedding", "01_data/locomo_memory_bge_large.npy"),
    ("data", "frozen_embedding", "01_data/locomo_qa_bge_large.npy"),
    ("retrieval", "gold", "02_artifacts/retrieval_gold_v2/*"),
    ("retrieval", "feature", "02_artifacts/p3_memory_features.csv"),
    ("system", "frozen_event", "02_artifacts/frozen_event_manifest.json"),
    ("system", "frozen_event", "02_artifacts/frozen_update_events.jsonl"),
    ("system", "frozen_event", "02_artifacts/p5_2_*events.jsonl"),
    ("system", "frozen_event", "02_artifacts/p5_3*_events*.jsonl"),
    ("system", "event_manifest", "02_artifacts/p5_3c_event_manifest.json"),
    ("retrieval", "formal_result", "05_reports/dense_global_kg_rerun/*"),
    ("retrieval", "formal_result", "05_reports/p1_compact_component_ablation/**/*"),
    ("retrieval", "formal_result", "05_reports/fusion_ablation_aligned/**/*"),
    ("reader", "preflight", "05_reports/dense_global_kg_rerun/reader_preflight/*"),
    ("retrieval", "formal_result", "05_reports/retrieval_main_table/*"),
    ("retrieval", "analysis", "05_reports/candidate_stage_analysis_v1/**/*"),
    ("reader", "formal_result", "05_reports/official_eval/gpt4o_dual_setting_locomo_corrected_v3/*"),
    ("reader", "formal_result", "05_reports/official_eval/gpt4o_corrected_densekg_v2/**/*"),
    ("reader", "formal_result", "05_reports/locomo_gpt4o_prompt_protocol_corrected_densekg_v2/**/*"),
    ("reader", "formal_result", "05_reports/reader_main_hingemem_style/*"),
    ("reader", "formal_result", "05_reports/llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected/*"),
    ("analysis", "formal_result", "05_reports/question_type_breakdown_v1/**/*"),
    ("analysis", "formal_result", "05_reports/error_analysis_v1/**/*"),
    ("all", "protocol", "05_reports/unified_evaluation_protocol_v1/**/*"),
    ("system", "formal_result", "05_reports/backend_equivalence/*"),
    ("system", "formal_result", "05_reports/backend_equivalence_v2/**/*"),
    ("system", "formal_result", "05_reports/p5_1_retrieval_visibility_v3/formal_fixed_20260803/*"),
    ("system", "formal_gate", "05_reports/locomo_workload_graph_v2_canonical_gate/**/*"),
    ("system", "formal_result", "05_reports/locomo_workload_graph_v2_100k/**/*"),
    ("system", "formal_result", "05_reports/experiment11_online_freshness_v2/**/*"),
    ("system", "status", "05_reports/p5_minimal_core/P5_1_INVALID_FOR_FINAL_RETRIEVAL.md"),
    ("system", "status", "05_reports/backend_system_eval/STATUS.md"),
    ("all", "audit", "05_reports/publication_readiness/*"),
    ("all", "audit", "05_reports/publication_readiness_v2/*"),
    ("literature", "protocol_evidence", "08_literature/*"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    # Keep the manifest aligned with Git's artifact policy. Large local traces
    # may exist beside compact reports, but ignored files are deliberately not
    # publication artifacts and must not appear in this inventory.
    visible = set(
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).splitlines()
    )
    rows = {}
    for layer, role, pattern in SPECS:
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative not in visible:
                continue
            rows[relative] = {
                "layer": layer,
                "artifact_role": role,
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "status": (
                    "NOT_CITATION_READY"
                    if relative.startswith("05_reports/backend_system_eval/")
                    else "CANONICAL"
                ),
            }

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "layer",
                "artifact_role",
                "path",
                "sha256",
                "size_bytes",
                "status",
            ),
        )
        writer.writeheader()
        writer.writerows(rows[path] for path in sorted(rows))
    print(f"Wrote {len(rows)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
