"""Aggregate controlled 100K graph-aware recovery runs into paper artifacts."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cassmem.serving.environment import ROOT


BASE = ROOT / "results" / "serving_100k" / "locomo_workload_graph_v2_100k" / "recovery_v2"
RUNS = BASE / "runs"
CELLS = ("cassandra-base", "cassandra-materialized", "neo4j-native", "neo4j-materialized")
METRICS = (
    "worker_downtime_ms", "update_backlog_peak", "restart_to_first_replay_ms",
    "restart_to_first_top10_ms", "backlog_drain_ms", "time_to_slo_recovery_ms",
    "pre_failure_pipeline_p95_ms", "post_restart_pipeline_p95_ms", "final_hit_at_10",
    "timeout_rate_5000ms", "read_error_rate", "update_error_rate", "missed_update_rate",
    "duplicate_update_rate",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summaries = []
    missing = []
    for cell in CELLS:
        for repetition in range(3):
            path = RUNS / f"{cell}_r{repetition}_summary.json"
            if not path.exists():
                missing.append(str(path))
            else:
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        raise RuntimeError(f"Missing formal runs: {missing}")

    probe_rows: dict[str, dict[str, dict]] = {}
    for cell in CELLS:
        path = RUNS / f"{cell}_post_recovery_probes.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            probe_rows[cell] = {row["qa_id"]: row for row in csv.DictReader(handle)}
    reference = probe_rows[CELLS[0]]
    mismatches = []
    for cell in CELLS[1:]:
        for qa_id in sorted(set(reference) | set(probe_rows[cell])):
            left = reference.get(qa_id)
            right = probe_rows[cell].get(qa_id)
            if left is None or right is None or left["top10_digest"] != right["top10_digest"]:
                mismatches.append({
                    "reference_cell": CELLS[0], "cell": cell, "qa_id": qa_id,
                    "reference_digest": "" if left is None else left["top10_digest"],
                    "cell_digest": "" if right is None else right["top10_digest"],
                })

    publication = []
    for cell in CELLS:
        rows = [row for row in summaries if row["cell"] == cell]
        item = {"cell": cell, "repetitions": len(rows), "run_status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            item[metric] = float(np.median(np.asarray(values, dtype=np.float64)))
        item["partial_commit_replays"] = sum(int(row["partial_commit_replays"]) for row in rows)
        item["post_recovery_top10_parity"] = "PASS" if not any(row["cell"] == cell for row in mismatches) else "FAIL"
        item["overall_status"] = "PASS" if item["run_status"] == "PASS" and item["post_recovery_top10_parity"] == "PASS" else "FAIL"
        publication.append(item)

    with (BASE / "recovery_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summaries)
    with (BASE / "recovery_publication_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(publication[0]))
        writer.writeheader(); writer.writerows(publication)
    with (BASE / "post_recovery_top10_mismatches.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("reference_cell", "cell", "qa_id", "reference_digest", "cell_digest")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(mismatches)

    lines = [
        "# Controlled Recovery under the 100K Graph-Aware Workload",
        "",
        "## Protocol",
        "",
        "- Four cells use the same 100K LoCoMo-shaped graph-aware logical-event contract.",
        "- c=32; 95% reads / 5% updates; 20 total arrivals/s; three repetitions.",
        "- Failure: the update/materializer worker stops after a raw-memory commit and before structured/index visibility; arrivals continue for a 10-second outage; a new worker replays and drains the backlog.",
        "- The database remains online. This is controlled application-pipeline recovery, not multi-node database failover.",
        "- Recovery SLO: five consecutive completed updates at pipeline latency <= 2,000 ms, with zero update error.",
        "",
        "## Main results",
        "",
        "| Cell | Backlog peak | First replay (ms) | First Top-10 hit (ms) | Drain (ms) | Restore SLO (ms) | Post-restart p95 (ms) | Timeout | Missed | Duplicate | Top-10 parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in publication:
        lines.append(
            f"| {row['cell']} | {row['update_backlog_peak']:.0f} | {row['restart_to_first_replay_ms']:.1f} | "
            f"{row['restart_to_first_top10_ms']:.1f} | {row['backlog_drain_ms']:.1f} | "
            f"{row['time_to_slo_recovery_ms']:.1f} | {row['post_restart_pipeline_p95_ms']:.1f} | "
            f"{row['timeout_rate_5000ms']:.1%} | {row['missed_update_rate']:.1%} | "
            f"{row['duplicate_update_rate']:.1%} | {row['post_recovery_top10_parity']} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- These results support recovery of the CassMem application pipeline under a controlled worker failure.",
        "- They do not establish Cassandra or Neo4j multi-node availability, failover, or distributed fault tolerance.",
        f"- Post-recovery Top-10 mismatches across cells: {len(mismatches)}.",
    ]
    report = BASE / "RECOVERY_V2_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status = "PASS" if all(row["overall_status"] == "PASS" for row in publication) and not mismatches else "FAIL"
    outputs = ("recovery_runs.csv", "recovery_publication_table.csv", "post_recovery_top10_mismatches.csv", "RECOVERY_V2_REPORT.md")
    manifest = {
        "status": status,
        "protocol_id": "graph-aware-100k-controlled-recovery-v2",
        "formal_runs": len(summaries),
        "cells": list(CELLS),
        "repetitions": 3,
        "post_recovery_top10_mismatches": len(mismatches),
        "outputs": {name: sha(BASE / name) for name in outputs},
    }
    (BASE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
