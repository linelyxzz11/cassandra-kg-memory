"""Aggregate Experiment 11 into publication-facing stage and freshness tables."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cassmem.serving.environment import ROOT


BASE = ROOT / "results" / "freshness" / "experiment11_online_freshness_v2"
RUNS = BASE / "runs"
CELLS = ("cassandra-base", "cassandra-materialized", "neo4j-native", "neo4j-materialized")
RATES = (1.0, 2.0, 5.0, 10.0)
MAIN_RATE = 5.0
SLO_P95_MS = 2000.0
SLO_DRAIN_MS = 2000.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def med(values) -> float:
    return float(np.median(np.asarray(list(values), dtype=np.float64)))


def pct(values, q) -> float:
    return float(np.percentile(np.asarray(list(values), dtype=np.float64), q))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_hit_conditioned_rate(rows: list[dict[str, str]], deadline_ms: int) -> float:
    """Deadline success among updates included by the post-sync Top-10 query."""
    eligible = [row for row in rows if int(row["final_hit_at_10"]) == 1]
    if not eligible:
        raise RuntimeError("FreshHit@10 is undefined because this scenario has no final Top-10 hits")
    return sum(float(row["pipeline_complete_ms"]) <= deadline_ms for row in eligible) / len(eligible)


def markdown_table(rows: list[dict]) -> str:
    fields = list(rows[0])
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    manifests = []
    all_events = []
    all_scenarios = []
    for cell in CELLS:
        manifest_path = RUNS / f"{cell}_c32_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        if manifest["status"] != "PASS" or manifest["repetitions"] != 3 or tuple(manifest["rates"]) != RATES:
            raise RuntimeError(f"Incomplete formal manifest: {manifest_path}")
        all_events.extend(read_csv(RUNS / f"{cell}_c32_events.csv"))
        all_scenarios.extend(read_csv(RUNS / f"{cell}_c32_scenarios.csv"))

    if len(all_events) != 600 * 4 or len(all_scenarios) != 12 * 4:
        raise RuntimeError(f"Unexpected cardinality: events={len(all_events)}, scenarios={len(all_scenarios)}")
    if any(float(row["error_rate"]) or float(row["missed_update_rate"]) or float(row["duplicate_update_rate"]) for row in all_scenarios):
        raise RuntimeError("Correctness failure in formal scenarios")

    # Table 11A: cumulative arrival-to-stage latency at the 100 ops/s main point.
    stage_fields = (
        ("Raw backend commit", "t_commit_ms"),
        ("Structured view visible", "t_structured_view_visible_ms"),
        ("Sparse+dense index visible", "t_index_visible_ms"),
        ("Pipeline complete", "pipeline_complete_ms"),
        ("First inclusion in Top-10 (final hits)", "time_to_top10_ms"),
    )
    table11a = []
    for cell in CELLS:
        for label, field in stage_fields:
            rep_stats = defaultdict(list)
            for row in all_events:
                if row["cell"] != cell or float(row["update_rate_s"]) != MAIN_RATE or row[field] == "":
                    continue
                rep_stats[int(row["repetition"])].append(float(row[field]))
            if sorted(rep_stats) != [0, 1, 2]:
                raise RuntimeError(f"Missing repetitions for {cell}/{field}")
            table11a.append({
                "Cell": cell,
                "Stage (cumulative from arrival)": label,
                "Update rate (/s)": MAIN_RATE,
                "Read rate (/s)": MAIN_RATE * 19,
                "p50 ms": round(med(pct(values, 50) for values in rep_stats.values()), 3),
                "p95 ms": round(med(pct(values, 95) for values in rep_stats.values()), 3),
                "p99 ms": round(med(pct(values, 99) for values in rep_stats.values()), 3),
            })

    # Table 11B: median of three scenario-level rate-sweep measurements.
    grouped = defaultdict(list)
    for row in all_scenarios:
        grouped[(row["cell"], float(row["update_rate_s"]))].append(row)
    event_groups = defaultdict(list)
    for row in all_events:
        event_groups[(row["cell"], float(row["update_rate_s"]), int(row["repetition"]))].append(row)
    table11b = []
    for cell in CELLS:
        for rate in RATES:
            rows = grouped[(cell, rate)]
            if len(rows) != 3:
                raise RuntimeError(f"Expected three scenarios for {cell}/{rate}")
            repetitions = [event_groups[(cell, rate, repetition)] for repetition in (0, 1, 2)]
            if any(not events for events in repetitions):
                raise RuntimeError(f"Missing event rows for {cell}/{rate}")
            item = {
                "Cell": cell,
                "Update rate (/s)": rate,
                "Total arrival rate (/s)": rate * 20,
                "Pipeline p95 ms": round(med(float(row["pipeline_complete_p95_ms"]) for row in rows), 3),
                "FreshHit@10 50ms (final-hit conditioned)": round(med(final_hit_conditioned_rate(events, 50) for events in repetitions), 4),
                "FreshHit@10 100ms (final-hit conditioned)": round(med(final_hit_conditioned_rate(events, 100) for events in repetitions), 4),
                "FreshHit@10 250ms (final-hit conditioned)": round(med(final_hit_conditioned_rate(events, 250) for events in repetitions), 4),
                "FreshHit@10 500ms (final-hit conditioned)": round(med(final_hit_conditioned_rate(events, 500) for events in repetitions), 4),
                "FreshHit@10 1s (final-hit conditioned)": round(med(final_hit_conditioned_rate(events, 1000) for events in repetitions), 4),
                "Pipeline visible @1s": round(med(float(row["pipeline_visible_1000ms"]) for row in rows), 4),
                "Final Hit@10": round(med(float(row["final_hit_at_10"]) for row in rows), 4),
                "Update backlog peak": int(round(med(float(row["update_backlog_peak"]) for row in rows))),
                "Backlog drain ms": round(med(float(row["backlog_drain_ms"]) for row in rows), 3),
                "Timeout rate (>5s)": round(med(float(row["timeout_rate_5000ms"]) for row in rows), 4),
                "Error rate": round(med(float(row["error_rate"]) for row in rows), 4),
                "Missed update rate": round(med(float(row["missed_update_rate"]) for row in rows), 4),
                "Duplicate update rate": round(med(float(row["duplicate_update_rate"]) for row in rows), 4),
            }
            item["Pipeline not visible @1s"] = round(1.0 - item["Pipeline visible @1s"], 4)
            item["SLO pass"] = int(
                item["Pipeline p95 ms"] <= SLO_P95_MS
                and item["Backlog drain ms"] <= SLO_DRAIN_MS
                and item["Timeout rate (>5s)"] == 0
                and item["Error rate"] == 0
                and item["Missed update rate"] == 0
                and item["Duplicate update rate"] == 0
            )
            table11b.append(item)

    sustainable = []
    for cell in CELLS:
        rows = [row for row in table11b if row["Cell"] == cell]
        passing = [float(row["Update rate (/s)"]) for row in rows if row["SLO pass"] == 1]
        sustainable.append({
            "Cell": cell,
            "Max tested sustainable update rate (/s)": max(passing) if passing else 0,
            "Corresponding total arrival rate (/s)": (max(passing) * 20) if passing else 0,
            "SLO": f"p95 pipeline <= {SLO_P95_MS:.0f} ms; drain <= {SLO_DRAIN_MS:.0f} ms; zero timeout/error/missed/duplicate",
        })

    BASE.mkdir(parents=True, exist_ok=True)
    paths = {
        "table11a_stage_latency.csv": table11a,
        "table11b_online_freshness.csv": table11b,
        "sustainable_update_rate.csv": sustainable,
    }
    for name, rows in paths.items():
        write_csv(BASE / name, rows)

    report = f"""# Experiment 11: Online Memory Freshness / Observed Time-to-Top10

## Frozen protocol

- Corpus: deterministic 100K LoCoMo-shaped trace replay, 171 namespaces.
- Cells: Cassandra-base, Cassandra-materialized, Neo4j-native, Neo4j-materialized.
- Workload: open-loop 95% reads / 5% updates, concurrency 32.
- Update rates: 1, 2, 5, and 10 updates/s (20, 40, 100, and 200 total operations/s).
- Sampling: at least 40 updates per rate/repetition; three repetitions; 2,400 measured staged updates and 48,000 total operations.
- Inputs prepared before t0: raw text, ERK/triples, frozen BGE embedding, and query embedding. LLM extraction and embedding generation are excluded.
- Aggregation: median of three repetition-level statistics. No failed, missed, or duplicate updates were removed.

## Timing boundaries

- `t_commit`: arrival to acknowledged raw-memory row/node commit.
- `t_structured_view_visible`: arrival to full graph digest, RawERK candidate projection, and relation-candidate visibility.
- `t_index_visible`: arrival to both sparse and dense index version visibility.
- `time_to_top10`: arrival to inclusion in the single post-update Dense + BM25-RawERK + query-wise ZScore Top-10 query.
- `FreshHit@10(T)`: among update-query pairs included in that post-sync Top-10, the fraction whose end-to-end completion is within deadline T. Final non-hits are excluded because their absence may reflect retrieval relevance rather than backend freshness.
- `Final Hit@10`: fraction of all updates included by the post-sync Top-10 query; it is reported separately and is not interpreted as a backend freshness-failure rate.
- `Pipeline complete`: arrival to completed Top-10 computation, including final non-hits.

## Table 11A. Stage latency at the main 100 ops/s point

{markdown_table(table11a)}

## Table 11B. Final-hit-conditioned deadline freshness and overload behavior

{markdown_table(table11b)}

## Sustainable update rate under the frozen SLO

SLO: pipeline p95 <= 2,000 ms, backlog drain <= 2,000 ms, and zero timeout/error/missed/duplicate updates.

{markdown_table(sustainable)}

## Conclusions and claim boundary

1. At light load (20 total ops/s), Neo4j is competitive and has lower p95 than Cassandra in this local setup. The result is not a universal claim that Cassandra has lower single-event latency.
2. At the main 100 ops/s point, Cassandra-materialized reaches structured-view visibility at p95 69.815 ms and complete Top-10 at p95 163.278 ms. Neo4j-materialized is queue-saturated: the corresponding p95 values are 16,716.849 ms and 17,605.288 ms.
3. Cassandra-materialized sustains the highest tested 10 updates/s (200 total ops/s) under the frozen SLO. Both Neo4j cells sustain 1 update/s (20 total ops/s); 2 updates/s already exceeds the drain and p95 limits.
4. Materialization reduces Cassandra structured-view p95 at the 100 ops/s point (124.351 -> 69.815 ms), but it does not remove the shared sparse/dense reranking cost. Neo4j materialization does not prevent queue saturation at 40+ total ops/s in this implementation.
5. Supported claim: under the tested 100K LoCoMo-shaped open-loop workload, Cassandra-materialized keeps final-hit-eligible structured evidence entering the observed CassMem Top-10 within the serving SLO at substantially higher arrival rates. Do not interpret final non-hits as stale backend results or claim that Cassandra is always faster at low load.
"""
    report_path = BASE / "EXPERIMENT11_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "status": "PASS",
        "protocol_id": "experiment11-online-freshness-v2",
        "cells": list(CELLS),
        "concurrency": 32,
        "rates": list(RATES),
        "repetitions": 3,
        "minimum_updates_per_scenario": 40,
        "read_update_ratio": "95:5",
        "formal_scenarios": len(all_scenarios),
        "formal_update_events": len(all_events),
        "main_stage_rate": MAIN_RATE,
        "slo": {"pipeline_p95_ms": SLO_P95_MS, "backlog_drain_ms": SLO_DRAIN_MS, "requires_zero_timeout_error_missed_duplicate": True},
        "outputs": {**{name: sha(BASE / name) for name in paths}, "EXPERIMENT11_REPORT.md": sha(report_path)},
    }
    (BASE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
