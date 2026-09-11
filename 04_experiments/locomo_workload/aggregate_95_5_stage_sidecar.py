"""Aggregate the frozen 100K LoCoMo-shaped 95:5 stage-timing sidecar.

The four cumulative visibility timestamps share one t0. Derived stage durations
are differences between adjacent timestamps. Time-to-Top10 is hit-conditional;
FreshHit@10 is always reported over all events.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "05_reports" / "locomo_workload_100k" / "stage_sidecar"
OUTPUT = ROOT / "05_reports" / "locomo_workload_100k" / "stage_sidecar_aggregate"
CELLS = (
    "cassandra-base",
    "cassandra-materialized",
    "neo4j-native",
    "neo4j-materialized",
)
CONCURRENCIES = (1, 8, 16, 32, 64)
REPETITIONS = (0, 1, 2)
PERCENTILES = (50, 95, 99)
FRESHNESS_DEADLINES_MS = (50, 100, 250, 500, 1000)

CUMULATIVE = (
    "commit_ms",
    "rawerk_visible_ms",
    "sparse_index_visible_ms",
    "dense_index_visible_ms",
)
STAGES = (
    "backend_commit_ms",
    "structured_view_after_commit_ms",
    "sparse_index_update_ms",
    "dense_index_update_ms",
    "candidate_fetch_ms",
    "sparse_scoring_ms",
    "dense_scoring_ms",
    "zscore_fusion_ms",
)


def q(values: list[float], percentile: int) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile, method="linear"))


def f4(value: float) -> float:
    return round(float(value), 4)


def read_rows() -> list[dict]:
    rows: list[dict] = []
    expected_files = 0
    for cell in CELLS:
        for concurrency in CONCURRENCIES:
            for repetition in REPETITIONS:
                expected_files += 1
                path = INPUT / f"{cell}_c{concurrency}_r{repetition}_updates.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                with path.open(encoding="utf-8", newline="") as handle:
                    part = list(csv.DictReader(handle))
                if len(part) != 250:
                    raise ValueError(f"{path.name}: expected 250 rows, got {len(part)}")
                rows.extend(part)
    if expected_files != 60 or len(rows) != 15_000:
        raise AssertionError((expected_files, len(rows)))
    return rows


def enrich(row: dict) -> dict:
    result = dict(row)
    for key in CUMULATIVE + (
        "candidate_fetch_ms",
        "sparse_scoring_ms",
        "dense_scoring_ms",
        "zscore_fusion_ms",
    ):
        result[key] = float(row[key])
    result["concurrency"] = int(row["concurrency"])
    result["repetition"] = int(row["repetition"])
    result["fresh_hit_at_10"] = int(row["fresh_hit_at_10"])
    result["projection_ok"] = int(row["projection_ok"])
    result["time_to_top10_ms"] = float(row["time_to_top10_ms"]) if row["time_to_top10_ms"] else None

    result["backend_commit_ms"] = result["commit_ms"]
    result["structured_view_after_commit_ms"] = result["rawerk_visible_ms"] - result["commit_ms"]
    result["sparse_index_update_ms"] = result["sparse_index_visible_ms"] - result["rawerk_visible_ms"]
    result["dense_index_update_ms"] = result["dense_index_visible_ms"] - result["sparse_index_visible_ms"]
    result["retrieval_after_dense_ms"] = sum(result[key] for key in STAGES[4:])
    result["end_to_end_attempt_ms"] = result["dense_index_visible_ms"] + result["retrieval_after_dense_ms"]
    return result


def grouped(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def metric_summary(rows: list[dict]) -> dict:
    out: dict[str, float | int] = {
        "events": len(rows),
        "hits": sum(row["fresh_hit_at_10"] for row in rows),
        "fresh_hit_at_10": f4(np.mean([row["fresh_hit_at_10"] for row in rows])),
        "projection_ok_rate": f4(np.mean([row["projection_ok"] for row in rows])),
    }
    hit_times = [row["time_to_top10_ms"] for row in rows if row["time_to_top10_ms"] is not None]
    out["time_to_top10_observations"] = len(hit_times)
    out["stale_or_missed_top10_rate"] = f4(1.0 - float(out["fresh_hit_at_10"]))
    for deadline in FRESHNESS_DEADLINES_MS:
        out[f"fresh_hit_at_10_within_{deadline}ms"] = f4(
            np.mean([
                int(row["time_to_top10_ms"] is not None and row["time_to_top10_ms"] <= deadline)
                for row in rows
            ])
        )
    metrics = CUMULATIVE + STAGES + ("retrieval_after_dense_ms", "end_to_end_attempt_ms")
    for metric in metrics:
        values = [row[metric] for row in rows]
        for percentile in PERCENTILES:
            out[f"{metric}_p{percentile}"] = f4(q(values, percentile))
    for percentile in PERCENTILES:
        out[f"time_to_top10_ms_p{percentile}_hits_only"] = f4(q(hit_times, percentile)) if hit_times else ""
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_bottlenecks(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for (cell, concurrency), part in sorted(grouped(rows, ("cell", "concurrency")).items()):
        means = {stage: float(np.mean([r[stage] for r in part])) for stage in STAGES}
        total_mean = sum(means.values())
        ranked = sorted(STAGES, key=lambda stage: means[stage], reverse=True)
        for rank, stage in enumerate(ranked, start=1):
            values = [r[stage] for r in part]
            output.append({
                "cell": cell,
                "concurrency": concurrency,
                "rank_by_mean": rank,
                "stage": stage,
                "mean_ms": f4(np.mean(values)),
                "share_of_mean_pipeline": f4(means[stage] / total_mean),
                "p50_ms": f4(q(values, 50)),
                "p95_ms": f4(q(values, 95)),
                "p99_ms": f4(q(values, 99)),
            })
    return output


def ratio(numerator, denominator):
    return f4(float(numerator) / float(denominator)) if float(denominator) else ""


def make_backend_comparisons(main_rows: list[dict]) -> list[dict]:
    lookup = {(r["cell"], int(r["concurrency"])): r for r in main_rows}
    pairs = (
        ("base_vs_native", "cassandra-base", "neo4j-native"),
        ("materialized", "cassandra-materialized", "neo4j-materialized"),
    )
    output: list[dict] = []
    for label, cass, neo in pairs:
        for concurrency in CONCURRENCIES:
            c = lookup[(cass, concurrency)]
            n = lookup[(neo, concurrency)]
            row = {
                "design_pair": label,
                "concurrency": concurrency,
                "cassandra_cell": cass,
                "neo4j_cell": neo,
                "fresh_hit_at_10_cassandra": c["fresh_hit_at_10"],
                "fresh_hit_at_10_neo4j": n["fresh_hit_at_10"],
                "fresh_hit_delta_cassandra_minus_neo4j": f4(c["fresh_hit_at_10"] - n["fresh_hit_at_10"]),
            }
            for metric in ("commit_ms", "rawerk_visible_ms", "dense_index_visible_ms", "end_to_end_attempt_ms"):
                for percentile in PERCENTILES:
                    key = f"{metric}_p{percentile}"
                    row[f"{key}_cassandra"] = c[key]
                    row[f"{key}_neo4j"] = n[key]
                    row[f"{key}_neo_over_cass"] = ratio(n[key], c[key])
            output.append(row)
    return output


def make_materialization_effects(main_rows: list[dict]) -> list[dict]:
    lookup = {(r["cell"], int(r["concurrency"])): r for r in main_rows}
    pairs = (
        ("cassandra", "cassandra-base", "cassandra-materialized"),
        ("neo4j", "neo4j-native", "neo4j-materialized"),
    )
    output: list[dict] = []
    for backend, baseline, materialized in pairs:
        for concurrency in CONCURRENCIES:
            b = lookup[(baseline, concurrency)]
            m = lookup[(materialized, concurrency)]
            row = {
                "backend": backend,
                "concurrency": concurrency,
                "baseline_cell": baseline,
                "materialized_cell": materialized,
                "fresh_hit_baseline": b["fresh_hit_at_10"],
                "fresh_hit_materialized": m["fresh_hit_at_10"],
                "fresh_hit_delta_materialized_minus_baseline": f4(m["fresh_hit_at_10"] - b["fresh_hit_at_10"]),
            }
            for metric in ("commit_ms", "rawerk_visible_ms", "dense_index_visible_ms", "end_to_end_attempt_ms"):
                for percentile in PERCENTILES:
                    key = f"{metric}_p{percentile}"
                    row[f"{key}_baseline"] = b[key]
                    row[f"{key}_materialized"] = m[key]
                    row[f"{key}_baseline_over_materialized"] = ratio(b[key], m[key])
            output.append(row)
    return output


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(label for _, label in columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key, _ in columns) + " |")
    return "\n".join(lines)


def make_report(main_rows: list[dict], bottlenecks: list[dict], comparisons: list[dict]) -> str:
    main_cols = [
        ("cell", "Cell"), ("concurrency", "c"), ("events", "N"),
        ("fresh_hit_at_10", "FreshHit@10"),
        ("fresh_hit_at_10_within_100ms", "Fresh@100ms"),
        ("fresh_hit_at_10_within_250ms", "Fresh@250ms"),
        ("fresh_hit_at_10_within_500ms", "Fresh@500ms"),
        ("fresh_hit_at_10_within_1000ms", "Fresh@1s"),
        ("time_to_top10_ms_p50_hits_only", "TTTop10 p50*"),
        ("time_to_top10_ms_p95_hits_only", "TTTop10 p95*"),
        ("time_to_top10_ms_p99_hits_only", "TTTop10 p99*"),
        ("commit_ms_p50", "Commit p50"), ("commit_ms_p95", "Commit p95"), ("commit_ms_p99", "Commit p99"),
        ("rawerk_visible_ms_p50", "RawERK p50"), ("rawerk_visible_ms_p95", "RawERK p95"), ("rawerk_visible_ms_p99", "RawERK p99"),
        ("end_to_end_attempt_ms_p50", "Attempt p50"), ("end_to_end_attempt_ms_p95", "Attempt p95"), ("end_to_end_attempt_ms_p99", "Attempt p99"),
    ]
    top = [row for row in bottlenecks if row["rank_by_mean"] == 1]
    top_cols = [
        ("cell", "Cell"), ("concurrency", "c"), ("stage", "Mean bottleneck"),
        ("mean_ms", "Mean ms"), ("share_of_mean_pipeline", "Share"),
        ("p95_ms", "Stage p95"), ("p99_ms", "Stage p99"),
    ]
    comp_cols = [
        ("design_pair", "Pair"), ("concurrency", "c"),
        ("fresh_hit_delta_cassandra_minus_neo4j", "FreshHit Δ Cass-Neo"),
        ("commit_ms_p95_neo_over_cass", "Commit p95 Neo/Cass"),
        ("rawerk_visible_ms_p95_neo_over_cass", "RawERK p95 Neo/Cass"),
        ("end_to_end_attempt_ms_p95_neo_over_cass", "Attempt p95 Neo/Cass"),
    ]
    return f"""# 100K LoCoMo-shaped 95:5 Stage-Timing Aggregate

## Protocol

- Four live cells × concurrency `{{1,8,16,32,64}}` × three repetitions.
- 250 measured updates per point; 750 events per cell/concurrency; 15,000 total.
- All 60 source summaries are PASS and all `projection_ok` values must equal 1.
- Percentiles pool the three frozen repetitions at the event level using NumPy linear quantiles.
- `TTTop10*` is conditional on successful Top-10 entry. `FreshHit@10` uses all events and must be interpreted jointly.
- Deadline FreshHit@10 counts a success only when the target both enters Top-10 and does so within 50/100/250/500/1000 ms.
- `Attempt` is the measured end-to-end update-and-retrieval attempt, including dense-index visibility plus candidate fetch and non-overlapping sparse/dense/fusion scoring.

## Four-cell main table (milliseconds)

{markdown_table(main_rows, main_cols)}

## Dominant stage by mean contribution

{markdown_table(top, top_cols)}

## Cassandra versus Neo4j at matched design

Values above 1.0 in latency ratios mean Neo4j is slower; values below 1.0 mean Cassandra is slower.

{markdown_table(comparisons, comp_cols)}

## Interpretation constraints

- Do not compare hit-conditional Time-to-Top10 without also reporting FreshHit@10.
- `rawerk_visible_ms`, sparse visibility, and dense visibility are cumulative timestamps, not additive stage durations.
- Backend and materialization effects are separated by the 2×2 design; this is not evidence that Cassandra dominates arbitrary graph traversal.
- Claims are limited to the frozen 100K LoCoMo-shaped, scope-known, 95:5 online memory workload.
"""


def main() -> None:
    rows = [enrich(row) for row in read_rows()]
    if any(row["projection_ok"] != 1 for row in rows):
        raise AssertionError("projection gate failed")
    for row in rows:
        if not (row["commit_ms"] <= row["rawerk_visible_ms"] <= row["sparse_index_visible_ms"] <= row["dense_index_visible_ms"]):
            raise AssertionError(f"non-monotonic cumulative stages: {row['op_id']}")
        if row["fresh_hit_at_10"] != int(row["time_to_top10_ms"] is not None):
            raise AssertionError(f"hit/time mismatch: {row['op_id']}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    main_rows = []
    for (cell, concurrency), part in sorted(grouped(rows, ("cell", "concurrency")).items()):
        if len(part) != 750:
            raise AssertionError((cell, concurrency, len(part)))
        main_rows.append({"cell": cell, "concurrency": concurrency, **metric_summary(part)})

    repetition_rows = []
    for (cell, concurrency, repetition), part in sorted(grouped(rows, ("cell", "concurrency", "repetition")).items()):
        repetition_rows.append({"cell": cell, "concurrency": concurrency, "repetition": repetition, **metric_summary(part)})

    bottlenecks = make_bottlenecks(rows)
    comparisons = make_backend_comparisons(main_rows)
    materialization = make_materialization_effects(main_rows)
    write_csv(OUTPUT / "four_cell_by_concurrency.csv", main_rows)
    write_csv(OUTPUT / "four_cell_by_concurrency_repetition.csv", repetition_rows)
    write_csv(OUTPUT / "stage_bottlenecks.csv", bottlenecks)
    write_csv(OUTPUT / "backend_comparisons.csv", comparisons)
    write_csv(OUTPUT / "materialization_effects.csv", materialization)
    (OUTPUT / "STAGE_TIMING_REPORT.md").write_text(
        make_report(main_rows, bottlenecks, comparisons), encoding="utf-8"
    )
    source_hash = hashlib.sha256()
    for path in sorted(INPUT.glob("*_updates.csv")):
        source_hash.update(path.name.encode("utf-8"))
        source_hash.update(hashlib.sha256(path.read_bytes()).digest())
    manifest = {
        "status": "PASS",
        "source_directory": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "source_files": 60,
        "source_updates_sha256": source_hash.hexdigest(),
        "events": len(rows),
        "cells": list(CELLS),
        "concurrencies": list(CONCURRENCIES),
        "repetitions": list(REPETITIONS),
        "quantile_method": "numpy.percentile(method=linear), pooled event-level across three frozen repetitions",
        "time_to_top10_scope": "hits_only; always pair with fresh_hit_at_10",
        "outputs": [
            "four_cell_by_concurrency.csv",
            "four_cell_by_concurrency_repetition.csv",
            "stage_bottlenecks.csv",
            "backend_comparisons.csv",
            "materialization_effects.csv",
            "STAGE_TIMING_REPORT.md",
        ],
    }
    (OUTPUT / "aggregate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
