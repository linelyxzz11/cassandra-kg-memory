"""Validate and summarize a completed resource-matched serving rerun.

This is a post-run analysis only. It does not connect to or modify either database.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parents[1]
CELLS = ("cassandra-base", "cassandra-materialized", "neo4j-native", "neo4j-materialized")
BACKENDS = ("cassandra", "neo4j")
RATES = (1.0, 2.0, 5.0, 10.0)
FRESHNESS_METRICS = (
    "queue_wait_p95_ms", "raw_write_call_p95_ms", "arrival_to_raw_ack_p95_ms",
    "pipeline_complete_p95_ms", "read_arrival_p95_ms", "update_backlog_peak",
    "total_backlog_peak", "backlog_drain_ms", "pipeline_visible_500ms",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def memory_gib(value: str) -> float:
    match = re.match(r"\s*([\d.]+)\s*([KMGT]?i?B)", value)
    require(match is not None, f"Unrecognized Docker memory sample: {value!r}")
    factors = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3,
               "TiB": 1024**4, "kB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}
    return float(match.group(1)) * factors[match.group(2)] / 1024**3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "results" / "serving_100k" /
                        "resource_matched_v3" / "resmatch_20260924_v3r1")
    parser.add_argument("--allow-missing-stats", action="store_true",
                        help="Diagnostic only: mark incomplete instead of accepting missing resource telemetry")
    args = parser.parse_args()
    base = args.base.resolve()
    protocol = read_json(base / "protocol.json")
    if "launcher_sha256" in protocol:
        require(sha256(ROOT / "experiments" / "serving_100k" / "run_resource_matched_v3.py") ==
                protocol["launcher_sha256"], "Frozen launcher changed")
    trace_manifest = ROOT / "results" / "serving_100k" / "locomo_workload_100k" / "traces" / "trace_manifest.json"
    require(sha256(trace_manifest) == protocol["trace_manifest_sha256"], "Trace manifest changed")
    for repetition in protocol["trace_manifest"]["repetitions"]:
        for item in repetition["files"]:
            require(sha256(ROOT / item["path"]) == item["sha256"], f"Frozen trace changed: {item['path']}")
    source_paths = {
        "adapter_sha256": ROOT / "src" / "cassmem" / "backend" / "live_cells_graph.py",
        "loader_sha256": ROOT / "experiments" / "serving_100k" / "run_graph_100k_load_gate_v2.py",
        "main_workload_sha256": ROOT / "experiments" / "serving_100k" / "run_graph_100k_workload_v2.py",
        "freshness_workload_sha256": ROOT / "experiments" / "freshness" / "run_experiment11_freshness_v2.py",
    }
    for field, path in source_paths.items():
        require(sha256(path) == protocol[field], f"Frozen source changed: {path}")

    report: dict = {"status": "PASS", "protocol_id": protocol["protocol_id"],
                    "main_runs": 0, "main_measured_events": 0,
                    "freshness_scenarios": 0, "freshness_updates": 0,
                    "freshness_reads": 0, "backend_resources": {}}
    for backend in BACKENDS:
        directory = base / backend
        for phase in ("preflight", "formal"):
            require(read_json(directory / f"{phase}_pass.json")["status"] == "PASS",
                    f"{backend} {phase} did not pass")
        load = read_json(directory / "load" / f"graph_100k_load_gate_{backend}_summary.json")
        require(load["status"] == "PASS" and load["graph_digest_mismatches"] == 0
                and load["relation_candidate_mismatches"] == 0, f"{backend} load gate failed")
        require(all(count == 100_000 for count in load["memory_counts"].values()),
                f"{backend} memory count differs from 100K")
        config = read_json(directory / "formal_container.json")
        require(config["memory_bytes"] == 12 * 1024**3
                and config["memory_swap_bytes"] == 12 * 1024**3
                and config["nano_cpus"] == 8_000_000_000,
                f"{backend} Docker budget mismatch")
        cgroup = read_json(directory / "formal_cgroup_after.json")
        require("oom_kill 0" in cgroup["memory.events"], f"{backend} OOM kill")
        peak_bytes = cgroup.get("memory.peak")
        if peak_bytes is None:
            require(args.allow_missing_stats, f"{backend} missing cgroup memory.peak")
            report["status"] = "INCOMPLETE_RUNTIME_SAMPLES"
        else:
            require(int(peak_bytes) <= 12 * 1024**3, f"{backend} cgroup memory peak exceeded budget")
        samples = [json.loads(line) for line in (directory / "formal_stats.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
        if not samples:
            require(args.allow_missing_stats, f"{backend} has no runtime resource samples")
            report["status"] = "INCOMPLETE_RUNTIME_SAMPLES"
        report["backend_resources"][backend] = {
            "memory_budget_gib": 12,
            "vcpus": 8,
            "docker_stats_samples": len(samples),
            "docker_stats_peak_memory_gib": round(max(memory_gib(s["MemUsage"]) for s in samples), 3) if samples else None,
            "docker_stats_peak_cpu_percent": round(max(float(s["CPUPerc"].rstrip("%")) for s in samples), 2) if samples else None,
            "cgroup_memory_peak_gib": round(int(peak_bytes) / 1024**3, 3) if peak_bytes else None,
            "cgroup_oom_kill": 0,
            "load_memory_count_per_cell": load["memory_counts"],
            "graph_digest_mismatches": 0,
            "relation_candidate_mismatches": 0,
        }

    for cell in CELLS:
        for concurrency in (1, 8, 16, 32, 64):
            for rep in (0, 1, 2):
                stem = f"{cell}_c{concurrency}_r{rep}"
                summary = read_json(base / "main" / "runs" / f"{stem}_summary.json")
                require(summary["status"] == "PASS" and summary["measured_operations"] == 5000,
                        f"Main run incomplete: {stem}")
                event_path = base / "main" / "runs" / f"{stem}_events.csv"
                with event_path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    count = 0
                    for event in reader:
                        require(event["status"] == "ok", f"Failed event: {stem}")
                        if event["op_type"] == "update":
                            require(event["graph_projection_ok"] == "1" and
                                    event["relation_candidate_ok"] == "1", f"Semantic check failed: {stem}")
                        count += 1
                require(count == 5000, f"Main event row count differs: {stem}: {count}")
                report["main_runs"] += 1
                report["main_measured_events"] += count

    freshness_summary = []
    for cell in CELLS:
        directory = base / "freshness" / "runs"
        manifest = read_json(directory / f"{cell}_c32_manifest.json")
        require(manifest["status"] == "PASS" and manifest["scenarios"] == 12
                and manifest["updates"] == 600 and manifest["errors"] == 0,
                f"Freshness manifest failed: {cell}")
        scenarios = rows(directory / f"{cell}_c32_scenarios.csv")
        events = rows(directory / f"{cell}_c32_events.csv")
        reads = rows(directory / f"{cell}_c32_reads.csv")
        require(len(scenarios) == 12 and len(events) == 600 and len(reads) == 11_400,
                f"Freshness row count failed: {cell}")
        require(all(row["status"] == "PASS" for row in scenarios), f"Freshness scenario failed: {cell}")
        require(all(row["structured_projection_ok"] == "1" and row["relation_candidate_ok"] == "1"
                    and row["sparse_index_visible"] == "1" and row["dense_index_visible"] == "1"
                    for row in events), f"Freshness visibility failed: {cell}")
        report["freshness_scenarios"] += len(scenarios)
        report["freshness_updates"] += len(events)
        report["freshness_reads"] += len(reads)
        arrivals = defaultdict(list)
        for event in events:
            arrivals[(int(event["repetition"]), float(event["update_rate_s"]))].append(float(event["t_commit_ms"]))
        for scenario in scenarios:
            key = (int(scenario["repetition"]), float(scenario["update_rate_s"]))
            scenario["arrival_to_raw_ack_p95_ms"] = float(np.percentile(arrivals[key], 95))
        for rate in RATES:
            group = [row for row in scenarios if float(row["update_rate_s"]) == rate]
            require(len(group) == 3, f"Missing repetitions for {cell} rate={rate}")
            output: dict = {"cell": cell, "update_rate_s": rate, "repetitions": 3,
                            "updates": sum(int(row["updates"]) for row in group),
                            "reference_top10_pairs": sum(int(row["fresh_hit_at_10_denominator"]) for row in group)}
            for metric in FRESHNESS_METRICS:
                values = [float(row[metric]) for row in group]
                output[f"{metric}_median"] = statistics.median(values)
                output[f"{metric}_min"] = min(values)
                output[f"{metric}_max"] = max(values)
            freshness_summary.append(output)

    require(report["main_runs"] == 60 and report["main_measured_events"] == 300_000
            and report["freshness_scenarios"] == 48 and report["freshness_updates"] == 2400
            and report["freshness_reads"] == 45_600, "Global count gate failed")
    out = base / "audit"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "freshness_by_rate.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(freshness_summary[0]))
        writer.writeheader()
        writer.writerows(freshness_summary)
    report["freshness_by_rate_sha256"] = sha256(csv_path)
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
