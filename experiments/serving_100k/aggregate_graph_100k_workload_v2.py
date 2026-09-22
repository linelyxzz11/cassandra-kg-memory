"""Aggregate the graph-aware v2 100K 95:5 main matrix."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

from cassmem.serving.environment import ROOT

BASE = ROOT / "results" / "serving_100k" / "locomo_workload_graph_v2_100k" / "main_95_5_v2"
RUNS = BASE / "runs"
CELLS = ("cassandra-base", "cassandra-materialized", "neo4j-native", "neo4j-materialized")
CONCURRENCIES = (1, 8, 16, 32, 64)
REPS = (0, 1, 2)
METRICS = ("throughput_ops_s", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "read_p50_ms", "read_p95_ms", "read_p99_ms", "update_p50_ms", "update_p95_ms", "update_p99_ms", "fresh_hit_at_10", "error_rate")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = []; missing = []
    for cell in CELLS:
        for concurrency in CONCURRENCIES:
            for rep in REPS:
                path = RUNS / f"{cell}_c{concurrency}_r{rep}_summary.json"
                if not path.exists(): missing.append(str(path)); continue
                row = json.loads(path.read_text(encoding="utf-8")); row["source_sha256"] = sha(path); raw.append(row)
    if missing: raise SystemExit("Missing formal summaries:\n" + "\n".join(missing))
    if any(row["status"] != "PASS" or row["measured_operations"] != 5000 for row in raw): raise SystemExit("At least one formal run failed its gate")
    aggregate = []
    for cell in CELLS:
        for concurrency in CONCURRENCIES:
            group = [row for row in raw if row["cell"]==cell and row["concurrency"]==concurrency]
            out = {"cell":cell,"concurrency":concurrency,"repetitions":len(group),"measured_operations_per_rep":5000,"reads_per_rep":4750,"updates_per_rep":250}
            for metric in METRICS:
                values=[float(row[metric]) for row in group]; out[f"{metric}_median"] = statistics.median(values); out[f"{metric}_min"] = min(values); out[f"{metric}_max"] = max(values)
            out["status"]="PASS"; aggregate.append(out)
    BASE.mkdir(parents=True,exist_ok=True)
    for name, rows in (("main_95_5_runs.csv",raw),("main_95_5_summary.csv",aggregate)):
        with (BASE/name).open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest={"status":"PASS","protocol_id":"graph-aware-logical-event-v2-100k-95r5u","cells":list(CELLS),"concurrency":list(CONCURRENCIES),"repetitions":3,"formal_runs":len(raw),"operations_per_run":5000,"read_update_mix":"95:5","aggregation":"median of three run-level statistics; min/max retained","outputs":{name:sha(BASE/name) for name in ("main_95_5_runs.csv","main_95_5_summary.csv")}}
    (BASE/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__": main()
