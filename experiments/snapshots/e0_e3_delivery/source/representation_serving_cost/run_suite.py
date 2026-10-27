"""Serial benchmark supervisor with durable progress and guaranteed restoration."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from common import ARMS, HERE, source_manifest, write_csv, write_json, sha
from environment import NAME, STATE, docker, inspect, restore, safe_config

OUT = HERE / "results" / "formal_20261012_v1"
KEYSPACE = "repcost_formal_v1"


def run(args, log):
    with log.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen([sys.executable, "-B", str(HERE / "benchmark.py"), *args],
                                 stdout=handle, stderr=subprocess.STDOUT, cwd=HERE)
        write_json(OUT / "current_process.json", {"pid": child.pid, "arguments": args, "log": str(log)})
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait(timeout=90)
            raise
    if code:
        raise RuntimeError(f"Benchmark failed ({code}); see {log}")


def matrix():
    trials = []
    # Rotate views and layouts across repetitions. No simultaneous trials.
    for rep in range(3):
        arms = ARMS[rep:] + ARMS[:rep]
        cells = ("cassandra-materialized", "cassandra-base") if rep % 2 == 0 else ("cassandra-base", "cassandra-materialized")
        for concurrency in (1, 8, 16, 32):
            for arm in arms:
                for cell in cells:
                    if cell == "cassandra-base" and concurrency not in (1, 32):
                        continue
                    trials.append({"mode":"closed", "rep":rep, "arm":arm, "cell":cell,
                                   "concurrency":concurrency, "rate":0, "seconds":180, "warmup":60})
        for rate in (1, 2, 5, 10):
            for arm in arms:
                for cell in cells:
                    if cell == "cassandra-base" and rate not in (1, 10):
                        continue
                    trials.append({"mode":"open", "rep":rep, "arm":arm, "cell":cell,
                                   "concurrency":32, "rate":rate, "seconds":120, "warmup":60})
    return trials


def aggregate():
    rows = []
    for p in sorted((OUT / "trials").glob("*/summary.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"trial":p.parent.name, **r})
    write_csv(OUT / "all_trials.csv", rows)
    groups = {}
    for r in rows:
        key = (r["arm"], r["cell"], r["concurrency"], r["offered_updates_per_second"])
        groups.setdefault(key, []).append(r)
    import numpy as np
    table = []
    for (arm,cell,concurrency,rate), group in groups.items():
        entry = {"arm":arm, "cell":cell, "concurrency":concurrency, "offered_updates_per_second":rate,
                 "repetitions":len(group), "all_pass":all(r["status"] == "PASS" for r in group)}
        for field in ("throughput_ops_s", "query_qps", "update_qps", "read_service_ms_p95",
                      "read_candidate_access_ms_p95", "update_arrival_to_retrievable_ms_p95",
                      "update_queue_wait_ms_p95", "update_raw_write_call_ms_p95", "update_ack_to_retrievable_ms_p95",
                      "ready_within_500ms_fraction", "peak_pending_updates"):
            vals = [r[field] for r in group if r.get(field) is not None]
            if vals:
                for stat,fn in (("median",np.median),("min",min),("max",max)):
                    entry[f"{field}_{stat}"] = float(fn(vals))
        table.append(entry)
    write_csv(OUT / "aggregated.csv", table)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "logs").mkdir()
    (OUT / "trials").mkdir()
    config = safe_config(NAME)
    assert config["state"] == "running"
    assert config["resources"] == {"Memory":12*1024**3,"NanoCpus":8*10**9,"MemorySwap":12*1024**3}
    assert all(not inspect(n)["State"]["Running"] for n in ("cassandra", "neo4j-kg"))
    gate = json.loads((HERE / "results/smoke_v1_gate/gate.json").read_text(encoding="utf-8"))
    audit = json.loads((HERE / "results/quality_20261012_v1/difference_audit.json").read_text(encoding="utf-8"))
    closed_gate = json.loads((HERE / "results/smoke_v1_closed32/summary.json").read_text(encoding="utf-8"))
    assert gate["status"] == audit["status"] == closed_gate["status"] == "PASS"
    trials = matrix()
    progress = {"status":"RUNNING", "stage":"load_100k", "completed":0, "total":len(trials),
                "minimum_timed_hours":sum(t["seconds"]+t["warmup"] for t in trials)/3600,
                "container":config, "sources":source_manifest(), "started_unix":time.time()}
    write_json(OUT / "progress.json", progress)
    write_json(OUT / "matrix.json", trials)
    import platform
    import importlib.metadata
    import psutil
    write_json(OUT / "client_environment.json", {
        "platform": platform.platform(), "python": sys.version, "executable": sys.executable,
        "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
        "host_ram_bytes": psutil.virtual_memory().total,
        "host_available_ram_bytes_at_start": psutil.virtual_memory().available,
        "drive_D_usage": psutil.disk_usage("D:/")._asdict(),
        "packages": {p:importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "cassandra-driver", "psutil")},
        "blas_threads": 1, "client_cpu_affinity": psutil.Process().cpu_affinity(),
        "database_protocol": 4, "read_write_consistency": "LOCAL_ONE", "replication_factor":1,
        "connection_pool": "one Cluster/Session per trial, driver protocol-v4 defaults; database calls synchronous within worker",
        "batching": "raw insert separate; structured view uses the existing adapter LOGGED batch per memory",
        "update_inputs": "precomputed source records/vectors copied to unique fresh IDs; insert-only logical arrivals",
    })
    stop = threading.Event()
    def monitor():
        with (OUT / "container_samples.jsonl").open("w", encoding="utf-8") as handle:
            while not stop.is_set():
                sample = {"unix":time.time()}
                try:
                    sample["stats"] = docker("stats", "--no-stream", "--format", "{{json .}}", NAME, timeout=20).stdout.strip()
                except Exception as exc:
                    sample["error"] = str(exc)
                handle.write(json.dumps(sample)+"\n")
                handle.flush()
                stop.wait(10)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        run(["load", "--keyspace",KEYSPACE,"--memories","100000","--output",str(OUT / "load")], OUT / "logs/load.log")
        docker("exec", NAME, "nodetool", "flush", KEYSPACE)
        write_json(OUT / "initial_storage.json", {"tablestats":docker("exec",NAME,"nodetool","tablestats",KEYSPACE).stdout,
                                                    "boundary":"Common canonical DB state; post-load flush, natural compaction; not lexical index bytes"})
        for i, point in enumerate(trials):
            for path, expected in progress["sources"].items():
                if path.endswith(".py") and sha(path) != expected:
                    raise RuntimeError(f"Source changed during suite; refusing mixed implementations: {path}")
            label = f"{i:03d}_{point['mode']}_{point['arm']}_{point['cell']}_c{point['concurrency']}_u{point['rate']}_r{point['rep']}"
            progress.update(stage=label)
            write_json(OUT / "progress.json",progress)
            print(f"[{i+1}/{len(trials)}] {label}",flush=True)
            run(["run","--keyspace",KEYSPACE,"--memories","100000","--output",str(OUT / "trials" / label),
                 "--arm",point["arm"],"--cell",point["cell"],"--concurrency",str(point["concurrency"]),
                 "--rep",str(point["rep"]),"--warmup-seconds",str(point["warmup"]),"--seconds",str(point["seconds"]),
                 "--rate",str(point["rate"])],OUT / "logs" / f"{label}.log")
            progress["completed"] = i+1
            write_json(OUT / "progress.json",progress)
            aggregate()
        progress.update(status="COMPLETED",stage="complete",finished_unix=time.time())
    except BaseException as exc:
        progress.update(status="FAILED",error=f"{type(exc).__name__}: {exc}",finished_unix=time.time())
        raise
    finally:
        stop.set()
        thread.join(timeout=25)
        write_json(OUT / "progress.json",progress)
        restore()


if __name__ == "__main__":
    main()
