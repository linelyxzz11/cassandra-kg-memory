"""
C1 stability follow-up: targeted re-runs for high-variance (system, clients) pairs.
Collects CPU/RAM resource metrics per trial.
"""
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
import threading
import psutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cassandra.cluster import Cluster
from neo4j import GraphDatabase

GRAPH_ID = "synth_100000_1781447372"
FANOUT = 20
HOP = 2
FRONTIER_WORKERS = 16

PROJ = Path("D:/memorytable/cassandra-kg-memory")
MANIFEST = PROJ / "results/c1_manifest_100k_h2.jsonl"
OUT_DIR = PROJ / "reports/c1_concurrency_100k_final_fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_PWD = os.environ["NEO4J_PASSWORD"]
NEO_L = "C1KGNode"
NEO_R = "C1KG_EDGE"

_cass_cluster = None
_cass_session = None
_cass_stmt = None
_cass_executor = None
_neo_driver = None


def init_backends():
    global _cass_cluster, _cass_session, _cass_stmt, _cass_executor, _neo_driver
    _cass_cluster = Cluster(["127.0.0.1"], port=9042)
    _cass_session = _cass_cluster.connect("ai_memory")
    _cass_stmt = _cass_session.prepare(
        "SELECT src_id, relation, dst_id, source FROM kg_edges_by_src WHERE graph_id=? AND src_id=?"
    )
    _cass_executor = ThreadPoolExecutor(max_workers=FRONTIER_WORKERS)
    _neo_driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PWD))


def shutdown_backends():
    if _cass_executor:
        _cass_executor.shutdown(wait=True)
    if _cass_session:
        _cass_cluster.shutdown()
    if _neo_driver:
        _neo_driver.close()


def cass_fetch(src):
    rows = _cass_session.execute(_cass_stmt, (GRAPH_ID, src))
    return [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]


def cass_one_hop_parallel(sources, rel):
    se = {}
    futures = {_cass_executor.submit(cass_fetch, src): src for src in sources}
    for f in as_completed(futures):
        src = futures[f]
        se[src] = [e for e in f.result() if e[1] == rel]
    return se


def cass_traverse(q, parallel=False):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for relation in q["relation_path"]:
        sources = sorted({s[0] for s in frontier})
        se = {}
        if parallel:
            se = cass_one_hop_parallel(sources, relation)
        else:
            for src in sources:
                se[src] = [e for e in cass_fetch(src) if e[1] == relation]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FANOUT]:
                if e[2] not in np:
                    nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier:
            break
    return tuple(sorted({s[2] for s in frontier}))


def neo_fetch(src, rel):
    cypher = (
        f"MATCH (n:{NEO_L} {{graph_id: $gid, node_id: $nid}})"
        f"-[r:{NEO_R} {{relation: $rel}}]->"
        f"(m:{NEO_L} {{graph_id: $gid}}) "
        "RETURN n.node_id AS s, r.relation AS r, m.node_id AS d, coalesce(r.source,'') AS src "
        "ORDER BY r, d, src"
    )
    with _neo_driver.session() as s:
        rows = list(s.run(cypher, gid=GRAPH_ID, nid=src, rel=rel))
    return [(r["s"], r["r"], r["d"], r["src"]) for r in rows]


def neo_traverse(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for relation in q["relation_path"]:
        sources = sorted({s[0] for s in frontier})
        se = {src: neo_fetch(src, relation) for src in sources}
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FANOUT]:
                if e[2] not in np:
                    nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier:
            break
    return tuple(sorted({s[2] for s in frontier}))


def path_hash(paths):
    canon = sorted([list(p) for p in sorted(paths)])
    return hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def spotcheck(system, queries):
    rng = random.Random()
    checks = rng.sample(queries, 10)
    for q in checks:
        if system == "neo4j":
            p = neo_traverse(q)
        elif system == "cassandra_parallel":
            p = cass_traverse(q, parallel=True)
        else:
            p = cass_traverse(q, parallel=False)
        if path_hash(p) != q["expected_path_hash"]:
            return False
    with (OUT_DIR / "correctness_spotcheck.jsonl").open("a", encoding="utf-8") as f:
        for q in checks:
            f.write(json.dumps({"query_id": q["query_id"], "system": system, "match": True,
                                "trial": "stability_followup"}, ensure_ascii=False) + "\n")
    return True


def monitor_resources(stop_event, samples):
    p = psutil.Process()
    while not stop_event.is_set():
        try:
            samples.append({
                "cpu_pct": p.cpu_percent(interval=0.1) / os.cpu_count() * 100,
                "rss_mb": p.memory_info().rss / 1024 / 1024,
                "vms_mb": p.memory_info().vms / 1024 / 1024,
            })
        except Exception:
            pass


def run_trial_with_monitor(system, clients, queries, warmup_s, measure_s, repeat_idx):
    print(f"\n  [{system} c={clients} r={repeat_idx+1}]", flush=True)

    per_thread = [[] for _ in range(clients)]
    stop = threading.Event()

    def client_loop(ci):
        ld = per_thread[ci]
        idx = ci % len(queries)
        while not stop.is_set():
            q = queries[idx]
            t0 = time.perf_counter()
            try:
                if system == "cassandra_naive":
                    cass_traverse(q, parallel=False)
                elif system == "cassandra_parallel":
                    cass_traverse(q, parallel=True)
                else:
                    neo_traverse(q)
                ld.append((time.perf_counter() - t0) * 1000)
            except Exception:
                ld.append(None)
            idx = (idx + 1) % len(queries)

    threads = [threading.Thread(target=client_loop, args=(i,), daemon=True) for i in range(clients)]
    for t in threads:
        t.start()
    time.sleep(warmup_s)

    # Start resource monitor
    monitor_samples = []
    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(monitor_stop, monitor_samples), daemon=True)
    monitor_thread.start()

    for ld in per_thread:
        ld.clear()
    meas_start = time.perf_counter()
    time.sleep(measure_s)
    meas_end = time.perf_counter()
    stop.set()
    monitor_stop.set()
    for t in threads:
        t.join(timeout=10)
    monitor_thread.join(timeout=3)

    meas_actual = meas_end - meas_start
    all_lat = [v for ld in per_thread for v in ld if v is not None]
    errors = sum(1 for ld in per_thread for v in ld if v is None)
    n = len(all_lat)
    if n == 0:
        return None, {}, {}

    all_lat.sort()
    mean_v = statistics.mean(all_lat)
    def pct(a, p):
        return a[int((len(a)-1)*p/100)]
    qps = n / max(meas_actual, 0.001)

    # Resource stats
    cpus = [s["cpu_pct"] for s in monitor_samples] if monitor_samples else [0]
    rss = [s["rss_mb"] for s in monitor_samples] if monitor_samples else [0]
    vms = [s["vms_mb"] for s in monitor_samples] if monitor_samples else [0]
    res = {
        "cpu_mean_pct": round(statistics.mean(cpus), 1),
        "cpu_max_pct": round(max(cpus), 1),
        "rss_mean_mb": round(statistics.mean(rss), 1),
        "rss_max_mb": round(max(rss), 1),
        "vms_max_mb": round(max(vms), 1),
        "n_samples": len(monitor_samples),
    }

    row = {
        "system": system, "clients": clients, "repeat": repeat_idx + 1,
        "warmup_s": warmup_s, "measure_s": round(meas_actual, 3),
        "queries": n, "errors": errors, "error_rate": round(errors/max(n+errors,1),6),
        "QPS": round(qps, 3), "mean_ms": round(mean_v, 3),
        "p50_ms": round(pct(all_lat, 50), 3),
        "p95_ms": round(pct(all_lat, 95), 3),
        "p99_ms": round(pct(all_lat, 99), 3),
        "qps_x_mean": round(qps * mean_v, 1),
        "frontier_workers": FRONTIER_WORKERS if system == "cassandra_parallel" else 0,
        "graph_id": GRAPH_ID, "cache": "disabled", "backend": "warm",
        "cpu_mean_pct": res["cpu_mean_pct"], "cpu_max_pct": res["cpu_max_pct"],
        "rss_mean_mb": res["rss_mean_mb"], "rss_max_mb": res["rss_max_mb"],
    }

    print(f"    QPS={qps:.1f} mean={mean_v:.1f}ms p95={pct(all_lat,95):.1f}ms "
          f"CPU_{res['cpu_mean_pct']:.0f}% RSS_{res['rss_max_mb']:.0f}MB", flush=True)
    return row, res, {}


def write_environment():
    info = {"os": platform.platform(), "python": sys.version, "cpu_logical": os.cpu_count()}
    info["cpu_model"] = platform.processor()
    try:
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        info["ram_gb"] = "unknown"

    from cassandra import __version__ as cv
    info["cassandra_driver"] = cv
    from neo4j import __version__ as nv
    info["neo4j_driver"] = nv

    try:
        c = Cluster(["127.0.0.1"], port=9042)
        s = c.connect()
        r = list(s.execute("SELECT release_version FROM system.local"))
        info["cassandra_version"] = r[0].release_version
        c.shutdown()
    except Exception as e:
        info["cassandra_version"] = f"error: {e}"

    try:
        d = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PWD))
        with d.session() as s:
            info["neo4j_version"] = s.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()["v"]
            try:
                pool = s.run("CALL dbms.listConnections() YIELD connectionId RETURN count(*) AS cnt").single()
                info["neo4j_connections"] = pool["cnt"]
            except Exception:
                pass
        d.close()
    except Exception as e:
        info["neo4j_version"] = f"error: {e}"

    info["cassandra_pool"] = "default (Cluster default)"
    info["neo4j_pool"] = "default (driver default, max_connection_lifetime=3600)"
    info["graph_id"] = GRAPH_ID
    info["frontier_workers"] = FRONTIER_WORKERS
    info["frontier_executor_scope"] = "process_shared"
    info["application_cache"] = "disabled"
    info["backend_state"] = "warm"
    info["docker_containers"] = "cassandra (local Docker), neo4j-kg (local Docker)"

    with (OUT_DIR / "environment.json").open("w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


def regenerate_final_summary():
    rows = []
    csv_path = OUT_DIR / "trial_summary.csv"
    with csv_path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["system"], int(r["clients"]))].append(r)

    final_rows = []
    for (sn, cl), group in sorted(groups.items()):
        qps = sorted([float(t["QPS"]) for t in group])
        mean_l = sorted([float(t["mean_ms"]) for t in group])
        p50_l = sorted([float(t["p50_ms"]) for t in group])
        p95_l = sorted([float(t["p95_ms"]) for t in group])
        p99_l = sorted([float(t["p99_ms"]) for t in group])
        err = sorted([float(t["error_rate"]) for t in group])
        mid = len(group) // 2
        def iqr(arr):
            q1 = arr[len(arr)//4]
            q3 = arr[3*len(arr)//4]
            return round(q3 - q1, 3)
        fr = {
            "system": sn, "clients": cl,
            "median_QPS": round(qps[mid], 3), "median_mean_ms": round(mean_l[mid], 3),
            "median_p50_ms": round(p50_l[mid], 3), "median_p95_ms": round(p95_l[mid], 3),
            "median_p99_ms": round(p99_l[mid], 3), "median_error_rate": round(err[mid], 6),
            "min_QPS": round(qps[0], 3), "max_QPS": round(qps[-1], 3),
            "iqr_QPS": iqr(qps), "repeats": len(group),
        }
        final_rows.append(fr)

    ff = ["system","clients","median_QPS","median_mean_ms","median_p50_ms","median_p95_ms",
          "median_p99_ms","median_error_rate","min_QPS","max_QPS","iqr_QPS","repeats"]
    with (OUT_DIR / "final_concurrency_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=ff, extrasaction="ignore")
        w.writeheader()
        w.writerows(final_rows)
    with (OUT_DIR / "final_concurrency_summary.json").open("w") as f:
        json.dump(final_rows, f, indent=2, ensure_ascii=False)

    with (OUT_DIR / "stability_followup_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=ff, extrasaction="ignore")
        w.writeheader()
        followup_rows = [fr for fr in final_rows if fr["repeats"] > 3]
        w.writerows(followup_rows)
        print(f"\n--- Stability Followup ---", flush=True)
        for fr in followup_rows:
            print(f"  {fr['system']:20s} c={fr['clients']:2d}  "
                  f"QPS={fr['median_QPS']:7.1f}  IQR={fr['iqr_QPS']:7.1f}  n={fr['repeats']}", flush=True)

    return final_rows


def main():
    if not NEO4J_PWD:
        print("ERROR: NEO4J_PASSWORD", flush=True); sys.exit(1)

    queries = []
    with MANIFEST.open() as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"Loaded {len(queries)} queries", flush=True)

    print("Init backends...", flush=True)
    init_backends()

    write_environment()

    # Targeted pairs: high variance in original 3 runs
    targets = [
        ("cassandra_parallel", 32),
        ("neo4j", 32),
        ("neo4j", 64),
    ]
    extra_repeats = 5
    warm, meas = 15, 45

    csv_path = OUT_DIR / "trial_summary.csv"
    jsonl_path = OUT_DIR / "trial_summary.jsonl"
    fields = None

    rng = random.Random(20260707)
    total = len(targets) * extra_repeats
    idx = 0
    all_ok = True

    for sys_name, cl in targets:
        for rep in range(3, 3 + extra_repeats):
            idx += 1
            print(f"\n[{idx}/{total}] {sys_name} clients={cl} repeat={rep+1}", flush=True)
            row, res, _ = run_trial_with_monitor(sys_name, cl, queries, warm, meas, rep)
            if row is None:
                all_ok = False
                continue

            if not fields:
                fields = list(row.keys())
                write_csv = True
            else:
                write_csv = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if write_csv:
                    w.writeheader()
                w.writerow(row)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if not spotcheck(sys_name, queries):
                print(f"  SPOTCHECK FAILED", flush=True)
                all_ok = False

    shutdown_backends()

    final = regenerate_final_summary()
    print(f"\n=== UPDATED FINAL SUMMARY ===", flush=True)
    for fr in final:
        print(f"  {fr['system']:20s} c={fr['clients']:2d}  "
              f"QPS={fr['median_QPS']:7.1f}  mean={fr['median_mean_ms']:7.1f}ms  "
              f"p95={fr['median_p95_ms']:7.1f}ms  p99={fr['median_p99_ms']:8.1f}ms  "
              f"min={fr['min_QPS']:7.1f}  max={fr['max_QPS']:7.1f}  IQR={fr['iqr_QPS']:7.1f}  n={fr['repeats']}",
              flush=True)

    print(f"\n{'ALL PASS' if all_ok else 'ISSUES'}", flush=True)


if __name__ == "__main__":
    main()
