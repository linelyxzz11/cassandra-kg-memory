"""
C1-1 fix: persistent backends, proper per-trial latency clearing, closed-loop timing
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cassandra.cluster import Cluster

try:
    from neo4j import GraphDatabase
    NEO4J_OK = True
except ImportError:
    NEO4J_OK = False

GRAPH_ID = "synth_100000_1781447372"
FANOUT = 20
HOP = 2
FRONTIER_WORKERS = 16

PROJ = Path("D:/memorytable/cassandra-kg-memory")
MANIFEST = PROJ / "results/c1_manifest_100k_h2.jsonl"
OUT_DIR = PROJ / "reports/c1_concurrency_100k_final_fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PWD = os.environ.get("NEO4J_PASSWORD")
NEO_LABEL = "C1KGNode"
NEO_REL = "C1KG_EDGE"

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
    if NEO4J_OK:
        _neo_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))


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
        f"MATCH (n:{NEO_LABEL} {{graph_id: $gid, node_id: $nid}})"
        f"-[r:{NEO_REL} {{relation: $rel}}]->"
        f"(m:{NEO_LABEL} {{graph_id: $gid}}) "
        "RETURN n.node_id AS src_id, r.relation AS relation, m.node_id AS dst_id,"
        " coalesce(r.source,'') AS source ORDER BY relation, dst_id, source"
    )
    with _neo_driver.session() as sess:
        rows = list(sess.run(cypher, gid=GRAPH_ID, nid=src, rel=rel))
    return [(r["src_id"], r["relation"], r["dst_id"], r["source"]) for r in rows]


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


def patch_manifest(queries):
    for q in queries:
        if q.get("expected_path_hash"):
            continue
        q["expected_path_hash"] = path_hash(cass_traverse(q, parallel=False))
    hashes = [q["expected_path_hash"] for q in queries]
    msha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    with MANIFEST.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return msha


def spotcheck(system, queries):
    rng = random.Random()
    checks = rng.sample(queries, 10)
    results = []
    for q in checks:
        if system == "neo4j":
            p = neo_traverse(q)
        elif system == "cassandra_parallel":
            p = cass_traverse(q, parallel=True)
        else:
            p = cass_traverse(q, parallel=False)
        match = path_hash(p) == q["expected_path_hash"]
        results.append({"query_id": q["query_id"], "system": system, "match": match})
    with (OUT_DIR / "correctness_spotcheck.jsonl").open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return all(r["match"] for r in results)


def run_trial(system, clients, queries, warmup_s, measure_s, repeat_idx):
    label = f"{system} clients={clients} repeat={repeat_idx+1}"
    print(f"\n  [{label}]", flush=True)

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

    # Clear all thread buffers in-place (threads write to per_thread[i], we clear per_thread[i])
    for ld in per_thread:
        ld.clear()
    meas_start = time.perf_counter()
    time.sleep(measure_s)
    meas_end = time.perf_counter()
    stop.set()
    for t in threads:
        t.join(timeout=10)

    meas_actual = meas_end - meas_start
    all_lat = [v for ld in per_thread for v in ld if v is not None]
    errors = sum(1 for ld in per_thread for v in ld if v is None)
    n = len(all_lat)
    if n == 0:
        return None

    all_lat.sort()
    mean_v = statistics.mean(all_lat)
    def pct(a, p):
        return a[int((len(a)-1)*p/100)]
    qps = n / max(meas_actual, 0.001)
    qxm = qps * mean_v

    print(f"    QPS={qps:.1f} mean={mean_v:.1f}ms p95={pct(all_lat,95):.1f}ms "
          f"qxm={qxm:.0f} {'***' if abs(qxm-1000*clients)>100*clients else ''}", flush=True)

    return {
        "system": system, "clients": clients, "repeat": repeat_idx + 1,
        "warmup_s": warmup_s, "measure_s": round(meas_actual, 3),
        "queries": n, "errors": errors, "error_rate": round(errors/max(n+errors,1),6),
        "QPS": round(qps, 3), "mean_ms": round(mean_v, 3),
        "p50_ms": round(pct(all_lat, 50), 3), "p95_ms": round(pct(all_lat, 95), 3),
        "p99_ms": round(pct(all_lat, 99), 3),
        "qps_x_mean": round(qxm, 1),
        "frontier_workers": FRONTIER_WORKERS if system=="cassandra_parallel" else 0,
        "graph_id": GRAPH_ID, "cache": "disabled", "backend": "warm",
    }


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

    if any(not q.get("expected_path_hash") for q in queries):
        print("Patching manifest hashes...", flush=True)
        msha = patch_manifest(queries)
    else:
        hashes = [q["expected_path_hash"] for q in queries]
        msha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    print(f"manifest_sha256={msha[:16]}...", flush=True)

    info = {"os": platform.platform(), "python": sys.version, "cpu_logical": os.cpu_count()}
    info["frontier_workers"] = FRONTIER_WORKERS
    info["frontier_executor_scope"] = "process_shared"
    info["application_cache"] = "disabled"
    info["graph_id"] = GRAPH_ID
    info["backend_state"] = "warm"
    info["manifest_sha256"] = msha
    with (OUT_DIR / "environment.json").open("w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    with (OUT_DIR / "run_config.json").open("w") as f:
        json.dump({"clients":[1,8,32,64],"repeats":3,"warmup_s":15,"measurement_s":45,
                   "hop":HOP,"fanout":FANOUT,"cycle_policy":"path",
                   "manifest":str(MANIFEST),"graph_id":GRAPH_ID,"manifest_sha256":msha}, f, indent=2)

    print(f"\n=== C1 FORMAL SWEEP ===", flush=True)
    print(f"Systems: cassandra_naive, cassandra_parallel, neo4j", flush=True)
    print(f"Clients: [1, 8, 32, 64] x 3 repeats", flush=True)
    print(f"Warmup: 15s  Measurement: 45s", flush=True)
    print(f"Total trials: {3*4*3}=36", flush=True)

    # Clean output files
    for fn in ["correctness_spotcheck.jsonl","trial_summary.csv","trial_summary.jsonl",
               "run_order.jsonl","failures.jsonl","final_concurrency_summary.csv",
               "final_concurrency_summary.json"]:
        (OUT_DIR / fn).unlink(missing_ok=True)

    systems = ["cassandra_naive", "cassandra_parallel", "neo4j"]
    clients_list = [1, 8, 32, 64]
    repeats = 3
    warm, meas = 15, 45

    # Generate randomized run order
    rng_order = random.Random(42)
    run_order = []
    for cl in clients_list:
        for rep in range(repeats):
            sorder = list(systems)
            rng_order.shuffle(sorder)
            for s in sorder:
                run_order.append({"system": s, "clients": cl, "repeat": rep})
    with (OUT_DIR / "run_order.jsonl").open("w", encoding="utf-8") as f:
        for e in run_order:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    csv_path = OUT_DIR / "trial_summary.csv"
    jsonl_path = OUT_DIR / "trial_summary.jsonl"
    failures_path = OUT_DIR / "failures.jsonl"
    fields = ["system","clients","repeat","warmup_s","measure_s","queries","errors","error_rate",
              "QPS","mean_ms","p50_ms","p95_ms","p99_ms","qps_x_mean","frontier_workers","graph_id","cache","backend"]

    all_rows = []
    total = len(run_order)
    all_ok = True

    for idx, entry in enumerate(run_order):
        sys_name = entry["system"]
        cl = entry["clients"]
        rep = entry["repeat"]
        trial_no = idx + 1
        print(f"\n[{trial_no}/{total}] {sys_name} clients={cl} repeat={rep+1}", flush=True)
        row = run_trial(sys_name, cl, queries, warm, meas, rep)
        if row is None:
            failure = {"trial": trial_no, "system": sys_name, "clients": cl, "repeat": rep+1,
                       "reason": "zero queries"}
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            all_ok = False
            continue
        all_rows.append(row)

        # Write immediately
        write_csv = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if write_csv:
                w.writeheader()
            w.writerow(row)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Spotcheck
        if not spotcheck(sys_name, queries):
            failure = {"trial": trial_no, "system": sys_name, "clients": cl, "repeat": rep+1,
                       "reason": "spotcheck_hash_mismatch"}
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(f"  SPOTCHECK FAILED", flush=True)
            all_ok = False

        # Error check
        if row["error_rate"] > 0:
            failure = {"trial": trial_no, "system": sys_name, "clients": cl, "repeat": rep+1,
                       "reason": f"error_rate={row['error_rate']}"}
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            all_ok = False

        # QPS validation
        qxm = row["qps_x_mean"]
        expected = 1000 * cl
        margin = max(100 * cl, 500)
        if abs(qxm - expected) > margin:
            failure = {"trial": trial_no, "system": sys_name, "clients": cl, "repeat": rep+1,
                       "reason": f"qps*mean={qxm:.0f} expected={expected:.0f} margin={margin:.0f}"}
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(f"  QPS*MEAN OFF: {qxm:.0f} vs expected {expected:.0f}", flush=True)
            all_ok = False

    shutdown_backends()

    # Failures placeholder if empty
    if not failures_path.exists():
        with failures_path.open("w", encoding="utf-8") as f:
            f.write("")

    # Aggregate final summary
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["system"], r["clients"])].append(r)

    final_rows = []
    for (sn, cl), group in sorted(groups.items()):
        qps = sorted([t["QPS"] for t in group])
        mean_l = sorted([t["mean_ms"] for t in group])
        p50_l = sorted([t["p50_ms"] for t in group])
        p95_l = sorted([t["p95_ms"] for t in group])
        p99_l = sorted([t["p99_ms"] for t in group])
        err = sorted([t["error_rate"] for t in group])
        mid = len(group) // 2
        fr = {
            "system": sn, "clients": cl,
            "median_QPS": qps[mid], "median_mean_ms": mean_l[mid],
            "median_p50_ms": p50_l[mid], "median_p95_ms": p95_l[mid],
            "median_p99_ms": p99_l[mid], "median_error_rate": err[mid],
            "min_QPS": qps[0], "max_QPS": qps[-1], "repeats": len(group),
        }
        final_rows.append(fr)

    final_fields = ["system","clients","median_QPS","median_mean_ms","median_p50_ms",
                    "median_p95_ms","median_p99_ms","median_error_rate","min_QPS","max_QPS","repeats"]
    with (OUT_DIR / "final_concurrency_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=final_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(final_rows)
    with (OUT_DIR / "final_concurrency_summary.json").open("w") as f:
        json.dump(final_rows, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}", flush=True)
    print(f"FINAL CONCURRENCY SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    for r in final_rows:
        print(f"  {r['system']:20s} c={r['clients']:2d}  "
              f"QPS={r['median_QPS']:8.1f}  mean={r['median_mean_ms']:7.1f}ms  "
              f"p50={r['median_p50_ms']:7.1f}ms  p95={r['median_p95_ms']:8.1f}ms  "
              f"p99={r['median_p99_ms']:8.1f}ms  n={r['repeats']}", flush=True)

    print(f"\n{'ALL PASS' if all_ok else 'ISSUES FOUND — check failures.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
