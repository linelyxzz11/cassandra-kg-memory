"""Concurrency sweep — read-only smoke test. Then full 120 trials."""
import csv, hashlib, json, os, random, statistics, sys, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from cassandra.cluster import Cluster
from neo4j import GraphDatabase

GR = "c3_scale_1M_seed42"; FAN = 20; HOP = 2; FW = 16
NEO_PWD = os.environ["NEO4J_PASSWORD"]
PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT  = PROJ / "reports/sysaxis_1m_concurrency_final"
MANIFEST = PROJ / "results/c3_manifest_scale_1m_h2.jsonl"
DATE_TAG = time.strftime("%Y%m%d")
OUT.mkdir(parents=True, exist_ok=True)

_cass_cluster = None; _cass_session = None; _cass_executor = None; _neo_driver = None

def init(fw=FW):
    global _cass_cluster, _cass_session, _cass_executor, _neo_driver
    _cass_cluster = Cluster(["127.0.0.1"], port=9042)
    _cass_session = _cass_cluster.connect("ai_memory")
    _cass_executor = ThreadPoolExecutor(max_workers=fw)
    _neo_driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", NEO_PWD))

def shutdown():
    _cass_executor.shutdown(wait=True); _cass_cluster.shutdown(); _neo_driver.close()

def cass_fetch(src):
    rows = _cass_session.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
    return [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]

def cass_read_opt(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier})
        se = {}
        futures = {_cass_executor.submit(cass_fetch, s): s for s in sources}
        for f in as_completed(futures):
            s = futures[f]; se[s] = [e for e in f.result() if e[1] == rel]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier}))

def cass_read_naive(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        for src in sources:
            se[src] = [e for e in cass_fetch(src) if e[1] == rel]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier}))

def neo_read(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        for src in sources:
            with _neo_driver.session() as s:
                rows = list(s.run(
                    "MATCH (n:C3KGNode {graph_id: $g, node_id: $n})-[r:C3KG_EDGE {relation: $rel}]->(m:C3KGNode {graph_id: $g}) "
                    "RETURN n.node_id AS s, r.relation AS r, m.node_id AS d, coalesce(r.source, '') AS src ORDER BY r, d, src",
                    g=GR, n=src, rel=rel))
            se[src] = [(row["s"], row["r"], row["d"], row["src"]) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier}))

def spotcheck(queries, sysname):
    rng = random.Random(42)
    for q in rng.sample(queries, 10):
        if sysname == "cassandra_opt": paths = cass_read_opt(q)
        elif sysname == "cassandra_naive": paths = cass_read_naive(q)
        else: paths = neo_read(q)
        h = hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()).hexdigest()
        if h != q["expected_path_hash"]: return False
    return True

def run_trial(sysname, clients, repeat_idx, mode, meas_s=45):
    tag = f"{sysname} c={clients} r={repeat_idx+1} {mode}"
    print(f"\n  [{tag}]", flush=True)
    queries = [json.loads(line) for line in open(MANIFEST)]

    per_lat = [[] for _ in range(clients)]; per_err = [0] * clients
    stop = threading.Event()

    def client_loop(ci):
        ridx = ci % len(queries)
        while not stop.is_set():
            q = queries[ridx]; t0 = time.perf_counter()
            try:
                if sysname == "cassandra_opt": cass_read_opt(q)
                elif sysname == "cassandra_naive": cass_read_naive(q)
                else: neo_read(q)
                per_lat[ci].append((time.perf_counter() - t0) * 1000)
            except Exception:
                per_err[ci] += 1
            ridx = (ridx + 1) % len(queries)

    threads = [threading.Thread(target=client_loop, args=(i,), daemon=True) for i in range(clients)]
    for t in threads: t.start()

    if mode == "warm":
        time.sleep(15)
        for l in per_lat: l.clear()

    t_start = time.perf_counter(); time.sleep(meas_s); t_end = time.perf_counter()
    stop.set()
    for t in threads: t.join(timeout=10)

    rl = sorted([v for l in per_lat for v in l]); nr = len(rl); errs = sum(per_err)
    meas_actual = t_end - t_start
    def pct(a, p): return a[int((len(a)-1)*p/100)] if a else None
    read_ok = spotcheck(queries, sysname)

    row = {
        "run_id": DATE_TAG, "system": sysname, "graph_id": GR,
        "clients": clients, "hop": HOP, "fanout": FAN, "cycle_policy": "path",
        "repeat": repeat_idx + 1, "mode": mode, "cold_mode": "process_cold" if mode == "cold" else "warm",
        "warmup_seconds": 15 if mode == "warm" else 0,
        "measurement_seconds": round(meas_actual, 3),
        "completed_reads": nr, "read_QPS": round(nr / max(meas_actual, 0.001), 3),
        "read_mean_ms": round(statistics.mean(rl), 3) if rl else None,
        "read_p50_ms": round(pct(rl, 50), 3), "read_p95_ms": round(pct(rl, 95), 3),
        "read_p99_ms": round(pct(rl, 99), 3),
        "read_error_count": errs, "error_rate": round(errs / max(nr, 1), 6),
        "cache_enabled": False, "cache_hit_rate": 0,
        "effective_latency_ms": round(statistics.mean(rl), 3) if rl else None,
        "frontier_workers": FW if sysname == "cassandra_opt" else (1 if sysname == "cassandra_naive" else 0),
        "relation_index_enabled": False, "backend_state": mode,
        "read_hash_spotcheck_passed": read_ok,
    }
    qps = f"{row['read_QPS']:.0f}" if row['read_QPS'] else "N/A"
    print(f"    reads={nr} QPS={qps} mean={row['read_mean_ms']:.1f}ms p95={row['read_p95_ms']:.1f}ms p99={row['read_p99_ms']:.1f}ms spot={read_ok} err={errs}", flush=True)
    return row


# ── Smoke ──
print("=== SMOKE TEST ===")
init()
queries = [json.loads(line) for line in open(MANIFEST)]

smoke_rows = []
for sysname in ["cassandra_opt", "neo4j"]:
    row = run_trial(sysname, 32, 0, "cold", 20)
    if row is None: shutdown(); exit(1)
    smoke_rows.append(row)

fields = list(smoke_rows[0].keys())
csv_path = OUT / "trial_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(smoke_rows)
with (OUT / "trial_summary.jsonl").open("w") as f:
    for r in smoke_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with (OUT / "correctness_spotcheck.jsonl").open("w") as f:
    for r in smoke_rows: f.write(json.dumps({"system": r["system"], "spotcheck": r["read_hash_spotcheck_passed"]}) + "\n")
with (OUT / "failures.jsonl").open("w") as f: pass

# Quick after-smoke guard
from cassandra.cluster import Cluster as C
import csv as csv_mod
c2 = C(["127.0.0.1"], port=9042); s2 = c2.connect("ai_memory")
csv_set = set()
with (PROJ / "results/c3_source_scale_1M.csv").open(encoding="utf-8-sig") as f:
    for r in csv_mod.DictReader(f): csv_set.add((r["src_id"], r["relation"], r["dst_id"], r["source"]))
cass_set = set(); cass_raw = 0
for src in sorted(set(e[0] for e in csv_set)):
    rows3 = s2.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
    for r in rows3: cass_set.add((str(r.src_id), str(r.relation), str(r.dst_id), str(r.source or ""))); cass_raw += 1
dup = cass_raw - len(cass_set); miss = len(csv_set - cass_set)
sg = {"csv": len(csv_set), "raw": cass_raw, "distinct": len(cass_set), "duplicates": dup, "missing": miss}
with (OUT / "read_graph_guard_after_smoke.json").open("w") as f: json.dump(sg, f, indent=2)
print(f"\nSmoke guard: raw={cass_raw} dup={dup} miss={miss} {'PASS' if dup==0 and miss==0 else 'FAIL'}")
c2.shutdown(); shutdown()
print("\n=== SMOKE COMPLETE ===")
for r in smoke_rows:
    print(f"  {r['system']}: QPS={r['read_QPS']:.0f} mean={r['read_mean_ms']:.1f}ms p95={r['read_p95_ms']:.1f}ms p99={r['read_p99_ms']:.1f}ms spot={r['read_hash_spotcheck_passed']} err={r['read_error_count']}")
