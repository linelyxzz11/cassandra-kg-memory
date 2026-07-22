"""Hop-depth smoke: 4 trials (cassandra+neo4j × hop=1,4 × cold × 20s). Read-only."""
import csv, hashlib, json, random, statistics, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from cassandra.cluster import Cluster
from neo4j import GraphDatabase

GR = "c3_scale_1M_seed42"; FAN = 20; CLIENTS = 8; FW = 16
PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT  = PROJ / "reports/sysaxis_1m_hop_depth_final"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = time.strftime("%Y%m%d")

_cass_session = None; _cass_executor = None; _neo_driver = None

def init(fw=FW):
    global _cass_session, _cass_executor, _neo_driver
    c = Cluster(["127.0.0.1"], port=9042)
    _cass_session = c.connect("ai_memory")
    _cass_executor = ThreadPoolExecutor(max_workers=fw)
    _neo_driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "password123"))

def shutdown():
    _cass_executor.shutdown(wait=True); _cass_session.cluster.shutdown(); _neo_driver.close()

def cass_fetch(src):
    rows = _cass_session.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
    return [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]

def cass_read(q):
    round_trips = 0; raw_rows = 0
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        futures = {_cass_executor.submit(cass_fetch, s): s for s in sources}
        for f in as_completed(futures):
            s = futures[f]; se[s] = [e for e in f.result() if e[1] == rel]
            round_trips += 1; raw_rows += len(f.result()) if not f.exception() else 0
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier})), round_trips, raw_rows

def neo_read(q):
    round_trips = 0; raw_rows = 0
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        for src in sources:
            with _neo_driver.session() as s:
                rows = list(s.run(
                    "MATCH (n:C3KGNode {graph_id: $g, node_id: $n})-[r:C3KG_EDGE {relation: $rel}]->(m:C3KGNode {graph_id: $g}) "
                    "RETURN n.node_id AS s, r.relation AS r, m.node_id AS d, coalesce(r.source,'') AS src ORDER BY r,d,src",
                    g=GR, n=src, rel=rel))
            round_trips += 1; raw_rows += len(rows)
            se[src] = [(row["s"], row["r"], row["d"], row["src"]) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier})), round_trips, raw_rows

def spotcheck(queries, sysname):
    rng = random.Random(42)
    for q in rng.sample(queries, 10):
        paths, _, _ = cass_read(q) if sysname == "cassandra_opt" else neo_read(q)
        h = hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()).hexdigest()
        if h != q["expected_path_hash"]: return False
    return True

def run_trial(sysname, hop):
    manifest_path = PROJ / f"results/sysaxis_1m_manifest_h{hop}.jsonl"
    tag = f"{sysname} hop={hop} cold"
    print(f"\n  [{tag}]", flush=True)
    queries = [json.loads(line) for line in open(manifest_path)]

    per_lat = [[] for _ in range(CLIENTS)]; per_err = [0] * CLIENTS
    per_rt = [[] for _ in range(CLIENTS)]; per_raw = [[] for _ in range(CLIENTS)]
    stop = threading.Event()

    def client_loop(ci):
        ridx = ci % len(queries)
        while not stop.is_set():
            q = queries[ridx]; t0 = time.perf_counter()
            try:
                if sysname == "cassandra_opt":
                    _, rt, rw = cass_read(q)
                else:
                    _, rt, rw = neo_read(q)
                per_lat[ci].append((time.perf_counter() - t0) * 1000)
                per_rt[ci].append(rt); per_raw[ci].append(rw)
            except Exception: per_err[ci] += 1
            ridx = (ridx + 1) % len(queries)

    threads = [threading.Thread(target=client_loop, args=(i,), daemon=True) for i in range(CLIENTS)]
    for t in threads: t.start()
    t_start = time.perf_counter(); time.sleep(20); t_end = time.perf_counter()
    stop.set()
    for t in threads: t.join(timeout=10)

    rl = sorted([v for l in per_lat for v in l]); nr = len(rl); errs = sum(per_err)
    rt_list = [v for l in per_rt for v in l]; rw_list = [v for l in per_raw for v in l]
    meas = t_end - t_start
    def pct(a, p): return a[int((len(a)-1)*p/100)] if a else None

    read_ok = spotcheck(queries, sysname)
    paths_mean = statistics.mean([q["expected_path_count"] for q in queries])
    paths_p95 = sorted([q["expected_path_count"] for q in queries])[int(len(queries)*0.95)]

    row = {
        "system": sysname, "hop": hop, "mode": "cold", "clients": CLIENTS,
        "measurement_seconds": round(meas, 3), "completed_reads": nr,
        "QPS": round(nr/max(meas,.001), 3), "read_mean_ms": round(statistics.mean(rl), 3),
        "read_p50_ms": round(pct(rl, 50), 3), "read_p95_ms": round(pct(rl, 95), 3),
        "read_p99_ms": round(pct(rl, 99), 3),
        "round_trips_mean": round(statistics.mean(rt_list), 1) if rt_list else None,
        "round_trips_p95": round(pct(rt_list, 95), 1),
        "raw_rows_mean": round(statistics.mean(rw_list), 1) if rw_list else None,
        "raw_rows_p95": round(pct(rw_list, 95), 1),
        "result_paths_mean": round(paths_mean, 1), "result_paths_p95": round(paths_p95, 1),
        "spotcheck_10_10": read_ok, "read_error_count": errs,
    }
    print(f"    reads={nr} QPS={row['QPS']:.0f} mean={row['read_mean_ms']:.1f}ms p95={row['read_p95_ms']:.1f}ms "
          f"rt={row['round_trips_mean']:.0f} rw={row['raw_rows_mean']:.0f} spot={read_ok} err={errs}", flush=True)
    return row

# ── Smoke ──
print("=== HOP-DEPTH SMOKE ===")
init()
smoke_rows = []
for hop in [1, 4]:
    for sysname in ["cassandra_opt", "neo4j"]:
        row = run_trial(sysname, hop)
        smoke_rows.append(row)

# Save
csv_path = OUT / "trial_summary.csv"
fields = list(smoke_rows[0].keys())
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(smoke_rows)
with (OUT / "trial_summary.jsonl").open("w") as f:
    for r in smoke_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with (OUT / "failures.jsonl").open("w") as f: pass

# After-smoke guard
import csv as csv_mod
csv_set = set()
with (PROJ / "results/c3_source_scale_1M.csv").open(encoding="utf-8-sig") as f:
    for r in csv_mod.DictReader(f): csv_set.add((r["src_id"], r["relation"], r["dst_id"], r["source"]))
cass_set = set(); cass_raw = 0
for src in sorted(set(e[0] for e in csv_set)):
    rows4 = _cass_session.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
    for r in rows4: cass_set.add((str(r.src_id), str(r.relation), str(r.dst_id), str(r.source or ""))); cass_raw += 1
dup = cass_raw - len(cass_set); miss = len(csv_set - cass_set)
sg = {"csv": len(csv_set), "raw": cass_raw, "distinct": len(cass_set), "duplicates": dup, "missing": miss}
with (OUT / "read_graph_guard_after_smoke.json").open("w") as f: json.dump(sg, f, indent=2)
print(f"\nSmoke guard: raw={cass_raw} dup={dup} miss={miss} {'PASS' if dup==0 else 'FAIL'}")

shutdown()
print(f"\n=== SMOKE RESULTS ===")
for r in smoke_rows:
    print(f"  {r['system']:15s} hop={r['hop']} QPS={r['QPS']:.0f} mean={r['read_mean_ms']:.1f}ms p95={r['read_p95_ms']:.1f}ms p99={r['read_p99_ms']:.1f}ms rt={r['round_trips_mean']:.0f} rw={r['raw_rows_mean']:.0f} spot={r['spotcheck_10_10']}")
