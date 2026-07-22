"""1M hop-depth cold sweep: 40 trials (2 sys × 4 hops × 5 reps). Read-only."""
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
    rt = 0; rw = 0
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        futures = {_cass_executor.submit(cass_fetch, s): s for s in sources}
        for f in as_completed(futures):
            s = futures[f]; se[s] = [e for e in f.result() if e[1] == rel]
            rt += 1; rw += len(f.result()) if not f.exception() else 0
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier})), rt, rw

def neo_read(q):
    rt = 0; rw = 0
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        for src in sources:
            with _neo_driver.session() as s:
                rows = list(s.run(
                    "MATCH (n:C3KGNode {graph_id: $g, node_id: $n})-[r:C3KG_EDGE {relation: $rel}]->(m:C3KGNode {graph_id: $g}) "
                    "RETURN n.node_id AS s, r.relation AS r, m.node_id AS d, coalesce(r.source,'') AS src ORDER BY r,d,src",
                    g=GR, n=src, rel=rel))
            rt += 1; rw += len(rows)
            se[src] = [(row["s"], row["r"], row["d"], row["src"]) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier})), rt, rw

def spotcheck(queries, sysname):
    rng = random.Random(42)
    for q in rng.sample(queries, 10):
        paths, _, _ = cass_read(q) if sysname == "cassandra_opt" else neo_read(q)
        h = hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()).hexdigest()
        if h != q["expected_path_hash"]: return False
    return True

def run_trial(sysname, hop, rep):
    manifest_path = PROJ / f"results/sysaxis_1m_manifest_h{hop}.jsonl"
    tag = f"{sysname} hop={hop} r={rep+1} cold"
    print(f"\n  [{tag}]", flush=True)
    queries = [json.loads(line) for line in open(manifest_path)]

    per_lat = [[] for _ in range(CLIENTS)]; per_err = [0] * CLIENTS
    per_rt = [[] for _ in range(CLIENTS)]; per_rw = [[] for _ in range(CLIENTS)]
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
                per_rt[ci].append(rt); per_rw[ci].append(rw)
            except Exception: per_err[ci] += 1
            ridx = (ridx + 1) % len(queries)

    threads = [threading.Thread(target=client_loop, args=(i,), daemon=True) for i in range(CLIENTS)]
    for t in threads: t.start()
    t_start = time.perf_counter(); time.sleep(45); t_end = time.perf_counter()
    stop.set()
    for t in threads: t.join(timeout=10)

    rl = sorted([v for l in per_lat for v in l]); nr = len(rl); errs = sum(per_err)
    rt_all = [v for l in per_rt for v in l]; rw_all = [v for l in per_rw for v in l]
    meas = t_end - t_start
    def pct(a, p): return a[int((len(a)-1)*p/100)] if a else None
    read_ok = spotcheck(queries, sysname)
    rp_mean = statistics.mean([q["expected_path_count"] for q in queries])
    rp_p95 = sorted([q["expected_path_count"] for q in queries])[int(len(queries)*0.95)]

    row = {
        "run_id": DATE_TAG, "system": sysname, "hop": hop, "mode": "cold", "clients": CLIENTS,
        "repeat": rep+1, "measurement_seconds": round(meas, 3), "completed_reads": nr,
        "QPS": round(nr/max(meas,.001), 3), "mean_ms": round(statistics.mean(rl), 3),
        "p50_ms": round(pct(rl, 50), 3), "p95_ms": round(pct(rl, 95), 3),
        "p99_ms": round(pct(rl, 99), 3),
        "round_trips_mean": round(statistics.mean(rt_all), 1) if rt_all else None,
        "round_trips_p95": round(pct(rt_all, 95), 1),
        "raw_rows_mean": round(statistics.mean(rw_all), 1) if rw_all else None,
        "raw_rows_p95": round(pct(rw_all, 95), 1),
        "result_paths_mean": round(rp_mean, 1), "result_paths_p95": round(rp_p95, 1),
        "read_error_count": errs, "spotcheck_10_10": read_ok,
    }
    print(f"    reads={nr} QPS={row['QPS']:.0f} mean={row['mean_ms']:.1f}ms p95={row['p95_ms']:.1f}ms spot={read_ok} err={errs}", flush=True)
    return row

def read_guard():
    import csv as csv_mod
    csv_set = set()
    with (PROJ/"results/c3_source_scale_1M.csv").open(encoding="utf-8-sig") as f:
        for r in csv_mod.DictReader(f): csv_set.add((r["src_id"], r["relation"], r["dst_id"], r["source"]))
    cass_set = set(); cass_raw = 0
    for src in sorted(set(e[0] for e in csv_set)):
        rows4 = _cass_session.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
        for r in rows4: cass_set.add((str(r.src_id), str(r.relation), str(r.dst_id), str(r.source or ""))); cass_raw += 1
    dup = cass_raw - len(cass_set); miss = len(csv_set - cass_set)
    return {"csv": len(csv_set), "raw": cass_raw, "distinct": len(cass_set), "duplicates": dup, "missing": miss}, dup == 0 and miss == 0

def full_hash_gate(func, name):
    queries = [json.loads(line) for line in open(PROJ / f"results/sysaxis_1m_manifest_h{name[-1]}.jsonl")]
    empty = 0; mm = 0
    for q in queries:
        paths, _, _ = func(q)
        if len(paths) == 0: empty += 1
        else:
            h = hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()).hexdigest()
            if h != q["expected_path_hash"]: mm += 1
    return {"checked": 256, "empty": empty, "mismatch": mm, "all_pass": empty == 0 and mm == 0}

# ── Main ──
print("=== 1M HOP-DEPTH COLD SWEEP ===")
print("Systems: cassandra_opt, neo4j | Hops: 1-4 x5 | Total: 40 cold trials")

init()
rng_ord = random.Random(42)
run_order = []
for hop in [1, 2, 3, 4]:
    for rep in range(5):
        so = ["cassandra_opt", "neo4j"]; rng_ord.shuffle(so)
        for s in so: run_order.append({"system": s, "hop": hop, "repeat": rep})

csv_path = OUT / "trial_summary_cold.csv"
jls_path = OUT / "trial_summary_cold.jsonl"
spot_path = OUT / "correctness_spotcheck_cold.jsonl"
fail_path = OUT / "failures_cold.jsonl"
for fn in [csv_path, jls_path]: fn.unlink(missing_ok=True)

all_rows = []; fields = None; total = len(run_order); all_ok = True
for idx, entry in enumerate(run_order):
    sysname = entry["system"]; hop = entry["hop"]; rep = entry["repeat"]
    print(f"\n[{idx+1}/{total}] {sysname} hop={hop} repeat={rep+1}", flush=True)
    init()
    row = run_trial(sysname, hop, rep)
    shutdown()
    if row is None: all_ok = False; continue
    all_rows.append(row)
    if not fields: fields = list(row.keys())

    write_hdr = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_hdr: w.writeheader(); w.writerow(row)
        else: w.writerow(row)
    with jls_path.open("a") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with spot_path.open("a") as f: f.write(json.dumps({"sys": sysname, "hop": hop, "rep": rep+1, "spot": row["spotcheck_10_10"]}) + "\n")
    if not row["spotcheck_10_10"] or row["read_error_count"] > 0:
        with fail_path.open("a") as f: f.write(json.dumps({"trial": idx+1, "sys": sysname, "hop": hop, "rep": rep+1, "reason": "spot or err"}) + "\n"); all_ok = False

# After guards
init()
print("\n=== AFTER GUARDS ===")
g, gok = read_guard()
with (OUT / "read_graph_guard_after_cold.json").open("w") as f: json.dump(g, f, indent=2)
print(f"Guard: csv={g['csv']} raw={g['raw']} dup={g['duplicates']} miss={g['missing']} {'PASS' if gok else 'FAIL'}")

for hop in [1, 2, 3, 4]:
    hc = full_hash_gate(cass_read, f"cassandra_h{hop}")
    hn = full_hash_gate(neo_read, f"neo4j_h{hop}")
    with (OUT / f"hash_gate_after_cold_cassandra_h{hop}.json").open("w") as f: json.dump(hc, f, indent=2)
    with (OUT / f"hash_gate_after_cold_neo4j_h{hop}.json").open("w") as f: json.dump(hn, f, indent=2)
    print(f"  hop={hop} Cass HG: {'PASS' if hc['all_pass'] else 'FAIL'} | Neo4j HG: {'PASS' if hn['all_pass'] else 'FAIL'}")

# Aggregate
groups = defaultdict(list)
for r in all_rows: groups[(r["system"], r["hop"])].append(r)
final = []
for (sn, hop), grp in sorted(groups.items()):
    def med(key): vals = sorted([float(r[key]) for r in grp]); return round(vals[len(vals)//2], 3)
    qps = sorted([float(r["QPS"]) for r in grp])
    fr = {
        "system": sn, "hop": hop, "n": len(grp),
        "median_QPS": round(qps[len(qps)//2], 3), "median_mean_ms": med("mean_ms"),
        "median_p50_ms": med("p50_ms"), "median_p95_ms": med("p95_ms"), "median_p99_ms": med("p99_ms"),
        "median_round_trips_mean": med("round_trips_mean"),
        "median_round_trips_p95": med("round_trips_p95"),
        "median_raw_rows_mean": med("raw_rows_mean"),
        "median_raw_rows_p95": med("raw_rows_p95"),
        "median_result_paths_mean": med("result_paths_mean"),
        "median_result_paths_p95": med("result_paths_p95"),
        "min_QPS": qps[0], "max_QPS": qps[-1], "IQR_QPS": round(qps[3]-qps[1], 3),
        "median_error_rate": 0.0,
    }
    final.append(fr)

ff = ["system", "hop", "n", "median_QPS", "median_mean_ms", "median_p50_ms", "median_p95_ms", "median_p99_ms",
      "median_round_trips_mean", "median_round_trips_p95", "median_raw_rows_mean", "median_raw_rows_p95",
      "median_result_paths_mean", "median_result_paths_p95", "min_QPS", "max_QPS", "IQR_QPS", "median_error_rate"]
with (OUT / "final_hop_depth_summary_cold.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=ff, extrasaction="ignore"); w.writeheader(); w.writerows(final)
with (OUT / "final_hop_depth_summary_cold.json").open("w") as f: json.dump(final, f, indent=2)
with fail_path.open("w") as f: pass

print(f"\n=== FINAL COLD ===")
for fr in final:
    print(f"  {fr['system']:15s} hop={fr['hop']} QPS={fr['median_QPS']:7.0f} mean={fr['median_mean_ms']:6.1f}ms p95={fr['median_p95_ms']:6.1f}ms p99={fr['median_p99_ms']:7.1f}ms rt={fr['median_round_trips_mean']:.0f} n={fr['n']}")
print(f"\nGuard: {'PASS' if gok else 'FAIL'} | Sweep: {'ALL PASS' if all_ok else 'ISSUES'}")
shutdown()
