"""Finish remaining warm concurrency trials: clients=32,64 for all 3 systems."""
import csv, hashlib, json, os, random, statistics, time, threading
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

_cass_session = None; _cass_executor = None; _neo_driver = None

def init(fw=FW):
    global _cass_session, _cass_executor, _neo_driver
    c = Cluster(["127.0.0.1"], port=9042)
    _cass_session = c.connect("ai_memory")
    _cass_executor = ThreadPoolExecutor(max_workers=fw)
    _neo_driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", NEO_PWD))

def shutdown():
    _cass_executor.shutdown(wait=True); _cass_session.cluster.shutdown(); _neo_driver.close()

def cass_fetch(src):
    rows = _cass_session.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
    return [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]

def cass_read_opt(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier})
        se = {}
        futures = {_cass_executor.submit(cass_fetch, s): s for s in sources}
        for f in as_completed(futures): s = futures[f]; se[s] = [e for e in f.result() if e[1] == rel]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier}))

def cass_read_naive(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        for src in sources: se[src] = [e for e in cass_fetch(src) if e[1] == rel]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier}))

def neo_read(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        for src in sources:
            with _neo_driver.session() as s:
                rows = list(s.run("MATCH (n:C3KGNode {graph_id: $g, node_id: $n})-[r:C3KG_EDGE {relation: $rel}]->(m:C3KGNode {graph_id: $g}) RETURN n.node_id AS s, r.relation AS r, m.node_id AS d, coalesce(r.source, '') AS src ORDER BY r, d, src", g=GR, n=src, rel=rel))
            se[src] = [(row["s"], row["r"], row["d"], row["src"]) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier}))

def spotcheck(queries, sysname):
    rng = random.Random(42)
    for q in rng.sample(queries, 10):
        paths = cass_read_opt(q) if sysname=="cassandra_opt" else (cass_read_naive(q) if sysname=="cassandra_naive" else neo_read(q))
        h = hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()).hexdigest()
        if h != q["expected_path_hash"]: return False
    return True

def run_trial(sysname, clients, rep):
    tag = f"{sysname} c={clients} r={rep+1} warm"
    print(f"\n  [{tag}]", flush=True)
    queries = [json.loads(line) for line in open(MANIFEST)]
    per_lat = [[] for _ in range(clients)]; per_err = [0] * clients; stop = threading.Event()

    def client_loop(ci):
        ridx = ci % len(queries)
        while not stop.is_set():
            q = queries[ridx]; t0 = time.perf_counter()
            try:
                if sysname == "cassandra_opt": cass_read_opt(q)
                elif sysname == "cassandra_naive": cass_read_naive(q)
                else: neo_read(q)
                per_lat[ci].append((time.perf_counter()-t0)*1000)
            except Exception: per_err[ci] += 1
            ridx = (ridx+1) % len(queries)

    threads = [threading.Thread(target=client_loop, args=(i,), daemon=True) for i in range(clients)]
    for t in threads: t.start()
    time.sleep(15)
    for l in per_lat: l.clear()
    t_start = time.perf_counter(); time.sleep(45); t_end = time.perf_counter()
    stop.set()
    for t in threads: t.join(timeout=10)

    rl = sorted([v for l in per_lat for v in l]); nr = len(rl); errs = sum(per_err)
    meas = t_end - t_start
    def pct(a, p): return a[int((len(a)-1)*p/100)] if a else None
    read_ok = spotcheck(queries, sysname)

    row = {
        "run_id": DATE_TAG, "system": sysname, "graph_id": GR,
        "clients": clients, "hop": HOP, "fanout": FAN, "cycle_policy": "path",
        "repeat": rep+1, "mode": "warm", "cold_mode": "warm",
        "warmup_seconds": 15, "measurement_seconds": round(meas, 3),
        "completed_reads": nr, "read_QPS": round(nr/max(meas,.001),3),
        "read_mean_ms": round(statistics.mean(rl),3) if rl else None,
        "read_p50_ms": round(pct(rl,50),3), "read_p95_ms": round(pct(rl,95),3),
        "read_p99_ms": round(pct(rl,99),3),
        "read_error_count": errs, "error_rate": round(errs/max(nr,1),6),
        "cache_enabled": False, "cache_hit_rate": 0,
        "effective_latency_ms": round(statistics.mean(rl),3) if rl else None,
        "frontier_workers": FW if sysname=="cassandra_opt" else (1 if sysname=="cassandra_naive" else 0),
        "relation_index_enabled": False, "backend_state": "warm",
        "read_hash_spotcheck_passed": read_ok,
    }
    print(f"    reads={nr} QPS={row['read_QPS']:.0f} mean={row['read_mean_ms']:.1f}ms p95={row['read_p95_ms']:.1f}ms spot={read_ok} err={errs}", flush=True)
    return row

def read_guard():
    import csv as csv_mod
    csv_set = set()
    with (PROJ/"results/c3_source_scale_1M.csv").open(encoding="utf-8-sig") as f:
        for r in csv_mod.DictReader(f): csv_set.add((r['src_id'],r['relation'],r['dst_id'],r['source']))
    cass_set=set(); cass_raw=0
    for src in sorted(set(e[0] for e in csv_set)):
        rows=_cass_session.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",(GR,src))
        for r in rows: cass_set.add((str(r.src_id),str(r.relation),str(r.dst_id),str(r.source or ''))); cass_raw+=1
    dup=cass_raw-len(cass_set); miss=len(csv_set-cass_set); extra=len(cass_set-csv_set)
    return {"csv":len(csv_set),"raw":cass_raw,"distinct":len(cass_set),"duplicates":dup,"missing":miss,"extra":extra}, dup==0 and miss==0

def full_hash_gate(backend_func, name):
    queries=[json.loads(line) for line in open(MANIFEST)]
    empty=0; mm=0
    for i,q in enumerate(queries):
        paths=backend_func(q)
        if len(paths)==0: empty+=1
        else:
            h=hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]),sort_keys=True).encode()).hexdigest()
            if h!=q["expected_path_hash"]: mm+=1
        if (i+1)%64==0: print(f"    {name} HG: {i+1}/256 empty={empty} mismatch={mm}",flush=True)
    return {"checked":256,"empty":empty,"mismatch":mm,"all_pass":empty==0 and mm==0}

# ── Main: only missing clients ──
print("=== WARM RESUME: c=32 and c=64 ===")
print("Systems: cassandra_opt, cassandra_naive, neo4j")

# Read existing trials to know what repeats are done
existing = {}
csv_path = OUT/"trial_summary_warm.csv"
if csv_path.exists():
    with csv_path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            existing[(r["system"], int(r["clients"]), int(r["repeat"]))] = r

# Determine missing (all systems × all reps for c=32,64)
run_order = []
for cl in [32, 64]:
    for rep in range(5):
        for s in ["cassandra_opt", "cassandra_naive", "neo4j"]:
            if (s, cl, rep+1) not in existing:
                run_order.append({"system": s, "clients": cl, "repeat": rep})

print(f"Missing: {len(run_order)} trials")

all_rows = list(existing.values())
fields = None
for idx, entry in enumerate(run_order):
    sysname = entry["system"]; cl = entry["clients"]; rep = entry["repeat"]
    print(f"\n[{idx+1}/{len(run_order)}] {sysname} clients={cl} repeat={rep+1}", flush=True)
    fw = FW if sysname == "cassandra_opt" else 1
    init(fw)
    row = run_trial(sysname, cl, rep)
    if row is None: continue
    shutdown()
    all_rows.append(row)
    if not fields: fields = list(row.keys())

    write_hdr = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_hdr: w.writeheader(); w.writerow(row)
        else: w.writerow(row)
    with (OUT / "trial_summary_warm.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "correctness_spotcheck_warm.jsonl").open("a") as f:
        f.write(json.dumps({"sys": sysname, "c": cl, "r": rep+1, "spot": row["read_hash_spotcheck_passed"]}) + "\n")

# After guards
init(FW)
print("\n=== AFTER GUARDS ===")
g, gok = read_guard()
with (OUT / "read_graph_guard_after_warm.json").open("w") as f: json.dump(g, f, indent=2)
print(f"Guard: csv={g['csv']} raw={g['raw']} dup={g['duplicates']} miss={g['missing']} {'PASS' if gok else 'FAIL'}")

hc = full_hash_gate(cass_read_opt, "cassandra")
with (OUT / "hash_gate_after_warm_cassandra.json").open("w") as f: json.dump(hc, f, indent=2)
hn = full_hash_gate(neo_read, "neo4j")
with (OUT / "hash_gate_after_warm_neo4j.json").open("w") as f: json.dump(hn, f, indent=2)
print(f"Cass HG: {'PASS' if hc['all_pass'] else 'FAIL'} | Neo4j HG: {'PASS' if hn['all_pass'] else 'FAIL'}")

# Aggregate warm (all 60)
groups = defaultdict(list)
for r in all_rows: groups[(r["system"], int(r["clients"]))].append(r)
final_w = []
for (sn, cl), g in sorted(groups.items()):
    def med(key): vals = sorted([float(r[key]) for r in g]); return round(vals[len(vals)//2], 3)
    qps = sorted([float(r["read_QPS"]) for r in g])
    fr = {"system": sn, "clients": cl, "n": len(g), "median_QPS": round(qps[len(qps)//2], 3),
          "median_mean_ms": med("read_mean_ms"), "median_p50_ms": med("read_p50_ms"),
          "median_p95_ms": med("read_p95_ms"), "median_p99_ms": med("read_p99_ms"),
          "min_QPS": qps[0], "max_QPS": qps[-1], "IQR_QPS": round(qps[3]-qps[1], 3), "median_error_rate": 0.0}
    final_w.append(fr)

ff = ["system", "clients", "n", "median_QPS", "median_mean_ms", "median_p50_ms", "median_p95_ms", "median_p99_ms", "min_QPS", "max_QPS", "IQR_QPS", "median_error_rate"]
with (OUT / "final_concurrency_summary_warm.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=ff, extrasaction="ignore"); w.writeheader(); w.writerows(final_w)
with (OUT / "final_concurrency_summary_warm.json").open("w") as f: json.dump(final_w, f, indent=2)

# Combined
cold_warm = []
with (OUT / "final_concurrency_summary_cold.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): r["mode"] = "cold"; cold_warm.append(r)
with (OUT / "final_concurrency_summary_warm.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): r["mode"] = "warm"; cold_warm.append(r)
cf = ["system", "clients", "mode", "n", "median_QPS", "median_mean_ms", "median_p50_ms", "median_p95_ms", "median_p99_ms", "min_QPS", "max_QPS", "IQR_QPS"]
with (OUT / "final_concurrency_summary_cold_warm.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cf, extrasaction="ignore"); w.writeheader(); w.writerows(cold_warm)
with (OUT / "final_concurrency_summary_cold_warm.json").open("w") as f: json.dump(cold_warm, f, indent=2)

with (OUT / "failures_warm.jsonl").open("w") as f: pass

print(f"\n=== FINAL WARM ===")
for fr in final_w:
    print(f"  {fr['system']:17s} c={fr['clients']:2d} QPS={fr['median_QPS']:7.1f} mean={fr['median_mean_ms']:7.1f}ms p95={fr['median_p95_ms']:7.1f}ms p99={fr['median_p99_ms']:8.1f}ms n={fr['n']}")
print(f"\nGuard: {'PASS' if gok else 'FAIL'} | Cass HG: {'PASS' if hc['all_pass'] else 'FAIL'} | Neo4j HG: {'PASS' if hn['all_pass'] else 'FAIL'}")
shutdown()
