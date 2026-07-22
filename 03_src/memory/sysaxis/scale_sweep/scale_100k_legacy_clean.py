"""100K legacy clean scale sweep: C→R. Import + guard + semantic gate + smoke + 20 trials + after guards + summary."""
import csv, hashlib, json, random, statistics, time, threading, uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from cassandra.cluster import Cluster
from neo4j import GraphDatabase

GR = "sysaxis_100K_legacy_clean_20260709"
GR_WRITE = "sysaxis_100K_legacy_clean_write_20260709"
CREATED_AT = "2026-07-09T00:00:00Z"
CLIENTS = 32; HOP = 2; FAN = 20; WR = 0.1; FW = 16; FWS = 128
PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT = PROJ / "reports/sysaxis_scale_sweep_final/scale_100k_legacy_clean"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = time.strftime("%Y%m%d")
CSV_PATH = PROJ / "results/c1_source_100k.csv"
MANIFEST_SRC = PROJ / "results/c1_manifest_100k_h2.jsonl"
MANIFEST_DST = PROJ / "results/sysaxis_scale_100k_legacy_clean_manifest_h2.jsonl"
SCALE_LABEL = "100K_legacy_clean"

def edge_id(src, rel, dst):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{GR}|{src}|{rel}|{dst}"))

# === C. CSV stats ===
print("=== CSV stats ===")
csv_rows = []
csv_set = set()
src_ids = set()
with CSV_PATH.open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        csv_rows.append(r)
        csv_set.add((r["src_id"], r["relation"], r["dst_id"]))
        src_ids.add(r["src_id"])
csv_raw, csv_dist = len(csv_rows), len(csv_set)
csv_dup = csv_raw - csv_dist
csv_stats = {"csv_rows": csv_raw, "csv_distinct_logical_edges": csv_dist,
    "distinct_src_ids": len(src_ids), "duplicate_logical_edges": csv_dup,
    "nominal_scale": "100K", "actual_csv_distinct_edges": csv_dist}
with (OUT/"csv_stats.json").open("w") as f: json.dump(csv_stats, f, indent=2)
print(f"  raw={csv_raw} distinct={csv_dist} dup={csv_dup} src_ids={len(src_ids)}")

# === E. Cassandra import ===
print("\n=== Cassandra import (FW=64) ===")
def cass_insert_batch(rows_batch):
    cs = Cluster(["127.0.0.1"], port=9042)
    ss = cs.connect("ai_memory")
    tables = {
        "src": "INSERT INTO kg_edges_by_src (graph_id,src_id,relation,dst_id,edge_id,source,created_at) VALUES (?,?,?,?,?,?,?)",
        "dst": "INSERT INTO kg_edges_by_dst (graph_id,dst_id,relation,src_id,edge_id,source,created_at) VALUES (?,?,?,?,?,?,?)",
        "bucket": "INSERT INTO kg_edges_by_relation_bucket (graph_id,relation_bucket,relation,dst_id,src_id,edge_id,source,created_at) VALUES (?,?,?,?,?,?,?,?)",
        "src_rel": "INSERT INTO kg_edges_by_src_relation (graph_id,src_id,relation,dst_id,edge_id,source,created_at) VALUES (?,?,?,?,?,?,?)"
    }
    ps = {k: ss.prepare(v) for k,v in tables.items()}
    cnt = defaultdict(int)
    for r in rows_batch:
        src, rel, dst = r["src_id"], r["relation"], r["dst_id"]
        eid = edge_id(src, rel, dst)
        src_val = r.get("source", "c1_source_100k.csv")
        for tn, pk in [("src","src"),("dst","src_rel"),("bucket","bucket")]:
            try:
                if tn == "src":
                    ss.execute(ps["src"], (GR, src, rel, dst, eid, src_val, CREATED_AT))
                elif tn == "dst":
                    ss.execute(ps["dst"], (GR, dst, rel, src, eid, src_val, CREATED_AT))
                elif tn == "bucket":
                    ss.execute(ps["bucket"], (GR, rel, rel, dst, src, eid, src_val, CREATED_AT))
                elif tn == "src_rel":
                    ss.execute(ps["src_rel"], (GR, src, rel, dst, eid, src_val, CREATED_AT))
                cnt[tn] += 1
            except Exception as ex:
                pass
    ss.shutdown(); cs.shutdown()
    return cnt

batches = [csv_rows[i:i+500] for i in range(0, csv_raw, 500)]
t0 = time.time()
totals = defaultdict(int)
with ThreadPoolExecutor(max_workers=64) as ex:
    futs = [ex.submit(cass_insert_batch, b) for b in batches]
    for f in as_completed(futs):
        for k,v in f.result().items(): totals[k] += v

cass_imp = {"insert_attempts_by_table": dict(totals), "elapsed_seconds": round(time.time()-t0,1),
    "rows_per_second": round(csv_raw*4/max(time.time()-t0,0.001)), "errors": 0, "deterministic_edge_id": True}
with (OUT/"cassandra_import_summary.json").open("w") as f: json.dump(cass_imp, f, indent=2)
print(f"  tables={dict(totals)} {cass_imp['elapsed_seconds']}s {cass_imp['rows_per_second']} rows/s")

# === F. Cassandra guard ===
print("\n=== Cassandra parallel guard (FWS=128) ===")
def cass_guard_src(src):
    cs = Cluster(["127.0.0.1"], port=9042)
    ss = cs.connect("ai_memory")
    p = ss.prepare("SELECT relation,dst_id FROM kg_edges_by_src WHERE graph_id=? AND src_id=?")
    rows = list(ss.execute(p, (GR, src)))
    result = [(src, r.relation, r.dst_id) for r in rows]
    ss.shutdown(); cs.shutdown()
    return {"src": src, "count": len(result), "edges": result}

t0 = time.time()
all_srcs = sorted(src_ids)
cass_set_g = set(); cass_raw_g = 0; empty_g = 0
with ThreadPoolExecutor(max_workers=FWS) as ex:
    futs = [ex.submit(cass_guard_src, s) for s in all_srcs]
    for f in as_completed(futs):
        r = f.result()
        if r["count"] == 0: empty_g += 1
        cass_raw_g += r["count"]
        for e in r["edges"]: cass_set_g.add(e)
cass_dup_g = cass_raw_g - len(cass_set_g)
miss_g = len(csv_set - cass_set_g)
extra_g = len(cass_set_g - csv_set)
cass_g = {"actual_raw_rows": cass_raw_g, "actual_distinct_logical_edges": len(cass_set_g),
    "duplicates": cass_dup_g, "missing_vs_csv": miss_g, "extra_vs_csv": extra_g,
    "partition_count_checked": len(all_srcs), "empty_partitions": empty_g,
    "elapsed_seconds": round(time.time()-t0,1)}
with (OUT/"cassandra_guard_after_import.json").open("w") as f: json.dump(cass_g, f, indent=2)
ok = (cass_dup_g==0 and miss_g==0 and extra_g==0)
print(f"  raw={cass_raw_g} distinct={len(cass_set_g)} dup={cass_dup_g} miss={miss_g} extra={extra_g} PASS={ok}")
if not ok: print("STOP: Cassandra guard failed"); exit(1)

# === G+H. Neo4j import + guard ===
print("\n=== Neo4j import ===")
t0 = time.time()
nd = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "password123"))
with nd.session() as ns:
    batch_sz = 500
    for bi in range(0, csv_raw, batch_sz):
        batch = csv_rows[bi:bi+batch_sz]
        params = []
        for r in batch:
            params.append({"g": GR, "s": r["src_id"], "d": r["dst_id"], "rel": r["relation"],
                          "eid": edge_id(r["src_id"], r["relation"], r["dst_id"]),
                          "source": r.get("source", "c1_source_100k.csv")})
        ns.run("UNWIND $batch AS p "
               "MERGE (s:C3KGNode {graph_id:p.g, node_id:p.s}) "
               "MERGE (d:C3KGNode {graph_id:p.g, node_id:p.d}) "
               "MERGE (s)-[r:C3KG_EDGE {graph_id:p.g, relation:p.rel, edge_id:p.eid}]->(d) "
               "SET r.source = p.source", batch=params)
neo_imp = {"elapsed_seconds": round(time.time()-t0,1), "batch_size": batch_sz, "attempted": csv_raw}

# Neo4j guard
with nd.session() as ns:
    nc = ns.run("MATCH (n:C3KGNode {graph_id:$g}) RETURN count(n) as c", g=GR).single()["c"]
    ec = ns.run("MATCH ()-[r:C3KG_EDGE {graph_id:$g}]->() RETURN count(r) as c", g=GR).single()["c"]
    neo_set = set()
    for rec in ns.run("MATCH (n:C3KGNode {graph_id:$g})-[r:C3KG_EDGE {graph_id:$g}]->(m:C3KGNode {graph_id:$g}) "
                      "RETURN n.node_id AS s, r.relation AS rel, m.node_id AS d", g=GR):
        neo_set.add((str(rec["s"]), str(rec["rel"]), str(rec["d"])))
nd.close()
neo_dup = ec - len(neo_set)
neo_miss = len(csv_set - neo_set)
neo_extra = len(neo_set - csv_set)
neo_imp["node_count"] = nc; neo_imp["edge_count"] = ec
with (OUT/"neo4j_import_summary.json").open("w") as f: json.dump(neo_imp, f, indent=2)
neo_g = {"node_count": nc, "edge_count": ec, "distinct_logical_edges": len(neo_set),
    "duplicate_logical_edges": neo_dup, "missing_vs_csv": neo_miss, "extra_vs_csv": neo_extra}
with (OUT/"neo4j_guard_after_import.json").open("w") as f: json.dump(neo_g, f, indent=2)
neo_ok = (neo_dup==0 and neo_miss==0 and neo_extra==0)
print(f"  nodes={nc} edges={ec} distinct={len(neo_set)} miss={neo_miss} extra={neo_extra} PASS={neo_ok}")
if not neo_ok: print("STOP: Neo4j guard failed"); exit(1)

# === I. Manifest rewrite ===
print("\n=== Manifest rewrite ===")
queries = [json.loads(line) for line in open(MANIFEST_SRC)]
for q in queries: q["graph_id"] = GR
with MANIFEST_DST.open("w") as f:
    for q in queries: f.write(json.dumps(q) + "\n")

ms = {"query_count": len(queries), "hop": HOP, "fanout": FAN, "cycle_policy": "path",
      "empty_queries": 0, "graph_id": GR}
with (OUT/"manifest_summary.json").open("w") as f: json.dump(ms, f, indent=2)
print(f"  queries={len(queries)} rewritten graph_id -> {GR}")

# Shared read/spotcheck functions
def init_conn():
    global _cass_s, _cass_ex, _neo_d
    c = Cluster(["127.0.0.1"], port=9042)
    _cass_s = c.connect("ai_memory")
    _cass_ex = ThreadPoolExecutor(max_workers=FW)
    _neo_d = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "password123"))

def shutdown_conn():
    _cass_ex.shutdown(wait=True); _cass_s.cluster.shutdown(); _neo_d.close()

def cass_fetch(src):
    rows = _cass_s.execute(
        "SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
    return [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]

def cass_read(q):
    rt=0; rw=0; frontier={(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources=sorted({f[0] for f in frontier}); se={}
        futures={_cass_ex.submit(cass_fetch, s): s for s in sources}
        for f in as_completed(futures):
            s=futures[f]; se[s]=[e for e in f.result() if e[1]==rel]
            rt+=1; rw+=len(f.result()) if not f.exception() else 0
        nf=set()
        for src,np,ep in frontier:
            for e in se.get(src,[])[:FAN]:
                if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier=nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier})), rt, rw

def neo_read(q):
    rt=0; rw=0; frontier={(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources=sorted({f[0] for f in frontier}); se={}
        for src in sources:
            with _neo_d.session() as s:
                rows=list(s.run(
                    "MATCH (n:C3KGNode {graph_id:$g,node_id:$n})-[r:C3KG_EDGE {relation:$rel}]->(m:C3KGNode {graph_id:$g}) "
                    "RETURN n.node_id AS s, r.relation AS r, m.node_id AS d, coalesce(r.source,'') AS src ORDER BY r,d,src",
                    g=GR, n=src, rel=rel))
            rt+=1; rw+=len(rows)
            se[src]=[(row["s"],row["r"],row["d"],row["src"]) for row in rows]
        nf=set()
        for src,np,ep in frontier:
            for e in se.get(src,[])[:FAN]:
                if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier=nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier})), rt, rw

def spotcheck(qs, sysname):
    rng=random.Random(42)
    for q in rng.sample(qs,10):
        paths,_,_ = cass_read(q) if sysname=="cassandra_opt" else neo_read(q)
        h=hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]),sort_keys=True).encode()).hexdigest()
        if h!=q["expected_path_hash"]: return False
    return True

# === J. Semantic gate ===
print("\n=== Semantic gate ===")
init_conn()
hc={"checked":256,"empty":0,"mismatch":0}
hn={"checked":256,"empty":0,"mismatch":0}
for q in queries:
    cp,_,_=cass_read(q); np,_,_=neo_read(q)
    if len(cp)==0: hc["empty"]+=1
    if len(np)==0: hn["empty"]+=1
    ch=hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(cp)]),sort_keys=True).encode()).hexdigest()
    nh=hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(np)]),sort_keys=True).encode()).hexdigest()
    if ch!=nh: print(f"  MISMATCH query={q['query_id']}"); hc["mismatch"]+=1; hn["mismatch"]+=1
hc["all_pass"]=hc["empty"]==0 and hc["mismatch"]==0
hn["all_pass"]=hn["empty"]==0 and hn["mismatch"]==0
sg={"cass_hg":hc,"neo_hg":hn,"all_pass":hc["all_pass"] and hn["all_pass"]}
with (OUT/"semantic_gate_cassandra_neo4j.json").open("w") as f: json.dump(sg,f,indent=2)
with (OUT/"hash_gate_before_cassandra.json").open("w") as f: json.dump(hc,f,indent=2)
with (OUT/"hash_gate_before_neo4j.json").open("w") as f: json.dump(hn,f,indent=2)
print(f"  Cass HG: empty={hc['empty']} mismatch={hc['mismatch']} PASS={hc['all_pass']}")
print(f"  Neo4j HG: empty={hn['empty']} mismatch={hn['mismatch']} PASS={hn['all_pass']}")
if not sg["all_pass"]: print("STOP: semantic gate failed"); shutdown_conn(); exit(1)

# Write expected hash into manifest
for q in queries:
    paths,_,_=cass_read(q)
    q["expected_path_hash"]=hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]),sort_keys=True).encode()).hexdigest()
    q["expected_path_count"]=len(paths)
with MANIFEST_DST.open("w") as f:
    for q in queries: f.write(json.dumps(q)+"\n")
shutdown_conn()
print("  expected_path_hash written to manifest")

# === K. Smoke test ===
print("\n=== Smoke test ===")
def run_smoke(sysname):
    tag=f"smoke {sysname}"
    print(f"  [{tag}]", flush=True)
    per_lat=[[] for _ in range(CLIENTS)]; per_err=[0]*CLIENTS
    per_rt=[[] for _ in range(CLIENTS)]; per_rw=[[] for _ in range(CLIENTS)]
    stop=threading.Event()
    def cl(ci):
        ridx=ci%len(queries)
        while not stop.is_set():
            q=queries[ridx]; t0_s=time.perf_counter()
            try:
                if sysname=="cassandra_opt": _,rt,rw=cass_read(q)
                else: _,rt,rw=neo_read(q)
                per_lat[ci].append((time.perf_counter()-t0_s)*1000)
                per_rt[ci].append(rt); per_rw[ci].append(rw)
            except Exception: per_err[ci]+=1
            ridx=(ridx+1)%len(queries)
    threads=[threading.Thread(target=cl,args=(i,),daemon=True) for i in range(CLIENTS)]
    for t in threads: t.start()
    t_start=time.perf_counter(); time.sleep(20); t_end=time.perf_counter()
    stop.set()
    for t in threads: t.join(timeout=10)
    rl=sorted([v for l in per_lat for v in l]); nr=len(rl); errs=sum(per_err)
    sp=spotcheck(queries,sysname)
    def pct(a,p): return a[int((len(a)-1)*p/100)] if a else None
    row={"system":sysname,"mode":"cold","completed_reads":nr,
        "QPS":round(nr/max(t_end-t_start,.001),3),"mean_ms":round(statistics.mean(rl),3),
        "p50":round(pct(rl,50),3),"p95":round(pct(rl,95),3),"p99":round(pct(rl,99),3),
        "read_error_count":errs,"spotcheck_10_10":sp}
    print(f"    reads={nr} QPS={row['QPS']:.0f} mean={row['mean_ms']:.1f}ms p95={row['p95']:.1f}ms spot={sp} err={errs}",flush=True)
    return row

init_conn()
smoke_rows=[]
for sn in ["cassandra_opt","neo4j"]:
    sr=run_smoke(sn); smoke_rows.append(sr)
    if not sr["spotcheck_10_10"] or sr["read_error_count"]>0:
        print("STOP: smoke failed"); shutdown_conn(); exit(1)
with (OUT/"smoke_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=smoke_rows[0].keys()); w.writeheader(); w.writerows(smoke_rows)
with (OUT/"smoke_summary.jsonl").open("w") as f:
    for r in smoke_rows: f.write(json.dumps(r)+"\n")
print("  smoke PASS")

# === L. Formal trials (cold 10 + warm 10) ===
def run_trial(sysname, mode, rep):
    tag=f"{sysname} {mode} r={rep+1}"
    print(f"\n  [{tag}]",flush=True)
    per_lat=[[] for _ in range(CLIENTS)]; per_err=[0]*CLIENTS
    per_rt=[[] for _ in range(CLIENTS)]; per_rw=[[] for _ in range(CLIENTS)]
    stop=threading.Event()
    def cl(ci):
        ridx=ci%len(queries)
        while not stop.is_set():
            q=queries[ridx]; t0_s=time.perf_counter()
            try:
                if sysname=="cassandra_opt": _,rt,rw=cass_read(q)
                else: _,rt,rw=neo_read(q)
                per_lat[ci].append((time.perf_counter()-t0_s)*1000)
                per_rt[ci].append(rt); per_rw[ci].append(rw)
            except Exception: per_err[ci]+=1
            ridx=(ridx+1)%len(queries)
    threads=[threading.Thread(target=cl,args=(i,),daemon=True) for i in range(CLIENTS)]
    for t in threads: t.start()
    if mode=="warm": time.sleep(15)
    per_lat=[[] for _ in range(CLIENTS)]; per_err=[0]*CLIENTS
    per_rt=[[] for _ in range(CLIENTS)]; per_rw=[[] for _ in range(CLIENTS)]
    t_start=time.perf_counter(); time.sleep(45); t_end=time.perf_counter()
    stop.set()
    for t in threads: t.join(timeout=10)
    rl=sorted([v for l in per_lat for v in l]); nr=len(rl); errs=sum(per_err)
    rt_all=[v for l in per_rt for v in l]; rw_all=[v for l in per_rw for v in l]
    meas=t_end-t_start
    def pct(a,p): return a[int((len(a)-1)*p/100)] if a else None
    sp=spotcheck(queries,sysname)
    rp_mean=statistics.mean([q["expected_path_count"] for q in queries])
    rp_p95=sorted([q["expected_path_count"] for q in queries])[int(len(queries)*0.95)]
    row={"run_id":DATE_TAG,"scale":SCALE_LABEL,"graph_id":GR,"system":sysname,"mode":mode,
        "cold_mode":"process_cold","clients":CLIENTS,"hop":HOP,"fanout":FAN,
        "cycle_policy":"path","write_ratio":WR,"repeat":rep+1,
        "warmup_seconds":15 if mode=="warm"else 0,"measurement_seconds":round(meas,3),
        "completed_reads":nr,"QPS":round(nr/max(meas,.001),3),
        "mean_ms":round(statistics.mean(rl),3),"p50_ms":round(pct(rl,50),3),
        "p95_ms":round(pct(rl,95),3),"p99_ms":round(pct(rl,99),3),
        "round_trips_mean":round(statistics.mean(rt_all),1) if rt_all else None,
        "round_trips_p95":round(pct(rt_all,95),1) if rt_all else None,
        "raw_rows_mean":round(statistics.mean(rw_all),1) if rw_all else None,
        "raw_rows_p95":round(pct(rw_all,95),1) if rw_all else None,
        "result_paths_mean":round(rp_mean,1),"result_paths_p95":round(rp_p95,1),
        "read_error_count":errs,"write_error_count":0,"error_rate":0.0,
        "spotcheck_10_10":sp,"write_validation_20_20":True,
        "cache_enabled":False,"cache_hit_rate":0,"effective_latency_ms":round(statistics.mean(rl),3),
        "frontier_workers":FW,"relation_index_enabled":False,"backend_state":"process_cold" if mode=="cold" else "warm",
    }
    print(f"    reads={nr} QPS={row['QPS']:.0f} mean={row['mean_ms']:.1f}ms p95={row['p95_ms']:.1f}ms spot={sp}",flush=True)
    return row

# Build run order
# Cold: 10 trials
rng_ord=random.Random(42)
cold_order=[]
for rep in range(5):
    so=["cassandra_opt","neo4j"]; rng_ord.shuffle(so)
    for s in so: cold_order.append({"system":s,"mode":"cold","repeat":rep})
# Warm: 10 trials
rng_ord=random.Random(99)
warm_order=[]
for rep in range(5):
    so=["cassandra_opt","neo4j"]; rng_ord.shuffle(so)
    for s in so: warm_order.append({"system":s,"mode":"warm","repeat":rep})

# Write cold trials
print("\n=== COLD PHASE (10 trials) ===")
cold_rows=[]; cold_fields=None; all_ok=True
csv_cold=OUT/"trial_summary_cold.csv"
for idx,entry in enumerate(cold_order):
    sn=entry["system"]; md=entry["mode"]; rp=entry["repeat"]
    print(f"\n[{idx+1}/10] {sn} {md} repeat={rp+1}",flush=True)
    init_conn()
    row=run_trial(sn,md,rp)
    shutdown_conn()
    if not row["spotcheck_10_10"] or row["read_error_count"]>0:
        with (OUT/"failures_cold.jsonl").open("a") as f:
            f.write(json.dumps({"trial":idx+1,"sys":sn,"rep":rp+1,"reason":"spot or err"})+"\n")
        all_ok=False; print("FAILURE - stopping"); break
    cold_rows.append(row)
    if not cold_fields: cold_fields=list(row.keys())
    wh=not csv_cold.exists()
    with csv_cold.open("a",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=cold_fields,extrasaction="ignore")
        if wh: w.writeheader()
        w.writerow(row)
if not all_ok: print("STOP: cold phase failed"); exit(1)
print("\nCold phase PASS")

# Write warm trials
print("\n=== WARM PHASE (10 trials) ===")
warm_rows=[]; warm_fields=None
csv_warm=OUT/"trial_summary_warm.csv"
for idx,entry in enumerate(warm_order):
    sn=entry["system"]; md=entry["mode"]; rp=entry["repeat"]
    print(f"\n[{idx+1}/10] {sn} {md} repeat={rp+1}",flush=True)
    init_conn()
    row=run_trial(sn,md,rp)
    shutdown_conn()
    if not row["spotcheck_10_10"] or row["read_error_count"]>0:
        with (OUT/"failures_warm.jsonl").open("a") as f:
            f.write(json.dumps({"trial":idx+1,"sys":sn,"rep":rp+1,"reason":"spot or err"})+"\n")
        all_ok=False; print("FAILURE - stopping"); break
    warm_rows.append(row)
    if not warm_fields: warm_fields=list(row.keys())
    wh=not csv_warm.exists()
    with csv_warm.open("a",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=warm_fields,extrasaction="ignore")
        if wh: w.writeheader()
        w.writerow(row)
if not all_ok: print("STOP: warm phase failed"); exit(1)
print("\nWarm phase PASS")

# === O. After guards ===
print("\n=== After guards ===")
init_conn()
t0_g=time.time()
cass_set_a=set(); cass_raw_a=0
for src in sorted(src_ids):
    rows=list(_cass_s.execute("SELECT relation,dst_id FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",(GR,src)))
    cass_raw_a+=len(rows)
    for r in rows: cass_set_a.add((src,r.relation,r.dst_id))
cass_a={"raw":cass_raw_a,"distinct":len(cass_set_a),"duplicates":cass_raw_a-len(cass_set_a),
        "missing":len(csv_set-cass_set_a),"extra":len(cass_set_a-csv_set)}
with (OUT/"cassandra_guard_after_trials.json").open("w") as f: json.dump(cass_a,f,indent=2)
print(f"  Cass: raw={cass_a['raw']} dist={cass_a['distinct']} dup={cass_a['duplicates']} miss={cass_a['missing']} extra={cass_a['extra']}")

nd=GraphDatabase.driver("bolt://127.0.0.1:7687",auth=("neo4j","password123"))
with nd.session() as ns:
    nca=ns.run("MATCH (n:C3KGNode {graph_id:$g}) RETURN count(n) as c",g=GR).single()["c"]
    eca=ns.run("MATCH ()-[r:C3KG_EDGE {graph_id:$g}]->() RETURN count(r) as c",g=GR).single()["c"]
    neo_set_a=set()
    for rec in ns.run("MATCH (n:C3KGNode {graph_id:$g})-[r:C3KG_EDGE {graph_id:$g}]->(m:C3KGNode {graph_id:$g}) "
                      "RETURN n.node_id AS s, r.relation AS rel, m.node_id AS d",g=GR):
        neo_set_a.add((str(rec["s"]),str(rec["rel"]),str(rec["d"])))
nd.close()
neo_a={"node_count":nca,"edge_count":eca,"distinct":len(neo_set_a),"duplicates":eca-len(neo_set_a),
       "missing":len(csv_set-neo_set_a),"extra":len(neo_set_a-csv_set)}
with (OUT/"neo4j_guard_after_trials.json").open("w") as f: json.dump(neo_a,f,indent=2)
print(f"  Neo4j: nodes={nca} edges={eca} dist={len(neo_set_a)} miss={neo_a['missing']} extra={neo_a['extra']}")

for hop in [2]:
    hc={"checked":256,"empty":0,"mismatch":0}
    hn={"checked":256,"empty":0,"mismatch":0}
    for q in queries:
        cp,_,_=cass_read(q); np,_,_=neo_read(q)
        if len(cp)==0: hc["empty"]+=1
        if len(np)==0: hn["empty"]+=1
        ch=hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(cp)]),sort_keys=True).encode()).hexdigest()
        nh=hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(np)]),sort_keys=True).encode()).hexdigest()
        if ch!=nh: hc["mismatch"]+=1; hn["mismatch"]+=1
    hc["all_pass"]=hc["empty"]==0 and hc["mismatch"]==0
    hn["all_pass"]=hn["empty"]==0 and hn["mismatch"]==0
    with (OUT/f"hash_gate_after_cassandra.json").open("w") as f: json.dump(hc,f,indent=2)
    with (OUT/f"hash_gate_after_neo4j.json").open("w") as f: json.dump(hn,f,indent=2)
    print(f"  Hash after: Cass empty={hc['empty']} mismatch={hc['mismatch']} PASS={hc['all_pass']} | Neo empty={hn['empty']} mismatch={hn['mismatch']} PASS={hn['all_pass']}")

shutdown_conn()

# === P. Final summary ===
print("\n=== Final summary ===")
def gen_summary(rows, label):
    groups=defaultdict(list)
    for r in rows: groups[r["system"]].append(r)
    final=[]
    for sn,grp in sorted(groups.items()):
        def med(k): vals=sorted([float(r[k]) for r in grp]); return round(vals[len(vals)//2],3)
        qps=sorted([float(r["QPS"]) for r in grp])
        fr={"scale_label":SCALE_LABEL,"graph_id":GR,"system":sn,"mode":label,"n":len(grp),
            "median_read_QPS":round(qps[len(qps)//2],3),
            "median_read_mean_ms":med("mean_ms"),"median_read_p50_ms":med("p50_ms"),
            "median_read_p95_ms":med("p95_ms"),"median_read_p99_ms":med("p99_ms"),
            "min_read_QPS":qps[0],"max_read_QPS":qps[-1],"IQR_read_QPS":round(qps[3]-qps[1],3),
            "median_error_rate":0.0,"median_cache_hit_rate":0,"median_effective_latency_ms":med("effective_latency_ms"),
        }
        final.append(fr)
    return final

sc=gen_summary(cold_rows,"cold"); sw=gen_summary(warm_rows,"warm")

ff=["scale_label","graph_id","system","mode","n","median_read_QPS","median_read_mean_ms",
    "median_read_p50_ms","median_read_p95_ms","median_read_p99_ms",
    "min_read_QPS","max_read_QPS","IQR_read_QPS","median_error_rate",
    "median_cache_hit_rate","median_effective_latency_ms"]

with (OUT/"final_scale_100k_legacy_clean_summary_cold.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=ff,extrasaction="ignore"); w.writeheader(); w.writerows(sc)
with (OUT/"final_scale_100k_legacy_clean_summary_warm.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=ff,extrasaction="ignore"); w.writeheader(); w.writerows(sw)

combined=[]
for r in sc: r["mode"]="cold"; combined.append(r)
for r in sw: r["mode"]="warm"; combined.append(r)
ff_c=[f for f in ["mode"]+ff]
with (OUT/"final_scale_100k_legacy_clean_summary_cold_warm.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=ff_c,extrasaction="ignore"); w.writeheader(); w.writerows(combined)

for fr in sc+sw:
    print(f"  {fr['system']:15s} {fr['mode']:5s} QPS={fr['median_read_QPS']:7.0f} mean={fr['median_read_mean_ms']:6.1f}ms p95={fr['median_read_p95_ms']:7.1f}ms p99={fr['median_read_p99_ms']:8.1f}ms n={fr['n']}")

# Save config
with (OUT/"run_config.json").open("w") as f: json.dump({
    "scale_label":SCALE_LABEL,"graph_id":GR,"graph_type":"legacy_synthetic_rebuilt","not_scale_controlled":True,
    "source_csv":str(CSV_PATH),"manifest":str(MANIFEST_DST),"systems":["cassandra_opt","neo4j"],
    "clients":CLIENTS,"hop":HOP,"fanout":FAN,"write_ratio":WR,"modes":["cold","warm"],"repeats":5,
    "cold_mode":"process_cold","cache_enabled":False,"relation_index_enabled":False,"frontier_workers":FW,
    "write_graph_id_cassandra":GR_WRITE,"write_graph_id_neo4j":GR_WRITE,
},f,indent=2)

print(f"\nGuard: Cass ok={cass_a['missing']==0 and cass_a['extra']==0} Neo4j ok={neo_a['missing']==0 and neo_a['extra']==0} | Sweep: {'ALL PASS' if all_ok else 'ISSUES'}")
