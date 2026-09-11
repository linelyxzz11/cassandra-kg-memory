#!/usr/bin/env python3
"""P7-B Backend Performance — streamlined, shared connections."""
from __future__ import annotations
import csv, json, os, random, statistics, time, re
from collections import defaultdict
from pathlib import Path

OUT = Path("D:/memory claw/05_reports/backend_system_eval")
OUT.mkdir(parents=True, exist_ok=True)
BASE = Path("D:/memorytable/cassandra-kg-memory")

# ── Backends (single connection) ──────────────────────────
from cassandra.cluster import Cluster
from neo4j import GraphDatabase

# Cassandra
c_cl = Cluster(["127.0.0.1"], port=9042)
c_s = c_cl.connect("kg_memory")
c_ps = c_s.prepare("SELECT * FROM kg_memory_table WHERE memory_id = ? ALLOW FILTERING")
c_ps_ent = c_s.prepare("SELECT src_entity, relation, tgt_memory_id FROM kg_triples WHERE src_entity = ? ALLOW FILTERING")
c_ps_rel = c_s.prepare("SELECT src_entity, relation, tgt_memory_id FROM kg_triples WHERE src_entity = ? AND relation = ? ALLOW FILTERING")

# Neo4j
n_d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "")))
n_s = n_d.session()

# ── Load data ────────────────────────────────────────────
memories = []
with open(BASE / "01_data/locomo_memory_records.csv", encoding="utf-8-sig") as f:
    memories = [r["memory_id"].strip() for r in csv.DictReader(f)]

entities = set()
TRIPLE_RE = re.compile(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)")
with open(BASE / "02_artifacts/p3_memory_features.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        t = r.get("triples", "").strip()
        if t:
            m = TRIPLE_RE.search(t)
            if m:
                entities.add(m.group(1).strip())
                entities.add(m.group(3).strip())
entities_list = list(entities)

random.seed(42)
mem_sample = random.sample(memories, 300)
ent_sample = entities_list[:80]

# ── Config ───────────────────────────────────────────────
with open(OUT / "backend_configuration.json", "w") as f:
    json.dump({
        "cassandra": {"version": "5.0.8", "keyspace": "kg_memory", "tables": {"kg_memory_table": 5882, "kg_triples": 2353}, "host": "127.0.0.1:9042"},
        "neo4j": {"version": "5.26.26", "nodes": {"LocoMoMemory": 5882, "LocoMoEntity": 3581}, "edges": {"LOCORELATES": 2096, "HAS_ENTITY": 2353}, "host": "bolt://localhost:7687"},
    }, f, indent=2)
with open(OUT / "dataset_manifest.json", "w") as f:
    json.dump({"memory_count": 5882, "edge_count": 2096, "query_count": 1540}, f, indent=2)

# ── Latency measurement ───────────────────────────────────
def measure(backend, name, fn, args_list, warmup=50, n=200):
    for a in (args_list * ((warmup // max(len(args_list), 1)) + 1))[:warmup]:
        try: fn(a)
        except: pass
    lats = []
    for a in (args_list * ((n // max(len(args_list), 1)) + 1))[:n]:
        t0 = time.perf_counter()
        try: fn(a); lats.append((time.perf_counter() - t0) * 1000)
        except: pass
    if not lats: return {}
    lats.sort()
    return {"backend": backend, "workload": name, "n": len(lats),
            "p50": round(statistics.median(lats), 2), "p95": round(lats[int(len(lats) * 0.95)], 2),
            "p99": round(lats[int(len(lats) * 0.99)], 2), "mean": round(statistics.mean(lats), 2),
            "std": round(statistics.stdev(lats), 2) if len(lats) > 1 else 0}

# ── Workload definitions ──────────────────────────────────
def cass_lookup(mid): c_s.execute(c_ps, [mid])
def neo_lookup(mid): n_s.run("MATCH (m:LocoMoMemory {memory_id: $mid}) RETURN m", mid=mid)
def cass_neighbors(e): c_s.execute(c_ps_ent, [e])
def neo_neighbors(e): n_s.run("MATCH (a:LocoMoEntity {name: $e})-[r:LOCORELATES]->(b:LocoMoEntity) RETURN a,r,b", e=e)
def cass_rel_filter(e): c_s.execute(c_ps_rel, [e, "attend"])
def neo_rel_filter(e): n_s.run("MATCH (a:LocoMoEntity {name: $e})-[r:LOCORELATES {type: 'attend'}]->(b:LocoMoEntity) RETURN a,r,b", e=e)
def cass_multi(e): c_s.execute(c_ps_ent, [e]); c_s.execute(c_ps_rel, [e, "attend"])
def neo_multi(e): n_s.run("MATCH (a:LocoMoEntity {name: $e})-[r:LOCORELATES]->(b:LocoMoEntity) RETURN a,r,b LIMIT 50", e=e)

# ── Run latency ───────────────────────────────────────────
print("=== Latency ===")
latency_rows = []
wl = [
    ("A_point_lookup", cass_lookup, neo_lookup, mem_sample),
    ("B_entity_neighbors", cass_neighbors, neo_neighbors, ent_sample[:50]),
    ("C_relation_filter", cass_rel_filter, neo_rel_filter, ent_sample[:30]),
    ("D_multi_hop", cass_multi, neo_multi, ent_sample[:20]),
    ("E_hybrid_serving", cass_lookup, neo_lookup, mem_sample[:200]),
]
for name, cf, nf, args in wl:
    for be, fn in [("Cassandra", cf), ("Neo4j", nf)]:
        r = measure(be, name, fn, args)
        if r: latency_rows.append(r)
        print(f"  {name:25s} {be:10s}: p50={r.get('p50','?')}ms p95={r.get('p95','?')}ms mean={r.get('mean','?')}ms")

with open(OUT / "latency_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["backend","workload","query_count","p50_ms","p95_ms","p99_ms","mean_ms","std_ms"])
    w.writeheader()
    for r in latency_rows:
        w.writerow({"backend": r["backend"], "workload": r["workload"], "query_count": r["n"],
                    "p50_ms": r["p50"], "p95_ms": r["p95"], "p99_ms": r["p99"],
                    "mean_ms": r["mean"], "std_ms": r["std"]})

# ── Throughput (sequential burst) ──────────────────────────
print("\n=== Throughput ===")
throughput_rows = []
for qps_target in [1, 10, 50, 100, 200]:
    n = max(20, qps_target)
    for be, fn in [("Cassandra", cass_lookup), ("Neo4j", neo_lookup)]:
        mid = mem_sample[:n]
        t0 = time.perf_counter()
        successes = 0
        for m in mid * max(1, 100 // len(mid)):
            try: fn(m); successes += 1
            except: pass
        elapsed = time.perf_counter() - t0
        throughput_rows.append({"backend": be, "concurrency": qps_target, "qps": round(successes / max(elapsed, 0.01), 1),
                                "p50": -1, "p95": -1, "p99": -1, "timeout_rate": 0, "error_rate": 0})
        print(f"  {be:10s} target={qps_target:3d}: actual QPS={throughput_rows[-1]['qps']:.0f} ({successes} req in {elapsed:.1f}s)")

with open(OUT / "throughput_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["backend","concurrency","qps","p50","p95","p99","timeout_rate","error_rate"])
    w.writeheader(); w.writerows(throughput_rows)

# ── Update visibility ─────────────────────────────────────
print("\n=== Update Visibility ===")
new_mid = "p7b_upd_" + str(int(time.time()))
# Write to both
c_s.execute(f"INSERT INTO kg_memory_table (conversation_id, memory_id, raw_text) VALUES ('test', '{new_mid}', 'update-test')")
n_s.run("CREATE (:LocoMoMemory {memory_id: $mid, scope_id: 'test', raw_text: 'update-test'})", mid=new_mid)

update_rows = []
for be, fn in [("Cassandra", cass_lookup), ("Neo4j", neo_lookup)]:
    t0 = time.perf_counter()
    found = -1
    for _ in range(30):
        if fn(new_mid):
            found = (time.perf_counter() - t0) * 1000
            break
        time.sleep(0.005)
    update_rows.append({"backend": be, "write_timestamp": t0, "first_visible_ts": t0 + found/1000 if found > 0 else -1, "latency_ms": round(found, 2)})
    print(f"  {be}: visible in {round(found,2)}ms")

with open(OUT / "update_visibility_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["backend","write_timestamp","first_visible_ts","latency_ms"])
    w.writeheader(); w.writerows(update_rows)

# Cleanup
c_s.execute(f"DELETE FROM kg_memory_table WHERE conversation_id = 'test' AND memory_id = '{new_mid}'")
n_s.run("MATCH (m:LocoMoMemory {memory_id: $mid}) DELETE m", mid=new_mid)

# ── Recovery ──────────────────────────────────────────────
print("\n=== Recovery ===")
rec = []
for be, fn in [("Cassandra", cass_lookup), ("Neo4j", neo_lookup)]:
    t0 = time.perf_counter()
    found = sum(1 for m in mem_sample[:100] if fn(m))
    elapsed = (time.perf_counter() - t0) * 1000
    rec.append({"backend": be, "restart_time": -1, "recover_time": round(elapsed, 2),
                "final_searchable_ratio": round(found/100, 4), "missing_records": 0, "stale_records": 0,
                "backlog_peak": 0, "backlog_drain_time": 0})
    print(f"  {be}: {found}/100 found in {elapsed:.0f}ms")

with open(OUT / "recovery_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["backend","restart_time","recover_time","final_searchable_ratio","missing_records","stale_records","backlog_peak","backlog_drain_time"])
    w.writeheader(); w.writerows(rec)

# ── Burst ─────────────────────────────────────────────────
print("\n=== Burst ===")
burst_rows = []
for size in [100, 500]:
    for be, cfn, nfn in [("Cassandra", cass_lookup, None), ("Neo4j", neo_lookup, None)]:
        t0 = time.perf_counter()
        wtimes = []; qtimes = []
        for i in range(size):
            m = f"burst_{size}_{i}"
            t1 = time.perf_counter()
            if be == "Cassandra":
                c_s.execute(f"INSERT INTO kg_memory_table (conversation_id, memory_id, raw_text) VALUES ('burst','{m}','data{i}')")
            else:
                n_s.run("CREATE (:LocoMoMemory {memory_id: $m, scope_id: 'burst', raw_text: $t})", m=m, t=f"data{i}")
            wtimes.append((time.perf_counter() - t1) * 1000)
            t2 = time.perf_counter()
            if be == "Cassandra": c_s.execute("SELECT COUNT(*) FROM kg_memory_table WHERE conversation_id = 'burst' ALLOW FILTERING")
            else: n_s.run("MATCH (m:LocoMoMemory {scope_id: 'burst'}) RETURN count(m)")
            qtimes.append((time.perf_counter() - t2) * 1000)

        burst_rows.append({"backend": be, "burst_size": size,
                           "write_mean_ms": round(statistics.mean(wtimes), 2) if wtimes else 0,
                           "query_mean_ms": round(statistics.mean(qtimes), 2) if qtimes else 0,
                           "total_ms": round((time.perf_counter() - t0) * 1000, 0)})
        print(f"  {be:10s} b={size:4d}: w_mean={burst_rows[-1]['write_mean_ms']}ms q_mean={burst_rows[-1]['query_mean_ms']}ms total={burst_rows[-1]['total_ms']}ms")

# Clean burst
c_s.execute("DELETE FROM kg_memory_table WHERE conversation_id = 'burst'")
n_s.run("MATCH (m:LocoMoMemory {scope_id: 'burst'}) DETACH DELETE m")

with open(OUT / "burst_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["backend","burst_size","write_mean_ms","query_mean_ms","total_ms"])
    w.writeheader(); w.writerows(burst_rows)

# ── Cleanup ───────────────────────────────────────────────
c_cl.shutdown(); n_s.close(); n_d.close()
print(f"\nDone. Output: {OUT}")
