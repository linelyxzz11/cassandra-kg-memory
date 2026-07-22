"""100K legacy graph integrity check: CSV stats + Neo4j stats + Cassandra parallel guard"""
import csv, json, time, hashlib, uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from cassandra.cluster import Cluster
from neo4j import GraphDatabase

GR = "synth_100000_1781447372"
PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT = PROJ / "reports/sysaxis_scale_sweep_final/scale_100k_legacy"
OUT.mkdir(parents=True, exist_ok=True)
CSV = PROJ / "results/c1_source_100k.csv"

t0 = time.time()

# === B. CSV stats ===
print("=== CSV ===")
csv_set = set()
csv_raw = 0
src_ids = set()
dst_ids = set()
with CSV.open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        csv_raw += 1
        edge = (r["graph_id"], r["src_id"], r["relation"], r["dst_id"])
        csv_set.add(edge)
        src_ids.add(r["src_id"])
        dst_ids.add(r["dst_id"])
csv_dup = csv_raw - len(csv_set)
csv_stats = {
    "csv_rows": csv_raw, "csv_distinct_logical_edges": len(csv_set),
    "distinct_src_ids": len(src_ids), "distinct_dst_ids": len(dst_ids),
    "duplicate_logical_edges": csv_dup
}
with (OUT/"csv_stats.json").open("w") as f: json.dump(csv_stats, f, indent=2)
print(f"  rows={csv_raw} distinct={len(csv_set)} dup={csv_dup} src={len(src_ids)} dst={len(dst_ids)} ({time.time()-t0:.1f}s)")

# === C. Neo4j stats ===
print("\n=== Neo4j ===")
nd = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "password123"))
with nd.session() as ns:
    nc = ns.run("MATCH (n:C3KGNode {graph_id:$g}) RETURN count(n) as c", g=GR).single()["c"]
    ec = ns.run("MATCH ()-[r:C3KG_EDGE {graph_id:$g}]->() RETURN count(r) as c", g=GR).single()["c"]
    neo_set = set()
    for rec in ns.run("MATCH (n:C3KGNode {graph_id:$g})-[r:C3KG_EDGE {graph_id:$g}]->(m:C3KGNode {graph_id:$g}) RETURN n.node_id AS src, r.relation AS rel, m.node_id AS dst", g=GR):
        neo_set.add((GR, str(rec["src"]), str(rec["rel"]), str(rec["dst"])))
    neo_dup = ec - len(neo_set)
neo_stats = {"node_count": nc, "edge_count": ec, "distinct_logical_edges": len(neo_set), "duplicate_logical_edges": neo_dup}
nd.close()
with (OUT/"neo4j_stats.json").open("w") as f: json.dump(neo_stats, f, indent=2)
print(f"  nodes={nc} edges={ec} distinct_logical={len(neo_set)} dup={neo_dup} ({time.time()-t0:.1f}s)")

# === D. Cassandra parallel guard ===
print("\n=== Cassandra (parallel 128) ===")
c = Cluster(["127.0.0.1"], port=9042)
s = c.connect("ai_memory")
ps = s.prepare("SELECT relation,dst_id FROM kg_edges_by_src WHERE graph_id=? AND src_id=?")
FWS = 128

def cass_src(src):
    try:
        cs = Cluster(["127.0.0.1"], port=9042)
        ss = cs.connect("ai_memory")
        p = ss.prepare("SELECT relation,dst_id FROM kg_edges_by_src WHERE graph_id=? AND src_id=?")
        rows = list(ss.execute(p, (GR, src)))
        result = [(GR, src, r.relation, r.dst_id) for r in rows]
        ss.shutdown(); cs.shutdown()
        return {"src": src, "count": len(result), "edges": result}
    except Exception as e:
        return {"src": src, "count": 0, "edges": [], "error": str(e)}

all_srcs = sorted(src_ids)
cass_set = set(); cass_raw = 0; empty = 0; errors = 0
with ThreadPoolExecutor(max_workers=FWS) as ex:
    futures = [ex.submit(cass_src, s) for s in all_srcs]
    for f in as_completed(futures):
        r = f.result()
        if "error" in r: errors += 1; continue
        if r["count"] == 0: empty += 1
        cass_raw += r["count"]
        for e in r["edges"]: cass_set.add(e)
s.shutdown(); c.shutdown()

cass_dup = cass_raw - len(cass_set)
missing = csv_set - cass_set
extra = cass_set - csv_set
cass_stats = {
    "actual_raw_rows": cass_raw, "actual_distinct_logical_edges": len(cass_set),
    "duplicates": cass_dup, "missing_vs_csv": len(missing), "extra_vs_csv": len(extra),
    "partition_count_checked": len(all_srcs), "empty_partitions": empty,
    "errors": errors, "elapsed_seconds": round(time.time()-t0, 1),
    "rows_per_second": round(cass_raw/max(time.time()-t0, 0.001))
}
with (OUT/"cassandra_parallel_guard.json").open("w") as f: json.dump(cass_stats, f, indent=2)
print(f"  raw={cass_raw} distinct={len(cass_set)} dup={cass_dup} missing={len(missing)} extra={len(extra)} empty={empty} errs={errors} ({cass_stats['elapsed_seconds']}s)")

# === E. Verdict ===
print("\n=== VERDICT ===")
csv_dist = csv_stats["csv_distinct_logical_edges"]
neo_dist = neo_stats["distinct_logical_edges"]
all_match = (cass_dup == 0 and neo_dup == 0 and len(missing) == 0 and
             len(extra) == 0 and cass_raw == csv_raw)
print(f"  CSV:   rows={csv_raw} distinct={csv_dist} dup={csv_dup}")
print(f"  Neo4j: edges={ec} distinct={neo_dist} dup={neo_dup}")
print(f"  Cass:  raw={cass_raw} distinct={len(cass_set)} dup={cass_dup} miss={len(missing)} extra={len(extra)}")
print(f"  ALL_MATCH: {all_match}")
if all_match: print("  => CASE 1: USE synth_100000_1781447372 as 100K_legacy")
else: print("  => CASE 2: NEED REBUILD clean 100K graph")
