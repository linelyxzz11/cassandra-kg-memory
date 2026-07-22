"""After-guard + Neo4j hash gate + final aggregation"""
import csv, json, hashlib
from collections import defaultdict
from pathlib import Path
from cassandra.cluster import Cluster
from neo4j import GraphDatabase

PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT = PROJ / "reports/sysaxis_1m_concurrency_final"
GR = "c3_scale_1M_seed42"
MANIFEST = PROJ / "results/c3_manifest_scale_1m_h2.jsonl"

# Guard
print("[Guard]", flush=True)
c = Cluster(["127.0.0.1"], port=9042)
s = c.connect("ai_memory")
csv_set = set()
with (PROJ / "results/c3_source_scale_1M.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        csv_set.add((r["src_id"], r["relation"], r["dst_id"], r["source"]))
cass_set = set(); cass_raw = 0
for src in sorted(set(e[0] for e in csv_set)):
    rows = s.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
    for r in rows:
        cass_set.add((str(r.src_id), str(r.relation), str(r.dst_id), str(r.source or ""))); cass_raw += 1
dup = cass_raw - len(cass_set); miss = len(csv_set - cass_set)
g = {"csv": len(csv_set), "raw": cass_raw, "distinct": len(cass_set), "duplicates": dup, "missing": miss}
with (OUT / "read_graph_guard_after_warm.json").open("w") as f:
    json.dump(g, f, indent=2)
print(f"Guard: csv={len(csv_set)} raw={cass_raw} dup={dup} miss={miss} {'PASS' if dup==0 else 'FAIL'}")

# Neo4j hash gate only (Cassandra already done)
d = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "password123"))
queries = [json.loads(line) for line in open(MANIFEST)]
empty = 0; mm = 0
for i, q in enumerate(queries):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        for src in sources:
            with d.session() as sx:
                rows = list(sx.run(
                    "MATCH (n:C3KGNode {graph_id: $g, node_id: $n})-[r:C3KG_EDGE {relation: $rel}]->(m:C3KGNode {graph_id: $g}) "
                    "RETURN n.node_id AS s, r.relation AS r, m.node_id AS d, coalesce(r.source, '') AS src ORDER BY r, d, src",
                    g=GR, n=src, rel=rel))
            se[src] = [(row["s"], row["r"], row["d"], row["src"]) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:20]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
    paths = tuple(sorted({f[2] for f in frontier}))
    if len(paths) == 0:
        empty += 1
    else:
        h = hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()).hexdigest()
        if h != q["expected_path_hash"]: mm += 1
hg = {"checked": 256, "empty": empty, "mismatch": mm, "all_pass": empty == 0 and mm == 0}
with (OUT / "hash_gate_after_warm_neo4j.json").open("w") as f:
    json.dump(hg, f, indent=2)
print(f"Neo4j HG: empty={empty} mismatch={mm} {'PASS' if hg['all_pass'] else 'FAIL'}")
d.close()

# Aggregate warm
rows_w = []
with (OUT / "trial_summary_warm.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): rows_w.append(r)
seen = {}
for r in rows_w: seen[(r["system"], int(r["clients"]), int(r["repeat"]))] = r
groups = defaultdict(list)
for r in seen.values(): groups[(r["system"], int(r["clients"]))].append(r)
final_w = []
for (sn, cl), grp in sorted(groups.items()):
    def med(key):
        vals = sorted([float(r[key]) for r in grp]); return round(vals[len(vals) // 2], 3)
    qps = sorted([float(r["read_QPS"]) for r in grp])
    fr = {"system": sn, "clients": cl, "n": len(grp), "median_QPS": round(qps[len(qps) // 2], 3),
          "median_mean_ms": med("read_mean_ms"), "median_p50_ms": med("read_p50_ms"),
          "median_p95_ms": med("read_p95_ms"), "median_p99_ms": med("read_p99_ms"),
          "min_QPS": qps[0], "max_QPS": qps[-1], "IQR_QPS": round(qps[3] - qps[1], 3), "median_error_rate": 0.0}
    final_w.append(fr)

ff = ["system", "clients", "n", "median_QPS", "median_mean_ms", "median_p50_ms", "median_p95_ms", "median_p99_ms", "min_QPS", "max_QPS", "IQR_QPS", "median_error_rate"]
with (OUT / "final_concurrency_summary_warm.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=ff, extrasaction="ignore"); w.writeheader(); w.writerows(final_w)
with (OUT / "final_concurrency_summary_warm.json").open("w") as f: json.dump(final_w, f, indent=2)

# Combined cold+warm
cw = []
with (OUT / "final_concurrency_summary_cold.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): r["mode"] = "cold"; cw.append(r)
with (OUT / "final_concurrency_summary_warm.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): r["mode"] = "warm"; cw.append(r)
cf = ["system", "clients", "mode", "n", "median_QPS", "median_mean_ms", "median_p50_ms", "median_p95_ms", "median_p99_ms", "min_QPS", "max_QPS", "IQR_QPS"]
with (OUT / "final_concurrency_summary_cold_warm.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cf, extrasaction="ignore"); w.writeheader(); w.writerows(cw)
with (OUT / "final_concurrency_summary_cold_warm.json").open("w") as f: json.dump(cw, f, indent=2)
with (OUT / "failures_warm.jsonl").open("w") as f: pass

c.shutdown()

print("\n=== 1M CONCURRENCY COLD+WARM FINAL ===")
for fr in cw:
    print(f"  {fr['system']:17s} c={int(fr['clients']):2d} {fr['mode']:5s} QPS={float(fr['median_QPS']):7.1f} mean={float(fr['median_mean_ms']):6.1f}ms p95={float(fr['median_p95_ms']):6.1f}ms p99={float(fr['median_p99_ms']):7.1f}ms")
