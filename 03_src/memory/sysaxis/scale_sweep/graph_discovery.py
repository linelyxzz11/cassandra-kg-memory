"""Cassandra discovery for synth_100000_1781447372"""
import json, time
from cassandra.cluster import Cluster
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GR = "synth_100000_1781447372"
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/sysaxis_scale_sweep_final/scale_100k")
BATCH = 500

def count_batch(start, end):
    c = Cluster(["127.0.0.1"], port=9042)
    s = c.connect("ai_memory")
    total = 0
    for src_id in range(start, end):
        try:
            rows = s.execute(
                "SELECT COUNT(*) as c FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
                (GR, str(src_id))
            )
            total += rows[0].c
        except Exception:
            pass
    s.shutdown()
    c.shutdown()
    return total

def count_distinct_batch(start, end):
    c = Cluster(["127.0.0.1"], port=9042)
    s = c.connect("ai_memory")
    distinct = set()
    for src_id in range(start, end):
        try:
            rows = s.execute(
                "SELECT src_id,relation,dst_id FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
                (GR, str(src_id))
            )
            for r in rows:
                distinct.add((r.src_id, r.relation, r.dst_id))
        except Exception:
            pass
    s.shutdown()
    c.shutdown()
    return distinct

print(f"Cassandra discovery: {GR}")
t0 = time.time()

# Count raw edges in parallel batches
max_src = 6100
batches = [(i, min(i+BATCH, max_src)) for i in range(1, max_src, BATCH)]
results = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(count_batch, s, e): (s, e) for s, e in batches}
    for f in as_completed(futures):
        results.append(f.result())
    ex.shutdown(wait=True)

raw = sum(results)
print(f"  raw_count = {raw}  ({time.time()-t0:.1f}s)")

# Count distinct edges (first 2000 src_ids sampled)
distinct_set = set()
sample_end = min(2000, max_src)
for src in range(1, sample_end):
    c2 = Cluster(["127.0.0.1"], port=9042)
    s2 = c2.connect("ai_memory")
    try:
        rs = s2.execute(
            "SELECT src_id,relation,dst_id FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
            (GR, str(src))
        )
        for r in rs:
            distinct_set.add((r.src_id, r.relation, r.dst_id))
    except Exception:
        pass
    s2.shutdown()
    c2.shutdown()

distinct_sample = len(distinct_set)
dup_estimate = raw - (raw * distinct_sample / (raw * sample_end / max_src)) if raw else 0
print(f"  distinct sample (first {sample_end} src_ids) = {distinct_sample}")
print(f"  total time = {time.time()-t0:.1f}s")

# Load existing discovery
with open(OUT/"graph_discovery.json") as f:
    disc = json.load(f)

disc["cassandra_candidates"] = [{
    "graph_id": GR,
    "raw_count": raw,
    "distinct_sample": distinct_sample,
    "sample_src_range": f"1-{sample_end}",
    "estimated_total_distinct": "unknown",
    "duplicates_known": False,
    "in_100k_range": 90000 <= raw <= 110000,
    "neo4j_nodes": 6013,
    "neo4j_edges": 101120,
    "note": "Legacy synthetic graph from B1 experiment. Not scale-controlled.",
    "graph_type": "legacy_synthetic",
    "source_csv": "results/c1_source_100k.csv",
    "manifest_candidate": "results/c1_manifest_100k_h2.jsonl"
}]

disc["recommendation"] = f"synth_100000_1781447372: Cass raw={raw}, Neo4j edges=101120. "
if 90000 <= raw <= 110000:
    disc["recommendation"] += "Cassandra raw count in 100K range. RECOMMENDED for legacy 100K sweep."
else:
    disc["recommendation"] += f"Cassandra raw={raw} unexpected. Consider new clean 100K import."

disc["scale_label"] = "100K_legacy"
disc["scale_controlled"] = False

with open(OUT/"graph_discovery.json", "w") as f:
    json.dump(disc, f, indent=2, default=str)

with open(OUT/"graph_discovery.txt", "w") as f:
    f.write("Graph Discovery — 100K Scale Sweep\n")
    f.write(f"graph_id = {GR}\n")
    f.write(f"Cassandra raw_count = {raw}\n")
    f.write(f"Cassandra distinct_sample (src_id 1-{sample_end}) = {distinct_sample}\n")
    f.write(f"Neo4j edge_count = 101120\n")
    f.write(f"Neo4j node_count = 6013\n")
    f.write(f"source_csv = results/c1_source_100k.csv (archived)\n")
    f.write(f"manifest = results/c1_manifest_100k_h2.jsonl (archived)\n")
    f.write(f"graph_type = legacy_synthetic\n")
    f.write(f"scale_label = 100K_legacy\n")
    f.write(f"in_100k_range = {90000 <= raw <= 110000}\n")
    f.write(f"\nRecommendation: {disc['recommendation']}\n")

print(f"\nDone. graph_discovery.json + .txt written to {OUT}")
print(f"raw={raw}  distinct_sample={distinct_sample}  neo4j_edges=101120")
