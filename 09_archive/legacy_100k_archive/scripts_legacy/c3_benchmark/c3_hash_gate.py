"""C3 1M scale hash gate — read-only, zero Cassandra writes."""
import csv, json, hashlib, random, time, sys
from collections import defaultdict
from pathlib import Path
from cassandra.cluster import Cluster

PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT  = PROJ / "reports/data_recovery"
GR   = "c3_scale_1M_seed42"
FAN  = 20; NQ  = 256

MANIFEST_PATH = PROJ / "results/c3_manifest_scale_1m_h2.jsonl"

c = Cluster(["127.0.0.1"], port=9042)
s = c.connect("ai_memory")

print("Loading graph from CSV for seed discovery...", flush=True)
edges = []
with (PROJ / "results/c3_source_scale_1M.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        edges.append(r)
by_src = defaultdict(list)
for e in edges: by_src[e["src_id"]].append(e)
all_srcs = sorted(by_src.keys())
print(f"  {len(edges)} edges, {len(all_srcs)} sources", flush=True)

print("Generating 256 query manifest...", flush=True)
rng = random.Random(20260707)
queries = []
seeds  = set()
for _ in range(NQ * 20):
    if len(queries) >= NQ: break
    seed = rng.choice(all_srcs)
    if seed in seeds: continue
    h1 = sorted(by_src.get(seed, []), key=lambda e: (e["relation"], e["dst_id"], e["source"]))[:FAN]
    if not h1: continue
    rps = set()
    for e1 in h1:
        h2 = sorted(by_src.get(e1["dst_id"], []), key=lambda e: (e["relation"], e["dst_id"], e["source"]))[:FAN]
        for e2 in h2: rps.add((e1["relation"], e2["relation"]))
    if not rps: continue
    seeds.add(seed)
    queries.append({"query_id": f"c3s-{len(queries):06d}", "graph_id": GR, "seed_id": seed,
                    "relation_path": list(rps)[0], "hop": 2, "fanout": FAN, "cycle_policy": "path"})

print(f"Computing expected_path_hash for {len(queries)} queries (READ ONLY)...", flush=True)
t0 = time.perf_counter()
for i, q in enumerate(queries):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier})
        se = {}
        for src in sources:
            rows = s.execute(
                "SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
                (GR, src))
            se[src] = [(str(r.src_id), str(r.relation), str(r.dst_id), str(r.source or ""))
                       for r in rows if r.relation == rel]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
    paths = tuple(sorted({f[2] for f in frontier}))
    q["expected_path_hash"] = hashlib.sha256(
        json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()
    ).hexdigest()
    if (i + 1) % 32 == 0:
        print(f"  {i+1}/{len(queries)} ({time.perf_counter()-t0:.0f}s)", flush=True)

# Save manifest
with MANIFEST_PATH.open("w") as f:
    for q in queries: f.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"  Saved to {MANIFEST_PATH}", flush=True)

# Hash gate: re-read and verify (self-check)
print("Hash gate — re-read verification...", flush=True)
empty = 0
mm    = 0
t1    = time.perf_counter()
for i, q in enumerate(queries):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier})
        se = {}
        for src in sources:
            rows = s.execute(
                "SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
                (GR, src))
            se[src] = [(str(r.src_id), str(r.relation), str(r.dst_id), str(r.source or ""))
                       for r in rows if r.relation == rel]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
    paths = tuple(sorted({f[2] for f in frontier}))
    if len(paths) == 0:
        empty += 1
    else:
        h = hashlib.sha256(
            json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()
        ).hexdigest()
        if h != q["expected_path_hash"]:
            mm += 1
    if (i + 1) % 64 == 0:
        print(f"  {i+1}/{len(queries)} empty={empty} mismatch={mm}", flush=True)

elapsed = time.perf_counter() - t0
hg = {
    "checked": len(queries), "empty": empty, "mismatch": mm,
    "all_pass": empty == 0 and mm == 0, "elapsed_s": round(elapsed, 1),
    "graph_id": GR,
}
with (OUT / "c3_post_restore_hash_gate.json").open("w") as f:
    json.dump(hg, f, indent=2)

c.shutdown()
print(f"\n{'='*50}")
print(f"C3 1M HASH GATE: {len(queries)} checked, empty={empty}, mismatch={mm} -> {'PASS' if hg['all_pass'] else 'FAIL'}")
print(f"Elapsed: {elapsed:.0f}s")
