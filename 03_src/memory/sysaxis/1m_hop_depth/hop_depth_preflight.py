"""Hop-depth preflight: guard + manifests + semantic gates. Read-only."""
import csv, json, hashlib, random, time, shutil
from collections import defaultdict, Counter
from pathlib import Path
from cassandra.cluster import Cluster
from neo4j import GraphDatabase

PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT  = PROJ / "reports/sysaxis_1m_hop_depth_final"
GR   = "c3_scale_1M_seed42"
FAN  = 20; NQ = 256; SC = 64
MANIFEST_DIR = PROJ / "results"

OUT.mkdir(parents=True, exist_ok=True)
d = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "password123"))
c = Cluster(["127.0.0.1"], port=9042)
s = c.connect("ai_memory")

# ── Guard ──
print("[0] Guard", flush=True)
csv_set = set()
with (PROJ/"results/c3_source_scale_1M.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): csv_set.add((r["src_id"], r["relation"], r["dst_id"], r["source"]))
cass_set = set(); cass_raw = 0
for src in sorted(set(e[0] for e in csv_set)):
    rows = s.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
    for r in rows: cass_set.add((str(r.src_id), str(r.relation), str(r.dst_id), str(r.source or ""))); cass_raw += 1
dup = cass_raw - len(cass_set); miss = len(csv_set - cass_set)
guard = {"csv": len(csv_set), "raw": cass_raw, "distinct": len(cass_set), "duplicates": dup, "missing": miss}
with (OUT/"read_graph_guard_before.json").open("w") as f: json.dump(guard, f, indent=2)
print(f"  Guard: csv={len(csv_set)} raw={cass_raw} dup={dup} miss={miss} {'PASS' if dup==0 else 'FAIL'}", flush=True)

# Schema
lines = []
for tbl in ["kg_edges_by_src", "kg_edges_by_dst", "kg_edges_by_relation_bucket", "kg_edges_by_src_relation"]:
    lines.append(f"--- {tbl} ---")
    for row in s.execute(f"SELECT * FROM system_schema.columns WHERE keyspace_name='ai_memory' AND table_name='{tbl}'"):
        lines.append(f"{row.column_name:30s} {row.kind:15s} {row.type}")
    lines.append("")
(OUT/"cassandra_schema_before.txt").write_text("\n".join(lines))

# ── Load graph ──
print("[1] Loading graph edges", flush=True)
edges = []
with (PROJ/"results/c3_source_scale_1M.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): edges.append(r)
by_src = defaultdict(list)
for e in edges: by_src[e["src_id"]].append(e)
all_srcs = sorted(by_src.keys())
print(f"  {len(edges)} edges, {len(all_srcs)} sources", flush=True)

# ── Cassandra traversal ──
def cass_fetch(src):
    rows = s.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
    return [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]

def cass_traverse(seed, rel_path):
    frontier = {(seed, (seed,), ())}
    for rel in rel_path:
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

def neo_traverse(seed, rel_path):
    frontier = {(seed, (seed,), ())}
    for rel in rel_path:
        sources = sorted({f[0] for f in frontier}); se = {}
        for src in sources:
            with d.session() as sx:
                rows = list(sx.run(
                    "MATCH (n:C3KGNode {graph_id: $g, node_id: $n})-[r:C3KG_EDGE {relation: $rel}]->(m:C3KGNode {graph_id: $g}) "
                    "RETURN n.node_id AS s, r.relation AS r, m.node_id AS d, coalesce(r.source,'') AS src ORDER BY r,d,src",
                    g=GR, n=src, rel=rel))
            se[src] = [(row["s"], row["r"], row["d"], row["src"]) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier}))

def path_hash(paths):
    return hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()).hexdigest()

# ── Manifest generation ──
def gen_manifest(hop, by_src, all_srcs):
    print(f"  Generating hop={hop}...", flush=True)
    rng = random.Random(20260707 + hop)
    queries = []; seeds_used = set()
    total_tries = 0
    for _ in range(NQ * 50):
        if len(queries) >= NQ: break
        seed = rng.choice(all_srcs)
        if seed in seeds_used and total_tries < 5000: continue
        total_tries += 1

        # Find a valid relation_path of length 'hop'
        h1 = sorted(by_src.get(seed, []), key=lambda e: (e["relation"], e["dst_id"], e["source"]))[:FAN]
        if not h1: continue

        # Try to find valid path chains
        found = False
        for e1 in h1:
            if found: break
            prev_dst = e1["dst_id"]
            rp = [e1["relation"]]
            valid = True
            for step in range(1, hop):
                h_next = sorted(by_src.get(prev_dst, []), key=lambda e: (e["relation"], e["dst_id"], e["source"]))[:FAN]
                if not h_next: valid = False; break
                rp.append(h_next[0]["relation"])
                prev_dst = h_next[0]["dst_id"]
            if valid:
                # Verify with Cassandra
                paths = cass_traverse(seed, rp)
                if len(paths) > 0 and seed not in seeds_used:
                    seeds_used.add(seed)
                    qid = f"hd-h{hop}-{len(queries):06d}"
                    queries.append({
                        "query_id": qid, "graph_id": GR, "seed_id": seed,
                        "relation_path": rp, "hop": hop, "fanout": FAN, "cycle_policy": "path",
                        "expected_path_count": len(paths),
                        "expected_path_hash": path_hash(paths),
                    })
                    found = True
        if len(queries) % 64 == 0:
            print(f"    hop={hop}: {len(queries)}/{NQ}", flush=True)

    print(f"    hop={hop}: {len(queries)} queries from {len(seeds_used)} seeds", flush=True)
    return queries[:NQ]

def manifest_summary(queries, hop, by_src):
    empty = sum(1 for q in queries if q["expected_path_count"] == 0)
    seeds = set(q["seed_id"] for q in queries)
    pc = sorted([q["expected_path_count"] for q in queries])
    rc = Counter(str(q["relation_path"]) for q in queries)
    ms = {
        "hop": hop, "query_count": len(queries),
        "distinct_seeds": len(seeds), "empty_queries": empty,
        "result_path_min": pc[0], "result_path_max": pc[-1],
        "result_path_p50": pc[len(pc)//2], "result_path_p95": pc[int(len(pc)*0.95)],
        "relation_path_distribution_top10": rc.most_common(10),
    }
    return ms

# ── Generate all manifests ──
print("\n[2] Generating manifests", flush=True)
manifests = {}
for hop in [1, 2, 3, 4]:
    qs = gen_manifest(hop, by_src, all_srcs)
    path = MANIFEST_DIR / f"sysaxis_1m_manifest_h{hop}.jsonl"
    with path.open("w") as f:
        for q in qs: f.write(json.dumps(q, ensure_ascii=False) + "\n")
    manifests[hop] = qs

# Copy h2 as alias
shutil.copy(MANIFEST_DIR / "sysaxis_1m_manifest_h2.jsonl", MANIFEST_DIR / "sysaxis_1m_manifest_h2.jsonl")

# Manifest summaries
print("\n[3] Manifest summaries", flush=True)
summaries = {}
for hop in [1, 2, 3, 4]:
    ms = manifest_summary(manifests[hop], hop, by_src)
    with (OUT / f"manifest_summary_h{hop}.json").open("w") as f: json.dump(ms, f, indent=2)
    summaries[hop] = ms
    print(f"  hop={hop}: q={ms['query_count']} seeds={ms['distinct_seeds']} path p50={ms['result_path_p50']} p95={ms['result_path_p95']} max={ms['result_path_max']}", flush=True)

diff = {
    f"h{hop}": {"queries": summaries[hop]["query_count"], "seeds": summaries[hop]["distinct_seeds"],
                 "path_min": summaries[hop]["result_path_min"], "path_p50": summaries[hop]["result_path_p50"],
                 "path_p95": summaries[hop]["result_path_p95"], "path_max": summaries[hop]["result_path_max"]}
    for hop in [1, 2, 3, 4]
}
with (OUT / "manifest_difficulty_comparison.json").open("w") as f: json.dump(diff, f, indent=2)

# ── Semantic gates ──
print("\n[4] Semantic gates (64 queries per hop)", flush=True)
for hop in [1, 2, 3, 4]:
    rng_sem = random.Random(42)
    checks = rng_sem.sample(manifests[hop], min(SC, len(manifests[hop])))
    cass_ok = 0; neo_ok = 0; empty = 0
    for q in checks:
        cass_p = cass_traverse(q["seed_id"], q["relation_path"])
        neo_p = neo_traverse(q["seed_id"], q["relation_path"])
        if len(cass_p) == 0: empty += 1; continue
        if path_hash(cass_p) == q["expected_path_hash"]: cass_ok += 1
        if set(cass_p) == set(neo_p): neo_ok += 1

    sg = {
        "hop": hop, "checked": len(checks), "empty": empty,
        "cassandra_mismatch": len(checks) - empty - cass_ok,
        "neo4j_disagreement": len(checks) - empty - neo_ok,
        "all_pass": empty == 0 and cass_ok == len(checks) - empty and neo_ok == len(checks) - empty,
    }
    with (OUT / f"semantic_gate_h{hop}_cassandra_neo4j.json").open("w") as f: json.dump(sg, f, indent=2)
    print(f"  hop={hop}: checked={len(checks)} empty={empty} cass_mismatch={sg['cassandra_mismatch']} neo_disagree={sg['neo4j_disagreement']} {'PASS' if sg['all_pass'] else 'FAIL'}", flush=True)

d.close(); c.shutdown()
print("\nDone.", flush=True)
