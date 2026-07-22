"""Neo4j import: c3_source_scale_1M.csv -> C3KGNode/C3KG_EDGE with graph_id=c3_scale_1M_seed42.
Does NOT delete old dense graph (c3_synth_1M_seed42)."""
import csv, json, hashlib, time
from pathlib import Path
from collections import defaultdict
from neo4j import GraphDatabase

PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT = PROJ / "reports/sysaxis_1m_write_ratio_final"
CSV_PATH = PROJ / "results/c3_source_scale_1M.csv"
MANIFEST = PROJ / "results/c3_manifest_scale_1m_h2.jsonl"
GR_SCALE = "c3_scale_1M_seed42"

d = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "password123"))

# ── A: Pre-check ──
print("[A] Pre-check Neo4j state", flush=True)
precheck = {"nodes_by_graph": {}, "edges_by_graph": {}}
with d.session() as s:
    for row in s.run("MATCH (n:C3KGNode) RETURN n.graph_id AS g, count(n) AS c"):
        precheck["nodes_by_graph"][row["g"]] = row["c"]
    for row in s.run("MATCH ()-[r:C3KG_EDGE]->() RETURN r.graph_id AS g, count(r) AS c"):
        precheck["edges_by_graph"][row["g"]] = row["c"]
with (OUT / "neo4j_scale_import_precheck.json").open("w") as f:
    json.dump(precheck, f, indent=2)
print(f"  Nodes: {precheck['nodes_by_graph']}", flush=True)
print(f"  Edges: {precheck['edges_by_graph']}", flush=True)

# ── B: Clean partial scale data if any ──
if GR_SCALE in precheck["nodes_by_graph"]:
    n_partial = precheck["nodes_by_graph"][GR_SCALE]
    print(f"[B] Found {n_partial} partial C3KGNode for {GR_SCALE}. Cleaning...", flush=True)
    with d.session() as s:
        s.run(f"MATCH (n:C3KGNode {{graph_id: '{GR_SCALE}'}}) DETACH DELETE n")
    print(f"  Deleted", flush=True)
else:
    print(f"[B] No partial data for {GR_SCALE}", flush=True)

# ── C: Import ──
print(f"[C] Loading CSV...", flush=True)
edges = []
nodes_set = set()
with CSV_PATH.open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        edges.append(r)
        nodes_set.add(r["src_id"]); nodes_set.add(r["dst_id"])
print(f"  {len(edges)} edges, {len(nodes_set)} distinct nodes", flush=True)

# Import nodes
print(f"  Importing nodes...", flush=True)
nl = [{"g": GR_SCALE, "nid": n} for n in nodes_set]
t0 = time.perf_counter()
for i in range(0, len(nl), 5000):
    with d.session() as s:
        s.run("UNWIND $rows AS r MERGE (n:C3KGNode {graph_id: r.g, node_id: r.nid})",
              rows=nl[i:i+5000]).consume()
print(f"  {len(nl)} nodes in {time.perf_counter()-t0:.0f}s", flush=True)

# Import edges
print(f"  Importing edges...", flush=True)
for i in range(0, len(edges), 5000):
    batch = [{"g": GR_SCALE, "s": e["src_id"], "d": e["dst_id"], "rel": e["relation"], "src": e["source"]}
             for e in edges[i:i+5000]]
    with d.session() as s:
        s.run(
            "UNWIND $rows AS r "
            "MATCH (s:C3KGNode {graph_id: r.g, node_id: r.s}) "
            "MATCH (d:C3KGNode {graph_id: r.g, node_id: r.d}) "
            "CREATE (s)-[:C3KG_EDGE {graph_id: r.g, relation: r.rel, source: r.src}]->(d)",
            rows=batch).consume()
    if (i + 5000) % 50000 == 0:
        print(f"  {i+5000}/{len(edges)} ({time.perf_counter()-t0:.0f}s)", flush=True)

elapsed = time.perf_counter() - t0
print(f"  {len(edges)} edges in {elapsed:.0f}s", flush=True)

# ── D: Preflight ──
print(f"[D] Preflight", flush=True)
csv_set = set((e["src_id"], e["relation"], e["dst_id"], e["source"]) for e in edges)
neo_set = set(); neo_raw = 0
with d.session() as s:
    rows = s.run(f"MATCH (s:C3KGNode {{graph_id: '{GR_SCALE}'}})-[r:C3KG_EDGE]->(d:C3KGNode {{graph_id: '{GR_SCALE}'}}) "
                 "RETURN s.node_id AS sid, r.relation AS rel, d.node_id AS did, r.source AS src")
    for row in rows:
        neo_set.add((row["sid"], row["rel"], row["did"], row["src"] or ""))
        neo_raw += 1

node_count = 0
with d.session() as s:
    node_count = s.run(f"MATCH (n:C3KGNode {{graph_id: '{GR_SCALE}'}}) RETURN count(n)").single()[0]

dup = neo_raw - len(neo_set); miss = len(csv_set - neo_set); extra = len(neo_set - csv_set)
preflight = {
    "source_csv_edges": len(edges),
    "source_csv_distinct_logical_edges": len(csv_set),
    "neo4j_nodes": node_count,
    "neo4j_edges": neo_raw,
    "duplicate_relationships": dup,
    "missing_edges": miss, "extra_edges": extra,
    "import_elapsed_s": round(elapsed, 1), "batch_size": 5000,
    "all_consistent": miss == 0 and extra == 0,
}
with (OUT / "neo4j_scale_import_summary.json").open("w") as f:
    json.dump(preflight, f, indent=2)
print(f"  Nodes={node_count} Edges={neo_raw} Dup={dup} Miss={miss} Extra={extra}", flush=True)
print(f"  {'PASS' if preflight['all_consistent'] else 'FAIL'}", flush=True)

if not preflight["all_consistent"]:
    d.close(); exit(1)

# ── E: Hash gate ──
print(f"[E] Hash gate (256 queries)", flush=True)
queries = [json.loads(line) for line in open(MANIFEST)]
empty = 0; mm = 0
t1 = time.perf_counter()
for i, q in enumerate(queries):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        for src in sources:
            with d.session() as s:
                rows = list(s.run(
                    f"MATCH (n:C3KGNode {{graph_id: $g, node_id: $n}})-[r:C3KG_EDGE {{relation: $rel}}]->(m:C3KGNode {{graph_id: $g}}) "
                    "RETURN n.node_id AS s, r.relation AS r, m.node_id AS d, coalesce(r.source,'') AS src ORDER BY r,d,src",
                    g=GR_SCALE, n=src, rel=rel))
            se[src] = [(row["s"], row["r"], row["d"], row["src"]) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:20]:
                if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
    paths = tuple(sorted({f[2] for f in frontier}))
    if len(paths) == 0: empty += 1
    else:
        h = hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()).hexdigest()
        if h != q["expected_path_hash"]: mm += 1
    if (i+1) % 64 == 0:
        print(f"  {i+1}/256 empty={empty} mismatch={mm} ({time.perf_counter()-t1:.0f}s)", flush=True)

elapsed_hg = time.perf_counter() - t1
hg = {"checked": 256, "empty": empty, "mismatch": mm, "all_pass": empty == 0 and mm == 0,
      "elapsed_s": round(elapsed_hg, 1), "graph_id": GR_SCALE}
with (OUT / "neo4j_hash_gate_scale_1m.json").open("w") as f:
    json.dump(hg, f, indent=2)
print(f"  Hash gate: empty={empty} mismatch={mm} {'PASS' if hg['all_pass'] else 'FAIL'} ({elapsed_hg:.0f}s)", flush=True)

d.close()
print("Done.", flush=True)
