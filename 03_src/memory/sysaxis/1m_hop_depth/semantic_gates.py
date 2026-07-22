"""Semantic gates: hop 1-4, Cassandra vs Neo4j, 64 queries each."""
import json, hashlib, random
from pathlib import Path
from cassandra.cluster import Cluster
from neo4j import GraphDatabase

PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT  = PROJ / "reports/sysaxis_1m_hop_depth_final"
GR   = "c3_scale_1M_seed42"
FAN  = 20; SC = 64

c = Cluster(["127.0.0.1"], port=9042)
s = c.connect("ai_memory")
d = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "password123"))

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
                    "RETURN n.node_id AS s, r.relation AS r, m.node_id AS d, coalesce(r.source,'') AS src ORDER BY r, d, src",
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

for hop in [1, 2, 3, 4]:
    qs = [json.loads(line) for line in open(PROJ / f"results/sysaxis_1m_manifest_h{hop}.jsonl")]
    rng = random.Random(42)
    checks = rng.sample(qs, min(SC, len(qs)))
    cass_mm = 0; neo_da = 0; empty = 0
    for q in checks:
        cp = cass_traverse(q["seed_id"], q["relation_path"])
        np2 = neo_traverse(q["seed_id"], q["relation_path"])
        if len(cp) == 0: empty += 1; continue
        if path_hash(cp) != q["expected_path_hash"]: cass_mm += 1
        if set(cp) != set(np2): neo_da += 1
    sg = {
        "hop": hop, "checked": len(checks), "empty": empty,
        "cassandra_mismatch": cass_mm, "neo4j_disagreement": neo_da,
        "all_pass": empty == 0 and cass_mm == 0 and neo_da == 0,
    }
    with (OUT / f"semantic_gate_h{hop}_cassandra_neo4j.json").open("w") as f:
        json.dump(sg, f, indent=2)
    print(f"  hop={hop}: checked={len(checks)} empty={empty} cass_mm={cass_mm} neo_da={neo_da} {'PASS' if sg['all_pass'] else 'FAIL'}")

d.close(); c.shutdown()
print("Done.")
