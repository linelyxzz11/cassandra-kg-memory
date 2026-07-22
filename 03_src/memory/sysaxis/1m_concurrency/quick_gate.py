"""Quick dual hash gate (64 queries)"""
import json, hashlib, random
from pathlib import Path
from cassandra.cluster import Cluster
from neo4j import GraphDatabase

PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT  = PROJ / "reports/sysaxis_1m_concurrency_final"
GR   = "c3_scale_1M_seed42"

c = Cluster(["127.0.0.1"], port=9042)
s = c.connect("ai_memory")
d = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "password123"))

queries = [json.loads(line) for line in open(PROJ / "results/c3_manifest_scale_1m_h2.jsonl")]
rng = random.Random(42)
checks = rng.sample(queries, 64)

def cass_read(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        for src in sources:
            rows = s.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
            se[src] = [(str(r.src_id), str(r.relation), str(r.dst_id), str(r.source or "")) for r in rows if r.relation == rel]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:20]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
    return tuple(sorted({f[2] for f in frontier}))

def neo_read(q):
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
    return tuple(sorted({f[2] for f in frontier}))

for backend, name in [(cass_read, "cassandra"), (neo_read, "neo4j")]:
    empty = 0; mm = 0
    for q in checks:
        paths = backend(q)
        if len(paths) == 0: empty += 1
        else:
            h = hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()).hexdigest()
            if h != q["expected_path_hash"]: mm += 1
    hg = {"checked": 64, "empty": empty, "mismatch": mm, "all_pass": empty == 0 and mm == 0}
    with (OUT / f"hash_gate_before_{name}.json").open("w") as f:
        json.dump(hg, f, indent=2)
    print(f"Hash gate {name}: 64 checked empty={empty} mismatch={mm} {'PASS' if hg['all_pass'] else 'FAIL'}")

d.close(); c.shutdown()
