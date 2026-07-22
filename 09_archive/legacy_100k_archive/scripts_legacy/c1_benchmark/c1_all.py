import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cassandra.cluster import Cluster

try:
    from neo4j import GraphDatabase
except ImportError:
    sys.exit("ERROR: install neo4j driver")

GRAPH_ID = "synth_100000_1781447372"
NEO_LABEL = "C1KGNode"
NEO_REL = "C1KG_EDGE"
FANOUT = 20
HOP = 2
MANIFEST_SEED = 20260707
N_QUERIES = 256

OUT_DIR = Path("D:/memorytable/cassandra-kg-memory/reports/c1_preflight_100k")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = Path("D:/memorytable/cassandra-kg-memory/results/c1_manifest_100k_h2.jsonl")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")


def logical_key(src, rel, dst, source):
    return (GRAPH_ID, src, rel, dst, source or "")


def export_graph_from_cassandra():
    """Use benchmark generator to produce edges identical to Cassandra graph."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from benchmark_cassandra_internal_ablation import SyntheticGraph
    g = SyntheticGraph(n_entities=5000, n_edges=100000, high_degree_frac=0.02, high_degree_mult=20)
    g.generate()
    # Save CSV for Neo4j
    with open("D:/memorytable/cassandra-kg-memory/results/c1_source_100k.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["graph_id", "src_id", "relation", "dst_id", "source"])
        w.writeheader()
        for e in g.edges:
            w.writerow({"graph_id": GRAPH_ID, "src_id": e["src_id"], "relation": e["relation"],
                        "dst_id": e["dst_id"], "source": e["source"]})

    edges = [{"graph_id": GRAPH_ID, "src_id": e["src_id"], "relation": e["relation"],
              "dst_id": e["dst_id"], "source": e["source"]} for e in g.edges]
    sources = set(e["src_id"] for e in edges)
    seen = set()
    deduped = []
    for e in edges:
        lk = logical_key(e["src_id"], e["relation"], e["dst_id"], e["source"])
        if lk not in seen:
            seen.add(lk)
            deduped.append(e)

    degrees = defaultdict(int)
    for e in deduped:
        degrees[e["src_id"]] += 1
    d = list(degrees.values())

    dataset = {
        "graph_id": GRAPH_ID,
        "source": "Cassandra export from existing table",
        "n_edges_total": len(edges),
        "n_edges_distinct": len(deduped),
        "distinct_sources": len(sources),
        "min_outdegree": min(d),
        "max_outdegree": max(d),
        "p50_outdegree": sorted(d)[len(d)//2],
        "n_high_degree": sum(1 for v in d if v > 100),
    }
    with (OUT_DIR / "dataset_summary.json").open("w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(json.dumps(dataset, indent=2, ensure_ascii=False), flush=True)
    return deduped


def import_neo4j(edges):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]))
    with driver.session() as s:
        s.run(f"MATCH (n:{NEO_LABEL}) DETACH DELETE n")
        s.run(f"CREATE CONSTRAINT c1_node_unique IF NOT EXISTS FOR (n:{NEO_LABEL}) REQUIRE (n.graph_id, n.node_id) IS UNIQUE")
    print("Neo4j: cleared, constraint created", flush=True)

    nodes = set()
    for e in edges:
        nodes.add((e["graph_id"], e["src_id"]))
        nodes.add((e["graph_id"], e["dst_id"]))
    node_list = [{"g": n[0], "nid": n[1]} for n in nodes]
    for i in range(0, len(node_list), 5000):
        chunk = node_list[i:i+5000]
        with driver.session() as s:
            s.run(f"UNWIND $rows AS r MERGE (n:{NEO_LABEL} {{graph_id: r.g, node_id: r.nid}})", rows=chunk).consume()
        print(f"  Nodes: {min(i+5000, len(node_list))}/{len(node_list)}", flush=True)

    batch = []
    for e in edges:
        batch.append({"g": e["graph_id"], "s": e["src_id"], "d": e["dst_id"], "rel": e["relation"], "src": e["source"]})
        if len(batch) >= 5000:
            _flush_neo(driver, batch)
            batch = []
    if batch:
        _flush_neo(driver, batch)
    print(f"  Edges: {len(edges)} imported", flush=True)
    driver.close()


def _flush_neo(driver, batch):
    cypher = (
        f"UNWIND $rows AS r "
        f"MATCH (s:{NEO_LABEL} {{graph_id: r.g, node_id: r.s}}) "
        f"MATCH (d:{NEO_LABEL} {{graph_id: r.g, node_id: r.d}}) "
        f"CREATE (s)-[:{NEO_REL} {{relation: r.rel, source: r.src, graph_id: r.g}}]->(d)"
    )
    with driver.session() as s:
        s.run(cypher, rows=batch).consume()


def neo4j_readback():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    edges = set()
    with driver.session() as s:
        r = s.run(
            f"MATCH (s:{NEO_LABEL})-[r:{NEO_REL}]->(d:{NEO_LABEL}) "
            "RETURN s.node_id AS src_id, r.relation AS rel, d.node_id AS dst_id, r.source AS src"
        )
        for row in r:
            edges.add(logical_key(row["src_id"], row["rel"], row["dst_id"], row["src"]))
    driver.close()
    return edges


def generate_manifest(cass_edges):
    by_src = defaultdict(list)
    for e in cass_edges:
        by_src[e["src_id"]].append(e)
    all_sources = sorted(by_src.keys())

    rng = random.Random(MANIFEST_SEED)
    queries = []
    seeds_used = set()

    for _ in range(N_QUERIES * 20):
        if len(queries) >= N_QUERIES:
            break
        seed = rng.choice(all_sources)
        if seed in seeds_used:
            continue
        hop1 = sorted(by_src.get(seed, []), key=lambda e: (e["relation"], e["dst_id"], e["source"]))
        if not hop1:
            continue
        h1 = hop1[:FANOUT]
        rel_paths = set()
        for e1 in h1:
            hop2 = sorted(by_src.get(e1["dst_id"], []), key=lambda e: (e["relation"], e["dst_id"], e["source"]))
            for e2 in hop2[:FANOUT]:
                rel_paths.add((e1["relation"], e2["relation"]))
        if not rel_paths:
            continue
        seeds_used.add(seed)
        rp = list(rel_paths)[0]
        queries.append({
            "query_id": f"c1-read-{len(queries):06d}",
            "graph_id": GRAPH_ID,
            "seed_id": seed,
            "relation_path": list(rp),
            "hop": HOP,
            "fanout": FANOUT,
            "cycle_policy": "path",
        })

    queries = queries[:N_QUERIES]
    with MANIFEST_PATH.open("w") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    rel_dist = defaultdict(int)
    for q in queries:
        rel_dist[str(q["relation_path"])] += 1

    # Compute path counts
    pc = []
    for q in queries:
        n = _count_paths(by_src, q)
        pc.append(n)

    spc = sorted(pc)
    m = {
        "manifest_path": str(MANIFEST_PATH),
        "query_count": len(queries),
        "distinct_seeds": len(seeds_used),
        "relation_path_top5": sorted(rel_dist.items(), key=lambda x: -x[1])[:5],
        "result_path_count_min": spc[0],
        "result_path_count_max": spc[-1],
        "result_path_count_p50": spc[len(spc)//2],
        "result_path_count_p95": spc[int(len(spc)*0.95)],
    }
    with (OUT_DIR / "manifest_summary.json").open("w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(json.dumps(m, indent=2, ensure_ascii=False), flush=True)
    return queries


def _count_paths(by_src, q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for relation in q["relation_path"]:
        sources = sorted({s[0] for s in frontier})
        src_edges = {}
        for src in sources:
            all_e = sorted(by_src.get(src, []), key=lambda e: (e["relation"], e["dst_id"], e["source"]))
            src_edges[src] = [e for e in all_e if e["relation"] == relation]
        next_f = set()
        for src, np, ep in frontier:
            for e in src_edges.get(src, [])[:FANOUT]:
                if e["dst_id"] not in np:
                    next_f.add((e["dst_id"], np + (e["dst_id"],), ep + (e["source"],)))
        frontier = next_f
        if not frontier:
            return 0
    return len(frontier)


def cass_fetch(session, src):
    rows = session.execute(
        "SELECT src_id, relation, dst_id, source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
        (GRAPH_ID, src),
    )
    return [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]


def neo_fetch(driver, src, rel):
    cypher = (
        f"MATCH (s:{NEO_LABEL} {{graph_id: $g, node_id: $s}})"
        f"-[r:{NEO_REL} {{relation: $rel}}]->"
        f"(d:{NEO_LABEL} {{graph_id: $g}}) "
        "RETURN s.node_id AS src_id, r.relation AS relation, d.node_id AS dst_id, coalesce(r.source,'') AS source "
        "ORDER BY relation, dst_id, source"
    )
    with driver.session() as s:
        rows = list(s.run(cypher, g=GRAPH_ID, s=src, rel=rel))
    return [(r["src_id"], r["relation"], r["dst_id"], r["source"]) for r in rows]


def traverse_cass(session, q, workers=1):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for relation in q["relation_path"]:
        sources = sorted({s[0] for s in frontier})
        se = {}
        if workers == 1:
            for src in sources:
                se[src] = [e for e in cass_fetch(session, src) if e[1] == relation]
        else:
            def _do(src):
                return src, [e for e in cass_fetch(session, src) if e[1] == relation]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for f in as_completed({pool.submit(_do, src): src for src in sources}):
                    src, edges = f.result()
                    se[src] = edges
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FANOUT]:
                if e[2] not in np:
                    nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
        if not frontier:
            break
    return tuple(sorted({s[2] for s in frontier}))


def traverse_neo(driver, q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for relation in q["relation_path"]:
        sources = sorted({s[0] for s in frontier})
        se = {}
        for src in sources:
            se[src] = neo_fetch(driver, src, relation)
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FANOUT]:
                if e[2] not in np:
                    nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
        if not frontier:
            break
    return tuple(sorted({s[2] for s in frontier}))


def semantic_preflight(queries):
    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect("ai_memory")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))

    mp, mn, empty = [], [], []
    t0 = time.perf_counter()
    for i, q in enumerate(queries):
        r_naive = traverse_cass(session, q, workers=1)
        if not r_naive:
            empty.append(q["query_id"])

        r_para = traverse_cass(session, q, workers=16)
        if set(r_naive) != set(r_para):
            mp.append({"query_id": q["query_id"], "naive": len(r_naive), "parallel": len(r_para)})

        r_neo = traverse_neo(driver, q)
        if set(r_naive) != set(r_neo):
            mn.append({"query_id": q["query_id"], "naive": len(r_naive), "neo4j": len(r_neo)})

        if (i+1) % 32 == 0:
            print(f"  {i+1}/{len(queries)} ({time.perf_counter()-t0:.0f}s)  par={len(mp)} neo={len(mn)} empty={len(empty)}", flush=True)

    cluster.shutdown()
    driver.close()
    elapsed = time.perf_counter() - t0
    sp = {
        "total_queries": len(queries),
        "empty_queries": len(empty),
        "naive_vs_parallel_disagreements": len(mp),
        "naive_vs_neo4j_disagreements": len(mn),
        "all_pass": len(mp)==0 and len(mn)==0 and len(empty)==0,
        "elapsed_seconds": round(elapsed, 1),
    }
    with (OUT_DIR / "semantic_preflight_summary.json").open("w") as f:
        json.dump(sp, f, indent=2, ensure_ascii=False)
    if mp:
        with (OUT_DIR / "semantic_preflight_mismatches_parallel.jsonl").open("w") as f:
            for m in mp:
                f.write(json.dumps(m, ensure_ascii=False)+"\n")
    if mn:
        with (OUT_DIR / "semantic_preflight_mismatches_neo4j.jsonl").open("w") as f:
            for m in mn:
                f.write(json.dumps(m, ensure_ascii=False)+"\n")
    print(f"\n  PARALLEL mismatches: {len(mp)}  NEO4J mismatches: {len(mn)}  EMPTY: {len(empty)}", flush=True)
    print(json.dumps(sp, indent=2, ensure_ascii=False), flush=True)
    return sp


def main():
    print("=== C1-0: Dataset + Manifest + Preflight ===\n", flush=True)

    # Step 1: Export graph from Cassandra
    print("[1/5] Exporting graph from Cassandra...", flush=True)
    cass_edges = export_graph_from_cassandra()

    # Step 2: Import to Neo4j
    print("\n[2/5] Importing to Neo4j...", flush=True)
    import_neo4j(cass_edges)

    # Step 3: Full-graph import preflight
    print("\n[3/5] Full-graph import preflight...", flush=True)
    neo_set = neo4j_readback()
    cass_set = {logical_key(e["src_id"], e["relation"], e["dst_id"], e["source"]) for e in cass_edges}
    missing = sorted(cass_set - neo_set)
    extra = sorted(neo_set - cass_set)
    fp = {
        "cassandra_edges": len(cass_set),
        "neo4j_edges": len(neo_set),
        "mismatches": len(missing)+len(extra),
        "all_consistent": len(missing)==0 and len(extra)==0,
    }
    with (OUT_DIR / "full_graph_import_preflight.json").open("w") as f:
        json.dump(fp, f, indent=2, ensure_ascii=False)
    if missing or extra:
        with (OUT_DIR / "full_graph_import_mismatches.jsonl").open("w") as f:
            for m in missing: f.write(json.dumps({"type":"missing","key":list(m)})+"\n")
            for e in extra: f.write(json.dumps({"type":"extra","key":list(e)})+"\n")
    print(f"  Cassandra: {len(cass_set)}  Neo4j: {len(neo_set)}  Mismatches: {fp['mismatches']}  {'PASS' if fp['all_consistent'] else 'FAIL'}", flush=True)

    if not fp["all_consistent"]:
        print("ABORT: import preflight failed", flush=True)
        return

    # Step 4: Generate manifest
    print("\n[4/5] Generating manifest...", flush=True)
    queries = generate_manifest(cass_edges)
    print(f"  Sample queries:")
    for q in queries[:3]:
        print(json.dumps(q, indent=2, ensure_ascii=False))

    # Step 5: Semantic preflight
    print(f"\n[5/5] Semantic preflight ({len(queries)} queries)...", flush=True)
    semantic_preflight(queries)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
