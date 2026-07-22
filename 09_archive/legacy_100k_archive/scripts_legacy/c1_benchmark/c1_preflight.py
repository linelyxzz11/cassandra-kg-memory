import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cassandra.cluster import Cluster

try:
    from neo4j import GraphDatabase
    NEO4J_OK = True
except ImportError:
    NEO4J_OK = False

GRAPH_ID = "c1_synth_100k_seed42"
FANOUT = 20
MANIFEST_PATH = Path("D:/memorytable/cassandra-kg-memory/results/c1_manifest_100k_h2.jsonl")
REPORT_DIR = Path("D:/memorytable/cassandra-kg-memory/reports/c1_preflight_100k")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest():
    queries = []
    with MANIFEST_PATH.open() as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def logical_key(src, rel, dst, source):
    return (GRAPH_ID, src, rel, dst, source or "")


def cassandra_fetch(session, src_id):
    rows = session.execute(
        "SELECT src_id, relation, dst_id, source FROM kg_edges_by_src "
        "WHERE graph_id=%s AND src_id=%s",
        (GRAPH_ID, src_id),
    )
    edges = [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]
    edges.sort(key=lambda e: (e[1], e[2], e[3]))
    return edges


def neo4j_fetch(driver, src_id, relation):
    cypher = (
        "MATCH (s:C1KGNode {graph_id: $g, node_id: $s})"
        "-[r:C1KG_EDGE {relation: $rel}]->"
        "(d:C1KGNode {graph_id: $g}) "
        "RETURN s.node_id AS src_id, r.relation AS relation, "
        "d.node_id AS dst_id, coalesce(r.source, '') AS source "
        "ORDER BY relation, dst_id, source"
    )
    with driver.session() as session:
        rows = list(session.run(cypher, g=GRAPH_ID, s=src_id, rel=relation))
    return [(r["src_id"], r["relation"], r["dst_id"], r["source"]) for r in rows]


def traverse(session, driver, query, workers=1, use_neo4j=False):
    rel_path = query["relation_path"]
    frontier = {(query["seed_id"], (query["seed_id"],), ())}

    for _, relation in enumerate(rel_path):
        sources = sorted({s[0] for s in frontier})
        src_edges = {}
        if workers == 1 and not use_neo4j:
            for src in sources:
                all_edges = cassandra_fetch(session, src)
                src_edges[src] = [e for e in all_edges if e[1] == relation]
        elif not use_neo4j:
            def _do(src):
                all_edges = cassandra_fetch(session, src)
                return src, [e for e in all_edges if e[1] == relation]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_do, src): src for src in sources}
                for f in as_completed(futures):
                    src, edges = f.result()
                    src_edges[src] = edges
        else:
            for src in sources:
                src_edges[src] = neo4j_fetch(driver, src, relation)

        next_frontier = set()
        for src, node_path, edge_path in frontier:
            edges = src_edges.get(src, [])
            candidates = [e for e in edges if e[2] not in node_path]
            for _, rel, dst, source in candidates[:FANOUT]:
                next_frontier.add((dst, node_path + (dst,), edge_path + (source,)))
        frontier = next_frontier
        if not frontier:
            break

    return tuple(sorted({s[2] for s in frontier}))


def main():
    print("=== C1 Semantic Preflight ===")
    queries = load_manifest()
    print(f"Loaded {len(queries)} queries")

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect("ai_memory")

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD")
    if not pwd:
        print("ERROR: NEO4J_PASSWORD not set")
        sys.exit(1)
    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    mismatches_parallel = []
    mismatches_neo4j = []
    empty_queries = []

    t0 = time.perf_counter()
    for i, q in enumerate(queries):
        r_naive = traverse(session, driver, q, workers=1, use_neo4j=False)
        if len(r_naive) == 0:
            empty_queries.append(q["query_id"])

        r_para = traverse(session, driver, q, workers=16, use_neo4j=False)
        if set(r_naive) != set(r_para):
            mismatches_parallel.append({
                "query_id": q["query_id"],
                "naive_paths": len(r_naive),
                "parallel_paths": len(r_para),
                "missing": sorted(set(r_naive) - set(r_para)),
                "extra": sorted(set(r_para) - set(r_naive)),
            })

        r_neo4j = traverse(session, driver, q, workers=1, use_neo4j=True)
        if set(r_naive) != set(r_neo4j):
            mismatches_neo4j.append({
                "query_id": q["query_id"],
                "naive_paths": len(r_naive),
                "neo4j_paths": len(r_neo4j),
                "missing": sorted(set(r_naive) - set(r_neo4j)),
                "extra": sorted(set(r_neo4j) - set(r_naive)),
            })

        if (i + 1) % 32 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i + 1}/{len(queries)} ({elapsed:.0f}s)   par_mismatch={len(mismatches_parallel)}  neo4j_mismatch={len(mismatches_neo4j)}  empty={len(empty_queries)}")

    cluster.shutdown()
    driver.close()

    elapsed = time.perf_counter() - t0
    mp = len(mismatches_parallel)
    mn = len(mismatches_neo4j)

    preflight = {
        "total_queries": len(queries),
        "empty_queries": len(empty_queries),
        "naive_vs_parallel_disagreements": mp,
        "naive_vs_neo4j_disagreements": mn,
        "naive_vs_parallel_all_pass": mp == 0,
        "naive_vs_neo4j_all_pass": mn == 0,
        "all_pass": mp == 0 and mn == 0 and len(empty_queries) == 0,
        "elapsed_seconds": round(elapsed, 1),
    }

    with (REPORT_DIR / "semantic_preflight_summary.json").open("w") as f:
        json.dump(preflight, f, indent=2, ensure_ascii=False)

    if mp:
        with (REPORT_DIR / "semantic_preflight_mismatches_parallel.jsonl").open("w") as f:
            for m in mismatches_parallel:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    if mn:
        with (REPORT_DIR / "semantic_preflight_mismatches_neo4j.jsonl").open("w") as f:
            for m in mismatches_neo4j:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"\n===== SEMANTIC PREFLIGHT =====")
    print(f"  Queries: {len(queries)}  Empty: {len(empty_queries)}")
    print(f"  na\"     (char replaced to avoid self-reference)ive_vs_parallel mismatches: {mp}")
    print(f"  naive_vs_neo4j mismatches: {mn}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  all_pass: {preflight['all_pass']}")
    print(json.dumps(preflight, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
