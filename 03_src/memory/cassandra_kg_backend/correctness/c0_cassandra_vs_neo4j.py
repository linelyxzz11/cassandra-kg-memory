import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from cassandra.cluster import Cluster

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


CASSANDRA_QUERY = (
    "SELECT graph_id, src_id, relation, dst_id, source "
    "FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s"
)
GRAPH_ID = "synth_100000_1781447372"
SEED = "entity_0"
RELATION_PATH = ["attended", "attended", "attended", "bought"]
HOP = 4
FANOUT = 20

REPORT_DIR = Path("D:/memorytable/cassandra-kg-memory/reports/c0_cassandra_neo4j_smoke")
CLOSURE_FILE = REPORT_DIR / "exported_closure.jsonl"


def logical_key(graph_id, src_id, relation, dst_id, source):
    return (graph_id, src_id, relation, dst_id, source or "")


def cassandra_naive_traversal():
    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect("ai_memory")

    frontier = {(SEED, (SEED,), ())}
    all_sources = []

    for depth, relation in enumerate(RELATION_PATH):
        sources = sorted({s[0] for s in frontier})
        all_sources.extend(sources)
        src_edges = {}
        for src in sources:
            rows = session.execute(CASSANDRA_QUERY, (GRAPH_ID, src))
            edges = [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]
            edges.sort(key=lambda e: (e[1], e[2], e[3]))
            filtered = [e for e in edges if e[1] == relation]
            src_edges[src] = filtered

        next_frontier = set()
        for src, node_path, edge_path in frontier:
            edges = src_edges.get(src, [])
            candidates = [e for e in edges if e[2] not in node_path]
            for _, rel, dst, source in candidates[:FANOUT]:
                next_frontier.add((dst, node_path + (dst,), edge_path + (source,)))
        frontier = next_frontier
        if not frontier:
            break

    paths = tuple(sorted({s[2] for s in frontier}))
    cluster.shutdown()
    return all_sources, paths, len(paths)


def export_closure():
    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect("ai_memory")

    all_sources, cass_paths, n_cass_paths = cassandra_naive_traversal()
    unique_sources = sorted(set(all_sources))
    print(f"Cassandra traversal: {n_cass_paths} paths, {len(all_sources)} queries, {len(unique_sources)} distinct sources")
    print(f"Sources: {unique_sources[:5]}..." if len(unique_sources) > 5 else f"Sources: {unique_sources}")

    logical_set = set()
    exported = []
    for src in unique_sources:
        rows = session.execute(CASSANDRA_QUERY, (GRAPH_ID, src))
        for r in rows:
            lk = logical_key(r.graph_id, r.src_id, r.relation, r.dst_id, r.source)
            if lk not in logical_set:
                logical_set.add(lk)
                exported.append({
                    "graph_id": r.graph_id,
                    "src_id": r.src_id,
                    "relation": r.relation,
                    "dst_id": r.dst_id,
                    "source": str(r.source or ""),
                })

    cluster.shutdown()
    return unique_sources, exported, list(logical_set), cass_paths


def neo4j_import_preflight(driver, exported_edges):
    distinct_exported = set()
    for e in exported_edges:
        distinct_exported.add(logical_key(e["graph_id"], e["src_id"], e["relation"], e["dst_id"], e["source"]))

    with driver.session() as session:
        result = session.run(
            "MATCH (s:C0KGNode)-[r:C0KG_EDGE]->(d:C0KGNode) "
            "WHERE r.graph_id = $gid "
            "RETURN s.node_id AS src_id, r.relation AS relation, d.node_id AS dst_id, r.source AS source "
            "ORDER BY src_id, relation, dst_id, source",
            gid=GRAPH_ID,
        )
        neo4j_set = set()
        for row in result:
            neo4j_set.add(logical_key(GRAPH_ID, row["src_id"], row["relation"], row["dst_id"], row["source"] or ""))

    missing_from_neo4j = sorted([list(k) for k in (distinct_exported - neo4j_set)])
    extra_in_neo4j = sorted([list(k) for k in (neo4j_set - distinct_exported)])
    mismatches = len(missing_from_neo4j) + len(extra_in_neo4j)
    return {
        "cassandra_distinct_edges": len(distinct_exported),
        "neo4j_distinct_edges": len(neo4j_set),
        "mismatches": mismatches,
        "all_consistent": mismatches == 0,
        "missing_from_neo4j_count": len(missing_from_neo4j),
        "extra_in_neo4j_count": len(extra_in_neo4j),
    }, missing_from_neo4j, extra_in_neo4j


def import_to_neo4j(driver, exported_edges):
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT c0kg_node_unique IF NOT EXISTS "
            "FOR (n:C0KGNode) REQUIRE (n.graph_id, n.node_id) IS UNIQUE"
        )
    print(f"  Constraint created")

    imported = 0
    batch = []
    for e in exported_edges:
        batch.append(e)
        if len(batch) >= 200:
            _flush_batch(driver, batch)
            imported += len(batch)
            batch = []
            print(f"  Imported {imported}/{len(exported_edges)}...", end="\r")
    if batch:
        _flush_batch(driver, batch)
        imported += len(batch)
    print(f"  Imported {imported}/{len(exported_edges)} edges        ")


def _flush_batch(driver, batch):
    for row in batch:
        cypher = (
            "MERGE (s:C0KGNode {graph_id: $graph_id, node_id: $src_id}) "
            "MERGE (d:C0KGNode {graph_id: $graph_id, node_id: $dst_id}) "
            "CREATE (s)-[r:C0KG_EDGE $props]->(d)"
        )
        with driver.session() as session:
            session.run(cypher, graph_id=row["graph_id"], src_id=row["src_id"],
                        dst_id=row["dst_id"], props={"relation": row["relation"], "source": row["source"] or "", "graph_id": row["graph_id"]})


def neo4j_one_hop(driver, src_id, relation):
    cypher = (
        "MATCH (s:C0KGNode {graph_id: $graph_id, node_id: $src_id})"
        "-[r:C0KG_EDGE {relation: $relation}]->"
        "(d:C0KGNode {graph_id: $graph_id}) "
        "RETURN s.node_id AS src_id, r.relation AS relation, "
        "d.node_id AS dst_id, coalesce(r.source, '') AS source "
        "ORDER BY relation, dst_id, source"
    )
    with driver.session() as session:
        rows = list(session.run(cypher, graph_id=GRAPH_ID, src_id=src_id, relation=relation))
    return [(r["src_id"], r["relation"], r["dst_id"], r["source"]) for r in rows]


def neo4j_traversal(driver):
    frontier = {(SEED, (SEED,), ())}
    for depth, relation in enumerate(RELATION_PATH):
        sources = sorted({s[0] for s in frontier})
        src_edges = {}
        for src in sources:
            edges = neo4j_one_hop(driver, src, relation)
            src_edges[src] = edges

        next_frontier = set()
        for src, node_path, edge_path in frontier:
            edges = src_edges.get(src, [])
            candidates = [e for e in edges if e[2] not in node_path]
            for _, rel, dst, source in candidates[:FANOUT]:
                next_frontier.add((dst, node_path + (dst,), edge_path + (source,)))
        frontier = next_frontier
        if not frontier:
            break
    paths = tuple(sorted({s[2] for s in frontier}))
    return paths, len(paths)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not NEO4J_AVAILABLE:
        print("ERROR: neo4j package not installed. Run: pip install neo4j")
        sys.exit(1)

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_pwd = os.environ.get("NEO4J_PASSWORD")
    if not neo4j_pwd:
        print("ERROR: Set NEO4J_PASSWORD environment variable")
        sys.exit(1)

    # Step 1: Cassandra traversal + export closure
    print("=== Step 1: Cassandra traversal + export closure ===")
    t0 = time.perf_counter()
    unique_sources, exported_edges, logical_keys, cass_paths = export_closure()
    n_exported = len(exported_edges)
    print(f"  Exported: {n_exported} distinct logical edges from {len(unique_sources)} source partitions")
    print(f"  Cassandra result_paths: {len(cass_paths)}")

    with CLOSURE_FILE.open("w", encoding="utf-8") as f:
        for e in exported_edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Saved: {CLOSURE_FILE}")

    # Step 2: Neo4j import
    print("\n=== Step 2: Neo4j import ===")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))

    # Clean existing C0 data
    with driver.session() as session:
        session.run("MATCH (n:C0KGNode) DETACH DELETE n")
    print("  Cleared existing C0KGNode/C0KG_EDGE")

    import_to_neo4j(driver, exported_edges)

    # Step 3: Import preflight
    print("\n=== Step 3: Import preflight ===")
    preflight, missing, extra = neo4j_import_preflight(driver, exported_edges)
    print(f"  Cassandra edges: {preflight['cassandra_distinct_edges']}")
    print(f"  Neo4j edges: {preflight['neo4j_distinct_edges']}")
    print(f"  Mismatches: {preflight['mismatches']}")
    print(f"  All consistent: {preflight['all_consistent']}")

    with (REPORT_DIR / "import_preflight_summary.json").open("w") as f:
        json.dump(preflight, f, indent=2, ensure_ascii=False)

    if missing or extra:
        with (REPORT_DIR / "import_preflight_mismatches.jsonl").open("w") as f:
            for m in missing:
                f.write(json.dumps({"type": "missing_from_neo4j", "key": m}, ensure_ascii=False) + "\n")
            for e in extra:
                f.write(json.dumps({"type": "extra_in_neo4j", "key": e}, ensure_ascii=False) + "\n")
        print("*** PREFLIGHT FAILED ***")
        driver.close()
        return

    # Step 4: Neo4j traversal
    print("\n=== Step 4: Neo4j traversal ===")
    neo4j_paths, n_neo4j = neo4j_traversal(driver)
    print(f"  Neo4j result_paths: {n_neo4j}")

    # Step 5: Compare
    print("\n=== Step 5: Compare ===")
    cass_set = set(cass_paths)
    neo4j_set = set(neo4j_paths)
    missing_paths = sorted(cass_set - neo4j_set)
    extra_paths = sorted(neo4j_set - cass_set)
    disagreements = len(missing_paths) + len(extra_paths)
    match = disagreements == 0
    print(f"  Cassandra paths: {len(cass_set)}")
    print(f"  Neo4j paths: {len(neo4j_set)}")
    print(f"  Disagreements: {disagreements}")
    print(f"  {'MATCH' if match else 'MISMATCH'}")

    if disagreements:
        with (REPORT_DIR / "traversal_mismatches.jsonl").open("w") as f:
            f.write(json.dumps({"missing": [list(p) for p in missing_paths[:50]], "extra": [list(p) for p in extra_paths[:50]]}, ensure_ascii=False) + "\n")

    driver.close()

    summary = {
        "graph_id": GRAPH_ID,
        "seed": SEED,
        "hop": HOP,
        "fanout": FANOUT,
        "relation_path": RELATION_PATH,
        "closure_source_partitions": len(unique_sources),
        "exported_distinct_logical_edges": n_exported,
        "neo4j_distinct_logical_edges": preflight["neo4j_distinct_edges"],
        "cassandra_result_paths": len(cass_paths),
        "neo4j_result_paths": n_neo4j,
        "cassandra_vs_neo4j_disagreements": disagreements,
        "import_preflight_mismatches": preflight["mismatches"],
        "all_pass": match and preflight["all_consistent"],
    }

    with (REPORT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t0
    print(f"\n===== C0-D SUMMARY =====")
    print(f"  Total time: {elapsed:.1f}s")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
