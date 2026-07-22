import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from cassandra.cluster import Cluster

try:
    from neo4j import GraphDatabase
    NEO4J_OK = True
except ImportError:
    NEO4J_OK = False

SEED = 42
N_ENTITIES = 5000
N_EDGES = 100000
GRAPH_ID = "c1_synth_100k_seed42"
FANOUT = 20
HOP = 2

RELATIONS = ["likes", "suitable_for", "related_to", "suggests", "visited",
             "talked_to", "helped", "works_at", "bought", "reviewed",
             "attended", "planned", "remembered"]

CSV_PATH = Path("D:/memorytable/cassandra-kg-memory/results/c1_source_100k.csv")
REPORT_DIR = Path("D:/memorytable/cassandra-kg-memory/reports/c1_preflight_100k")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def generate_graph():
    rng = random.Random(SEED)
    entities = [f"entity_{i}" for i in range(N_ENTITIES)]
    edges = []
    edges_per = N_EDGES // N_ENTITIES
    generated = 0
    for entity in entities:
        for _ in range(edges_per):
            if generated >= N_EDGES:
                break
            rel = rng.choice(RELATIONS)
            dst = rng.choice(entities)
            sid_val = f"D{rng.randint(1, 100)}:{rng.randint(1, 50)}"
            edges.append({
                "graph_id": GRAPH_ID,
                "src_id": entity,
                "relation": rel,
                "dst_id": dst,
                "source": f"synthetic|{sid_val}",
            })
            generated += 1
    return edges


def write_csv(edges, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["graph_id", "src_id", "relation", "dst_id", "source"])
        w.writeheader()
        w.writerows(edges)


def logical_key(e):
    return (e["graph_id"], e["src_id"], e["relation"], e["dst_id"], e["source"])


def import_cassandra(edges):
    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect("ai_memory")
    insert = session.prepare(
        "INSERT INTO kg_edges_by_src (graph_id, src_id, relation, dst_id, edge_id, src_type, dst_type, confidence, source, created_at) "
        "VALUES (?, ?, ?, ?, now(), 'ENTITY', 'ENTITY', 1.0, ?, toTimestamp(now()))"
    )
    for i, e in enumerate(edges):
        session.execute(insert, (e["graph_id"], e["src_id"], e["relation"], e["dst_id"], e["source"]))
        if (i + 1) % 10000 == 0:
            print(f"  Cassandra: {i + 1}/{len(edges)}...")
    cluster.shutdown()
    print(f"  Cassandra import done: {len(edges)} edges")


def import_neo4j(edges):
    if not NEO4J_OK:
        print("ERROR: neo4j not installed")
        sys.exit(1)
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD")
    if not pwd:
        print("ERROR: NEO4J_PASSWORD not set")
        sys.exit(1)
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as s:
        s.run("MATCH (n:C1KGNode) DETACH DELETE n")
        s.run("CREATE CONSTRAINT c1kg_node_unique IF NOT EXISTS FOR (n:C1KGNode) REQUIRE (n.graph_id, n.node_id) IS UNIQUE")
    print("  Neo4j: cleared + constraint created")

    # Batch nodes
    nodes = set()
    for e in edges:
        nodes.add((e["graph_id"], e["src_id"]))
        nodes.add((e["graph_id"], e["dst_id"]))
    node_list = [{"g": n[0], "nid": n[1]} for n in nodes]
    for i in range(0, len(node_list), 5000):
        chunk = node_list[i:i + 5000]
        with driver.session() as s:
            s.run("UNWIND $rows AS r MERGE (n:C1KGNode {graph_id: r.g, node_id: r.nid})", rows=chunk).consume()
        print(f"  Neo4j nodes: {min(i + 5000, len(node_list))}/{len(node_list)}...")

    # Batch edges using UNWIND
    batch = []
    for e in edges:
        batch.append({
            "g": e["graph_id"],
            "s": e["src_id"],
            "d": e["dst_id"],
            "rel": e["relation"],
            "src": e["source"],
        })
        if len(batch) >= 5000:
            _flush_neo4j_edges(driver, batch)
            batch = []
            print(f"  Neo4j edges: ...")
    if batch:
        _flush_neo4j_edges(driver, batch)

    driver.close()


def _flush_neo4j_edges(driver, batch):
    cypher = (
        "UNWIND $rows AS r "
        "MATCH (s:C1KGNode {graph_id: r.g, node_id: r.s}) "
        "MATCH (d:C1KGNode {graph_id: r.g, node_id: r.d}) "
        "CREATE (s)-[:C1KG_EDGE {relation: r.rel, source: r.src, graph_id: r.g}]->(d)"
    )
    with driver.session() as s:
        s.run(cypher, rows=batch).consume()


def neo4j_full_readback():
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    driver = GraphDatabase.driver(uri, auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]))
    edges = set()
    with driver.session() as s:
        result = s.run(
            "MATCH (s:C1KGNode)-[r:C1KG_EDGE]->(d:C1KGNode) "
            "RETURN r.graph_id AS g, s.node_id AS s_id, r.relation AS rel, d.node_id AS d_id, r.source AS src"
        )
        for row in result:
            edges.add(logical_key({"graph_id": row["g"], "src_id": row["s_id"],
                                   "relation": row["rel"], "dst_id": row["d_id"], "source": row["src"] or ""}))
    driver.close()
    return edges


def main():
    print(f"Generating graph: {N_ENTITIES} entities, {N_EDGES} edges, seed={SEED}")
    edges = generate_graph()
    n_actual = len(edges)
    print(f"  {n_actual} edges generated")

    # Deduplicate by logical key
    seen = set()
    dedupe = []
    for e in edges:
        lk = logical_key(e)
        if lk not in seen:
            seen.add(lk)
            dedupe.append(e)
    n_distinct = len(dedupe)
    print(f"  {n_distinct} distinct logical edges ({n_actual - n_distinct} duplicates removed)")
    edges = dedupe

    write_csv(edges, CSV_PATH)
    print(f"  Source CSV: {CSV_PATH}")

    # Dataset summary
    src_counts = defaultdict(int)
    for e in edges:
        src_counts[e["src_id"]] += 1
    degrees = list(src_counts.values())

    dataset = {
        "graph_id": GRAPH_ID,
        "seed": SEED,
        "n_entities": N_ENTITIES,
        "n_edges_generated": n_actual,
        "n_edges_distinct": n_distinct,
        "min_outdegree": min(degrees), "max_outdegree": max(degrees),
        "p50_outdegree": sorted(degrees)[len(degrees) // 2],
        "source_csv": str(CSV_PATH.resolve()),
    }
    with (REPORT_DIR / "dataset_summary.json").open("w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(json.dumps(dataset, indent=2, ensure_ascii=False))

    # Import to Cassandra
    print("\nImporting to Cassandra...")
    import_cassandra(edges)
    print("  Cassandra import done")

    # Import to Neo4j
    print("\nImporting to Neo4j...")
    import_neo4j(edges)
    print("  Neo4j import done")

    # Full-graph edge-set preflight
    print("\nFull-graph import preflight...")
    neo4j_set = neo4j_full_readback()
    csv_set = {logical_key(e) for e in edges}
    missing = sorted(csv_set - neo4j_set)
    extra = sorted(neo4j_set - csv_set)
    mismatches = len(missing) + len(extra)
    preflight = {
        "cassandra_csv_edges": len(csv_set),
        "neo4j_edges": len(neo4j_set),
        "mismatches": mismatches,
        "all_consistent": mismatches == 0,
    }
    with (REPORT_DIR / "full_graph_import_preflight.json").open("w") as f:
        json.dump(preflight, f, indent=2, ensure_ascii=False)
    if mismatches:
        with (REPORT_DIR / "full_graph_import_mismatches.jsonl").open("w") as f:
            for m in missing[:50]:
                f.write(json.dumps({"type": "missing_from_neo4j", "key": list(m)}) + "\n")
            for e in extra[:50]:
                f.write(json.dumps({"type": "extra_in_neo4j", "key": list(e)}) + "\n")
    print(json.dumps(preflight, indent=2))
    print(f"  {'PASS' if mismatches == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
