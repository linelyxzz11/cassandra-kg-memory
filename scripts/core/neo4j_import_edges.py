import argparse
import csv
from pathlib import Path

from neo4j import GraphDatabase


def clear_database(session):
    session.run("MATCH (n) DETACH DELETE n")


def create_constraints(session):
    session.run("""
    CREATE CONSTRAINT kg_node_id IF NOT EXISTS
    FOR (n:KGNode)
    REQUIRE n.node_id IS UNIQUE
    """)


def write_batch(session, rows):
    session.run(
        """
        UNWIND $rows AS row
        MERGE (s:KGNode {node_id: row.src_id})
        SET s.node_type = row.src_type
        MERGE (d:KGNode {node_id: row.dst_id})
        SET d.node_type = row.dst_type
        MERGE (s)-[r:KG_EDGE {
            graph_id: row.graph_id,
            relation: row.relation,
            dst_id: row.dst_id
        }]->(d)
        SET r.confidence = row.confidence,
            r.source = row.source
        """,
        rows=rows,
    )


def import_edges(session, csv_file, graph_id, batch_size):
    batch = []

    with Path(csv_file).open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            batch.append({
                "graph_id": graph_id,
                "src_id": row["src_id"],
                "src_type": row["src_type"],
                "relation": row["relation"],
                "dst_id": row["dst_id"],
                "dst_type": row["dst_type"],
                "confidence": float(row["confidence"]),
                "source": row["source"],
            })

            if len(batch) >= batch_size:
                write_batch(session, batch)
                batch.clear()

        if batch:
            write_batch(session, batch)


def main():
    parser = argparse.ArgumentParser(
        description="Import KG edge CSV into Neo4j."
    )

    parser.add_argument("--file", required=True)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password123")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--clear", action="store_true")

    args = parser.parse_args()

    driver = GraphDatabase.driver(
        args.uri,
        auth=(args.user, args.password),
    )

    with driver.session() as session:
        if args.clear:
            clear_database(session)

        create_constraints(session)
        import_edges(session, args.file, args.graph_id, args.batch_size)

    driver.close()

    print("Neo4j import completed.")
    print(f"Input file: {args.file}")
    print(f"Graph id: {args.graph_id}")


if __name__ == "__main__":
    main()