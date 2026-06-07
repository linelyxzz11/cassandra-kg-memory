import argparse
import csv
from pathlib import Path

from neo4j import GraphDatabase


def clear_all(session):
    session.run("MATCH (n) DETACH DELETE n")


def create_constraints(session):
    session.run("CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.node_id IS UNIQUE")


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


def import_edges(session, csv_file, batch_size):
    batch = []
    total = 0

    with Path(csv_file).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            source_with_evidence = f"{row['source']}|{row['evidence']}"
            batch.append({
                "graph_id": row["graph_id"],
                "src_id": row["src_id"],
                "src_type": row["src_type"],
                "relation": row["relation"],
                "dst_id": row["dst_id"],
                "dst_type": row["dst_type"],
                "confidence": float(row["confidence"]),
                "source": source_with_evidence,
            })

            if len(batch) >= batch_size:
                write_batch(session, batch)
                total += len(batch)
                batch.clear()
                if total % 500 == 0:
                    print(f"  {total} edges imported...")

    if batch:
        write_batch(session, batch)
        total += len(batch)

    return total


def main():
    parser = argparse.ArgumentParser(description="Import LoCoMo KG edges into Neo4j.")
    parser.add_argument("--file", default="results/locomo_kg_edges_spacy.csv")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password123")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    with driver.session() as session:
        create_constraints(session)
        if args.clear:
            clear_all(session)
            print("Neo4j database cleared.\n")
        total = import_edges(session, args.file, args.batch_size)
        print(f"\nImported {total} edges into Neo4j.")

    driver.close()


if __name__ == "__main__":
    main()
