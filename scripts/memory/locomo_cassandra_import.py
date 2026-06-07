import argparse
import csv
from pathlib import Path

from cassandra.cluster import Cluster
from cassandra.query import BatchStatement, BatchType


INSERT_SRC = """
INSERT INTO kg_edges_by_src (
    graph_id, src_id, relation, dst_id, edge_id,
    src_type, dst_type, confidence, source, created_at
)
VALUES (?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""

INSERT_DST = """
INSERT INTO kg_edges_by_dst (
    graph_id, dst_id, relation, src_id, edge_id,
    src_type, dst_type, confidence, source, created_at
)
VALUES (?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""

INSERT_RELATION_BUCKET = """
INSERT INTO kg_edges_by_relation_bucket (
    graph_id, relation, bucket, src_id, dst_id, edge_id,
    src_type, dst_type, confidence, source, created_at
)
VALUES (?, ?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""

INSERT_SRC_RELATION = """
INSERT INTO kg_edges_by_src_relation (
    graph_id, src_id, relation, dst_id, edge_id,
    src_type, dst_type, confidence, source, created_at
)
VALUES (?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""

TRUNCATE_TABLES = [
    "TRUNCATE kg_edges_by_src",
    "TRUNCATE kg_edges_by_dst",
    "TRUNCATE kg_edges_by_relation_bucket",
    "TRUNCATE kg_edges_by_src_relation",
]


def stable_bucket(value, bucket_count):
    return abs(hash(value)) % bucket_count


def load_edges(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "graph_id": row["graph_id"],
                "src_id": row["src_id"],
                "src_type": row["src_type"],
                "relation": row["relation"],
                "dst_id": row["dst_id"],
                "dst_type": row["dst_type"],
                "confidence": float(row["confidence"]),
                "source": f"{row['source']}|{row['evidence']}",
            }


def clear_tables(session):
    for stmt in TRUNCATE_TABLES:
        session.execute(stmt)
    print("All four tables truncated.\n")


def insert_edges(session, args):
    insert_src = session.prepare(INSERT_SRC)
    insert_dst = session.prepare(INSERT_DST)
    insert_relation_bucket = session.prepare(INSERT_RELATION_BUCKET)
    insert_src_relation = session.prepare(INSERT_SRC_RELATION)

    logical_edges = 0
    physical_writes = 0
    batch = BatchStatement(batch_type=BatchType.UNLOGGED)

    for edge in load_edges(args.file):
        gid = edge["graph_id"]
        bucket = stable_bucket(edge["dst_id"], args.bucket_count)

        batch.add(insert_src, (
            gid, edge["src_id"], edge["relation"], edge["dst_id"],
            edge["src_type"], edge["dst_type"], edge["confidence"], edge["source"],
        ))
        batch.add(insert_dst, (
            gid, edge["dst_id"], edge["relation"], edge["src_id"],
            edge["src_type"], edge["dst_type"], edge["confidence"], edge["source"],
        ))
        batch.add(insert_relation_bucket, (
            gid, edge["relation"], bucket, edge["src_id"], edge["dst_id"],
            edge["src_type"], edge["dst_type"], edge["confidence"], edge["source"],
        ))
        batch.add(insert_src_relation, (
            gid, edge["src_id"], edge["relation"], edge["dst_id"],
            edge["src_type"], edge["dst_type"], edge["confidence"], edge["source"],
        ))

        logical_edges += 1
        physical_writes += 4

        if logical_edges % args.batch_size == 0:
            session.execute(batch)
            batch.clear()
            if logical_edges % 500 == 0:
                print(f"  {logical_edges} logical edges ({physical_writes} physical writes)...")

    if len(batch) > 0:
        session.execute(batch)

    graph_ids = set()
    for edge in load_edges(args.file):
        graph_ids.add(edge["graph_id"])
    graph_ids = sorted(graph_ids)

    print(f"\nImport completed.")
    print(f"  Graphs imported: {len(graph_ids)} ({', '.join(graph_ids[:5])}{'...' if len(graph_ids) > 5 else ''})")
    print(f"  Logical edges   : {logical_edges}")
    print(f"  Physical writes : {physical_writes} (4x amplification)")


def main():
    parser = argparse.ArgumentParser(description="Import LoCoMo KG edges into Cassandra.")
    parser.add_argument("--file", default="results/locomo_kg_edges_spacy.csv")
    parser.add_argument("--bucket-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--keyspace", default="ai_memory")
    parser.add_argument("--clear", action="store_true", help="Truncate all four tables before import")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9042)
    args = parser.parse_args()

    cluster = Cluster([args.host], port=args.port)
    session = cluster.connect(args.keyspace)

    if args.clear:
        clear_tables(session)

    insert_edges(session, args)
    cluster.shutdown()


if __name__ == "__main__":
    main()
